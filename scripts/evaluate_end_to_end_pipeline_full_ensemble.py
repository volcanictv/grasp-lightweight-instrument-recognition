"""Full end-to-end pipeline using this project's actual BEST segmentation
result (the 4-way ensemble, AP50_segm 0.8594 -- three Mask R-CNN
checkpoints + fine-tuned SAM2, fused via weighted_fusion_merge) instead of
the single official MobileNetV3 model used in evaluate_end_to_end_pipeline.py.
Built directly on request, 2026-09-02: that first run used a weaker
segmentation component than the project's best, which needed a stated
reason, not a silent substitution.

Same idea as the other script: the ensemble's own fused class label
(inherited from whichever cluster member scored highest -- see
inference/ensemble.py, not a real class vote) is discarded and replaced by
the Task B ensemble (region_baseline + region_letterbox_crop) applied to
each fused detection's box+mask crop.

Usage:
    python scripts/evaluate_end_to_end_pipeline_full_ensemble.py \\
        --maskrcnn experiments/instance_segmentation_maskrcnn_official_20260901-180758/best.pt:maskrcnn_mobilenet_v3 \\
        --maskrcnn experiments/instance_segmentation_maskrcnn_resnet50_coco_20260901-233358/best.pt:maskrcnn_resnet50_coco \\
        --maskrcnn experiments/instance_segmentation_maskrcnn_official_scratch_20260901-211934/best.pt:maskrcnn_mobilenet_v3 \\
        --sam2-box-detector experiments/detection_weighted_loss_20260901-001115/best.pt \\
        --sam2-decoder-checkpoint experiments/sam2_decoder_finetune_20260901-235635/decoder_best.pt \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \\
        --classifier-a experiments/region_baseline_20260831-182451/best.pt \\
        --classifier-b experiments/region_letterbox_crop_20260902-152750/best.pt \\
        --num-frames 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from surgical_ai.data import splits
from surgical_ai.data.detection_dataset import GraspDetectionDataset, build_detection_transforms
from surgical_ai.data.transforms import build_transforms
from surgical_ai.inference.ensemble import weighted_fusion_merge
from surgical_ai.models import build_model
from surgical_ai.models.detectors.registry import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maskrcnn", action="append", required=True, dest="maskrcnn_specs", help="checkpoint_path:registry_name, repeatable")
    parser.add_argument("--sam2-box-detector", type=Path, required=True)
    parser.add_argument("--sam2-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--classifier-a", type=Path, required=True)
    parser.add_argument("--classifier-b", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--maskrcnn-score-threshold", type=float, default=0.05)
    parser.add_argument("--sam2-score-threshold", type=float, default=0.5)
    parser.add_argument("--fusion-iou-threshold", type=float, default=0.5)
    parser.add_argument("--final-score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=100)
    return parser.parse_args()


def pad_to_square(crop: np.ndarray) -> np.ndarray:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - ch) // 2, (side - cw) // 2
    square[top : top + ch, left : left + cw] = crop
    return square


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def collect_maskrcnn_detections(model, image_tensor: torch.Tensor, score_threshold: float) -> list[dict]:
    output = model([image_tensor])[0]
    boxes = output["boxes"].cpu().tolist()
    labels = output["labels"].cpu().tolist()
    scores = output["scores"].cpu().tolist()
    masks = output["masks"].cpu().numpy()[:, 0]
    return [
        {"box": box, "label": label - 1, "score": score, "mask": mask}
        for box, label, score, mask in zip(boxes, labels, scores, masks)
        if score >= score_threshold
    ]


def collect_sam2_detections(sam2_predictor, box_detector, image_np: np.ndarray, image_tensor: torch.Tensor, score_threshold: float) -> list[dict]:
    with torch.no_grad():
        det_out = box_detector([image_tensor])[0]
    boxes = det_out["boxes"].cpu().numpy()
    labels = det_out["labels"].cpu().numpy()
    scores = det_out["scores"].cpu().numpy()
    keep = scores >= score_threshold

    dets = []
    if keep.any():
        with torch.no_grad():
            sam2_predictor.set_image(image_np)
            for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                mask_input, _uc, _l, unnorm_box = sam2_predictor._prep_prompts(None, None, box, None, True)
                pred_masks, _iou, _lr = sam2_predictor._predict(
                    None, None, unnorm_box, mask_input, multimask_output=False, return_logits=True
                )
                prob = torch.sigmoid(pred_masks[0, 0]).cpu().numpy()
                dets.append({"box": box.tolist(), "label": int(label) - 1, "score": float(score), "mask": prob})
    return dets


@torch.no_grad()
def classify_fused_detections(fused: list[dict], frame_np: np.ndarray, clf_a, clf_b, transform_a, transform_b, device, final_score_threshold: float):
    height, width = frame_np.shape[:2]
    results = []
    for d in fused:
        if d["score"] < final_score_threshold:
            continue
        x1, y1, x2, y2 = d["box"]
        x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
        x2i, y2i = min(width, int(round(x2))), min(height, int(round(y2)))
        if x2i <= x1i or y2i <= y1i:
            continue
        binary_mask = d["mask"] >= 0.5
        crop_raw = frame_np[y1i:y2i, x1i:x2i]
        crop_mask = binary_mask[y1i:y2i, x1i:x2i]
        masked_crop = (crop_raw * crop_mask[:, :, None]).astype(np.uint8)

        tensor_a = transform_a(Image.fromarray(masked_crop)).unsqueeze(0).to(device)
        tensor_b = transform_b(Image.fromarray(pad_to_square(masked_crop))).unsqueeze(0).to(device)
        probs_a = torch.softmax(clf_a(tensor_a), dim=1)
        probs_b = torch.softmax(clf_b(tensor_b), dim=1)
        probs = ((probs_a + probs_b) / 2).cpu().numpy()[0]
        results.append({"box": d["box"], "score": d["score"], "class_idx": int(probs.argmax())})
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _, val_split = splits.resolve_train_val_split("official")
    val_ds = GraspDetectionDataset(args.data_root, val_split, include_masks=True, transform=build_detection_transforms(train=False))

    maskrcnn_models = []
    for spec in args.maskrcnn_specs:
        checkpoint_path, registry_name = spec.rsplit(":", 1)
        model = build_detector(registry_name, num_classes=len(val_ds.class_names_ordered()), pretrained=False).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        maskrcnn_models.append(model)

    box_detector = build_detector("fasterrcnn_mobilenet_v3", num_classes=len(val_ds.class_names_ordered()), pretrained=False).to(device)
    box_detector.load_state_dict(torch.load(args.sam2_box_detector, map_location=device))
    box_detector.eval()

    sam2_predictor = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=str(device)))
    sam2_predictor.model.sam_mask_decoder.load_state_dict(torch.load(args.sam2_decoder_checkpoint, map_location=device))
    sam2_predictor.model.sam_mask_decoder.eval()

    clf_a = build_model("mobilenet_v3_small", num_classes=args.num_classes, pretrained=False, freeze_backbone=False).to(device)
    clf_a.load_state_dict(torch.load(args.classifier_a, map_location=device), strict=False)
    clf_a.eval()
    clf_b = build_model("mobilenet_v3_small", num_classes=args.num_classes, pretrained=False, freeze_backbone=False).to(device)
    clf_b.load_state_dict(torch.load(args.classifier_b, map_location=device), strict=False)
    clf_b.eval()
    transform_a = build_transforms(args.image_size, train=False)
    transform_b = build_transforms(args.image_size, train=False)

    to_tensor = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def run_once(idx: int):
        file_name, anns = val_ds.samples[idx]
        image = Image.open(val_ds.frames_root / file_name).convert("RGB")
        image_np = np.array(image)
        image_tensor = to_tensor(image).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        dets_all = [collect_maskrcnn_detections(m, image_tensor, args.maskrcnn_score_threshold) for m in maskrcnn_models]
        dets_all.append(collect_sam2_detections(sam2_predictor, box_detector, image_np, image_tensor, args.sam2_score_threshold))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_detect = time.perf_counter()

        fused = weighted_fusion_merge(dets_all, iou_threshold=args.fusion_iou_threshold)
        final_dets = classify_fused_detections(fused, image_np, clf_a, clf_b, transform_a, transform_b, device, args.final_score_threshold)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_classify = time.perf_counter()

        timings = {
            "detect_segment_ms": (t_detect - t0) * 1000,
            "classify_ms": (t_classify - t_detect) * 1000,
            "total_ms": (t_classify - t0) * 1000,
            "n_instances": len(final_dets),
        }
        return final_dets, anns, timings

    print(f"warming up ({args.num_warmup} frames)...")
    for i in range(args.num_warmup):
        run_once(i)

    n_frames = min(args.num_frames, len(val_ds.samples))
    print(f"timing {n_frames} real frames (this is slow -- SAM2's image encoder alone is ~400ms/frame)...")
    all_timings = []
    n_tp, n_correct = 0, 0
    for i in range(n_frames):
        final_dets, anns, timings = run_once(i)
        all_timings.append(timings)

        gt = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            gt.append({"box": [x, y, x + w, y + h], "label": val_ds._id_to_index[a["category_id"]], "matched": False})

        for det in sorted(final_dets, key=lambda d: -d["score"]):
            best_gt, best_iou = None, 0.0
            for g in gt:
                if g["matched"]:
                    continue
                iou = box_iou(det["box"], g["box"])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
            if best_gt is not None and best_iou >= args.iou_thresh:
                best_gt["matched"] = True
                n_tp += 1
                if det["class_idx"] == best_gt["label"]:
                    n_correct += 1

        if i % 20 == 0:
            print(f"  {i}/{n_frames}", flush=True)

    detect_ms = np.array([t["detect_segment_ms"] for t in all_timings])
    classify_ms = np.array([t["classify_ms"] for t in all_timings])
    total_ms = np.array([t["total_ms"] for t in all_timings])
    n_inst = np.array([t["n_instances"] for t in all_timings])

    print(f"\n=== Latency over {n_frames} real official-{val_split} frames (native 800x1280, Titan Xp, FULL 4-way ensemble) ===")
    print(f"detect+segment (3 Mask R-CNN + SAM2, fused): median={np.median(detect_ms):.2f}ms p95={np.percentile(detect_ms,95):.2f}ms")
    print(f"classification (all instances): median={np.median(classify_ms):.2f}ms p95={np.percentile(classify_ms,95):.2f}ms")
    print(f"TOTAL end-to-end per frame: median={np.median(total_ms):.2f}ms p95={np.percentile(total_ms,95):.2f}ms")
    print(f"mean instances/frame: {n_inst.mean():.2f} (min={n_inst.min()}, max={n_inst.max()})")

    print(f"\n=== Classification accuracy on real (non-oracle) detected+matched instances ===")
    print(f"TP-matched instances (IoU>={args.iou_thresh}, score>={args.final_score_threshold}): {n_tp}")
    print(f"classification accuracy on those TP instances: {n_correct/n_tp:.4f}" if n_tp else "no TP instances found")


if __name__ == "__main__":
    main()
