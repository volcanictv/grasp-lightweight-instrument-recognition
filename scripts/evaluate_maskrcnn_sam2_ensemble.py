"""Four-way ensemble: the three Mask R-CNN checkpoints already ensembled
in scripts/evaluate_maskrcnn_ensemble.py (0.8553 AP50_segm) plus the
fine-tuned SAM2 decoder (box-prompted by the same fasterrcnn_mobilenet_v3
detector used throughout this project, docs/DECISIONS.md 2026-09-02) as a
fourth member. SAM2 was the weakest individual model (0.7966), but the
three-way ensemble already showed a weaker model can still improve the
ensemble via error diversity -- testing whether that holds a second time.

Kept as a separate script rather than folded into evaluate_maskrcnn_
ensemble.py because SAM2 has a fundamentally different inference
interface (image encoder + box-prompted decoder) than the torchvision
Mask R-CNN models there; forcing both through one code path would
obscure both. Applies the same OOM lesson from that script: only the
ensemble's predictions are accumulated for the full test set, not every
individual model's.

Usage:
    python scripts/evaluate_maskrcnn_sam2_ensemble.py \\
        --maskrcnn experiments/instance_segmentation_maskrcnn_official_20260901-180758/best.pt:maskrcnn_mobilenet_v3 \\
        --maskrcnn experiments/instance_segmentation_maskrcnn_resnet50_coco_20260901-233358/best.pt:maskrcnn_resnet50_coco \\
        --maskrcnn experiments/instance_segmentation_maskrcnn_official_scratch_20260901-211934/best.pt:maskrcnn_mobilenet_v3 \\
        --sam2-box-detector experiments/detection_weighted_loss_20260901-001115/best.pt \\
        --sam2-decoder-checkpoint experiments/sam2_decoder_finetune_20260901-235635/decoder_best.pt \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \\
        --data-root ./GraSP --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data import splits  # noqa: E402
from surgical_ai.data.detection_dataset import GraspDetectionDataset, build_detection_transforms, collate_fn  # noqa: E402
from surgical_ai.data.mask_utils import decode_instance_mask  # noqa: E402
from surgical_ai.evaluation.detection import compute_occlusion_fractions  # noqa: E402
from surgical_ai.evaluation.segmentation import evaluate_instance_ap50, mask_iou  # noqa: E402
from surgical_ai.inference.ensemble import weighted_fusion_merge  # noqa: E402
from surgical_ai.models.detectors.registry import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maskrcnn", action="append", required=True, dest="maskrcnn_specs", help="checkpoint_path:registry_name, repeatable.")
    parser.add_argument("--sam2-box-detector", type=Path, required=True)
    parser.add_argument("--sam2-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--sam2-score-threshold", type=float, default=0.5, help="SAM2's boxes come from the detector -- use its normal decision threshold.")
    parser.add_argument("--fusion-iou-threshold", type=float, default=0.5)
    parser.add_argument("--recall-score-threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_maskrcnn_detections(model: torch.nn.Module, image_tensor: torch.Tensor, score_threshold: float) -> list[dict]:
    output = model([image_tensor])[0]
    boxes = output["boxes"].cpu().tolist()
    labels = output["labels"].cpu().tolist()
    scores = output["scores"].cpu().tolist()
    masks = output["masks"].cpu().numpy()[:, 0]
    dets = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < score_threshold:
            continue
        dets.append({"box": box, "label": label - 1, "score": score, "mask": mask})
    return dets


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


def evaluate_predictions(
    predictions: list[list[tuple]], gt_instances: list[list[tuple]], val_ds, occlusion_fractions: dict,
    class_names: list[str], recall_score_threshold: float, label: str,
) -> None:
    ap50 = evaluate_instance_ap50(predictions, gt_instances, class_names)
    counts = {"isolated": 0, "light": 0, "heavy": 0}
    hits = {"isolated": 0, "light": 0, "heavy": 0}
    for image_idx in range(len(val_ds.samples)):
        _file_name, anns = val_ds.samples[image_idx]
        preds = [(m, lbl, s) for m, lbl, s in predictions[image_idx] if s >= recall_score_threshold]
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if not gt_mask.any():
                continue
            gt_label = val_ds._id_to_index[a["category_id"]]
            frac = occlusion_fractions.get(a["id"], 0.0)
            bucket = "isolated" if frac <= 0.0 else ("heavy" if frac > 0.5 else "light")
            counts[bucket] += 1
            if any(lbl == gt_label and mask_iou(gt_mask, m) >= 0.5 for m, lbl, _s in preds):
                hits[bucket] += 1
    recall_str = ", ".join(f"{b}={hits[b] / counts[b] if counts[b] else float('nan'):.3f}" for b in ("isolated", "light", "heavy"))
    print(f"[{label}] AP50_segm={ap50['map50']:.4f}  occlusion recall: {recall_str}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _, val_split = splits.resolve_train_val_split("official")
    val_ds = GraspDetectionDataset(args.data_root, val_split, include_masks=True, transform=build_detection_transforms(train=False))
    class_names = val_ds.class_names_ordered()
    occlusion_fractions = compute_occlusion_fractions(val_ds)

    maskrcnn_models = []
    for spec in args.maskrcnn_specs:
        checkpoint_path, registry_name = spec.rsplit(":", 1)
        model = build_detector(registry_name, num_classes=len(class_names), pretrained=False).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        maskrcnn_models.append(model)

    box_detector = build_detector("fasterrcnn_mobilenet_v3", num_classes=len(class_names), pretrained=False).to(device)
    box_detector.load_state_dict(torch.load(args.sam2_box_detector, map_location=device))
    box_detector.eval()

    sam2_predictor = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=str(device)))
    sam2_predictor.model.sam_mask_decoder.load_state_dict(torch.load(args.sam2_decoder_checkpoint, map_location=device))
    sam2_predictor.model.sam_mask_decoder.eval()

    n_samples = len(val_ds.samples) if args.limit is None else min(args.limit, len(val_ds.samples))
    preds_ensemble: list[list[tuple]] = []
    gt_instances: list[list[tuple]] = []

    to_tensor = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    for idx in range(n_samples):
        file_name, anns = val_ds.samples[idx]
        image = Image.open(val_ds.frames_root / file_name).convert("RGB")
        image_np = np.array(image)
        image_tensor = to_tensor(image).to(device)

        dets_all = [collect_maskrcnn_detections(m, image_tensor, args.score_threshold) for m in maskrcnn_models]
        dets_all.append(collect_sam2_detections(sam2_predictor, box_detector, image_np, image_tensor, args.sam2_score_threshold))

        fused = weighted_fusion_merge(dets_all, iou_threshold=args.fusion_iou_threshold)
        preds_ensemble.append([(d["mask"] >= 0.5, d["label"], d["score"]) for d in fused])

        gts = []
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if gt_mask.any():
                gts.append((gt_mask, val_ds._id_to_index[a["category_id"]]))
        gt_instances.append(gts)

        if idx % 50 == 0:
            print(f"{idx}/{n_samples}", flush=True)

    for _ in range(len(val_ds.samples) - n_samples):
        preds_ensemble.append([])
        gt_instances.append([])

    evaluate_predictions(preds_ensemble, gt_instances, val_ds, occlusion_fractions, class_names, args.recall_score_threshold, "4-way ensemble (3 Mask R-CNN + SAM2)")


if __name__ == "__main__":
    main()
