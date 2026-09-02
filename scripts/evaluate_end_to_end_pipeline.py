"""Full detect -> classify -> segment pipeline, image in, final instances
out: not something this project had chained together end-to-end before --
every prior number is one stage in isolation (Task B's 0.889 accuracy uses
oracle ground-truth boxes/masks; the Mask R-CNN's own class head, per
docs/error_analysis.md, has the same visually-similar-instrument confusion
as the box detector). Requested directly, 2026-09-02: what does the actual
pipeline cost and score when the stages are wired together for real.

Pipeline: `maskrcnn_mobilenet_v3` (the official box-then-mask checkpoint,
AP50_segm 0.8101) gives boxes + masks in one forward pass; its own class
head is DISCARDED and replaced by the Task B ensemble
(region_baseline + region_letterbox_crop, macro-F1 0.848 on oracle boxes)
applied to each detected box+mask crop, since the ensemble is the more
accurate classifier of the two per the error analysis. This is the
concrete form of the "detect, classify, segment" pipeline the project's
components support today, without training anything new.

Latency: single frame at native 800x1280 (CLAUDE.md's detection
convention, no resize), warmed up, `torch.cuda.synchronize()`'d, timed as
one unit (Mask R-CNN forward pass + the full per-instance classification
loop, since that loop is a real, variable-count cost of running this
pipeline on a frame), median/p95 over real official-test frames (not a
fixed dummy tensor, since instance count varies frame to frame and that
variability is part of the real cost).

Accuracy: matches each kept detection (score >= --score-thresh) to ground
truth by IoU, then reports classification accuracy on the matched (TP)
instances -- the genuinely new number this script produces, since it's
scored against the DETECTOR's real boxes, not Task B's oracle ones.
Localization/mask quality itself is not re-measured here; the official
Mask R-CNN's already-reported AP50_segm (0.8101) covers that.

Usage:
    python scripts/evaluate_end_to_end_pipeline.py \\
        --maskrcnn-checkpoint experiments/instance_segmentation_maskrcnn_official_20260901-180758/best.pt \\
        --classifier-a experiments/region_baseline_20260831-182451/best.pt \\
        --classifier-b experiments/region_letterbox_crop_20260902-152750/best.pt \\
        --num-frames 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from surgical_ai.data.detection_dataset import GraspDetectionDataset, build_detection_transforms
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model
from surgical_ai.models.detectors.registry import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maskrcnn-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-a", type=Path, required=True)
    parser.add_argument("--classifier-b", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--num-warmup", type=int, default=10)
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
def run_pipeline_once(
    frame_np: np.ndarray, image_tensor: torch.Tensor, maskrcnn, clf_a, clf_b,
    transform_a, transform_b, device: torch.device, score_thresh: float,
):
    """One frame through the full pipeline. Returns (detections, timings_ms)."""
    height, width = frame_np.shape[:2]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    output = maskrcnn([image_tensor.to(device)])[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_detect = time.perf_counter()

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks = output["masks"].cpu().numpy()  # [N, 1, H, W], float probs

    detections = []
    for box, score, mask in zip(boxes, scores, masks):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
        x2i, y2i = min(width, int(round(x2))), min(height, int(round(y2)))
        if x2i <= x1i or y2i <= y1i:
            continue
        binary_mask = (mask[0] >= 0.5)
        crop_raw = frame_np[y1i:y2i, x1i:x2i]
        crop_mask = binary_mask[y1i:y2i, x1i:x2i]
        masked_crop = (crop_raw * crop_mask[:, :, None]).astype(np.uint8)

        img_a = Image.fromarray(masked_crop)
        img_b = Image.fromarray(pad_to_square(masked_crop))
        tensor_a = transform_a(img_a).unsqueeze(0).to(device)
        tensor_b = transform_b(img_b).unsqueeze(0).to(device)

        probs_a = torch.softmax(clf_a(tensor_a), dim=1)
        probs_b = torch.softmax(clf_b(tensor_b), dim=1)
        probs = ((probs_a + probs_b) / 2).cpu().numpy()[0]
        detections.append({
            "box": [x1, y1, x2, y2], "score": float(score),
            "class_idx": int(probs.argmax()), "class_probs": probs,
        })

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_classify = time.perf_counter()

    timings = {
        "detect_segment_ms": (t_detect - t0) * 1000,
        "classify_ms": (t_classify - t_detect) * 1000,
        "total_ms": (t_classify - t0) * 1000,
        "n_instances": len(detections),
    }
    return detections, timings


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    ds = GraspDetectionDataset(args.data_root, args.split, transform=build_detection_transforms(train=False))
    class_names = ds.class_names_ordered()

    maskrcnn = build_detector("maskrcnn_mobilenet_v3", num_classes=args.num_classes, pretrained=False).to(device)
    maskrcnn.load_state_dict(torch.load(args.maskrcnn_checkpoint, map_location=device))
    maskrcnn.eval()

    clf_a = build_model("mobilenet_v3_small", num_classes=args.num_classes, pretrained=False, freeze_backbone=False).to(device)
    clf_a.load_state_dict(torch.load(args.classifier_a, map_location=device), strict=False)
    clf_a.eval()
    clf_b = build_model("mobilenet_v3_small", num_classes=args.num_classes, pretrained=False, freeze_backbone=False).to(device)
    clf_b.load_state_dict(torch.load(args.classifier_b, map_location=device), strict=False)
    clf_b.eval()

    transform_a = build_transforms(args.image_size, train=False)
    transform_b = build_transforms(args.image_size, train=False)

    n_frames = min(args.num_frames, len(ds.samples))
    frame_indices = list(range(n_frames))

    print(f"warming up ({args.num_warmup} frames)...")
    for i in frame_indices[: args.num_warmup]:
        file_name, _anns = ds.samples[i]
        frame_np = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        image_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
        run_pipeline_once(frame_np, image_tensor, maskrcnn, clf_a, clf_b, transform_a, transform_b, device, args.score_thresh)

    print(f"timing {n_frames} real frames...")
    all_timings = []
    n_tp, n_correct = 0, 0
    for i in frame_indices:
        file_name, anns = ds.samples[i]
        frame_np = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        image_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0

        detections, timings = run_pipeline_once(
            frame_np, image_tensor, maskrcnn, clf_a, clf_b, transform_a, transform_b, device, args.score_thresh
        )
        all_timings.append(timings)

        gt = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            gt.append({"box": [x, y, x + w, y + h], "label": ds._id_to_index[a["category_id"]], "matched": False})

        for det in sorted(detections, key=lambda d: -d["score"]):
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

    detect_ms = np.array([t["detect_segment_ms"] for t in all_timings])
    classify_ms = np.array([t["classify_ms"] for t in all_timings])
    total_ms = np.array([t["total_ms"] for t in all_timings])
    n_inst = np.array([t["n_instances"] for t in all_timings])

    print(f"\n=== Latency over {n_frames} real official-{args.split} frames (native 800x1280, Titan Xp) ===")
    print(f"detect+segment (Mask R-CNN forward pass): median={np.median(detect_ms):.2f}ms p95={np.percentile(detect_ms,95):.2f}ms")
    print(f"classification (per-frame total, all instances): median={np.median(classify_ms):.2f}ms p95={np.percentile(classify_ms,95):.2f}ms")
    print(f"TOTAL end-to-end per frame: median={np.median(total_ms):.2f}ms p95={np.percentile(total_ms,95):.2f}ms")
    print(f"mean instances/frame: {n_inst.mean():.2f} (min={n_inst.min()}, max={n_inst.max()})")
    print(f"classification cost per instance (classify_ms / n_instances, frames with >=1 instance): "
          f"{np.mean(classify_ms[n_inst>0] / n_inst[n_inst>0]):.2f}ms")

    print(f"\n=== Classification accuracy on real (non-oracle) detected+matched instances ===")
    print(f"TP-matched instances (IoU>={args.iou_thresh}, score>={args.score_thresh}): {n_tp}")
    print(f"classification accuracy on those TP instances: {n_correct/n_tp:.4f}" if n_tp else "no TP instances found")
    print("(compare to Task B's oracle-box ensemble accuracy of 0.889 -- this number uses the "
          "detector's own real boxes/masks instead of ground truth)")


if __name__ == "__main__":
    main()
