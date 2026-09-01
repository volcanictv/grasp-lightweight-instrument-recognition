"""Evaluate tracking-by-detection's effect on the Milestone 9 segmenter's
instance-level metrics (AP50_segm, occlusion-stratified recall) -- the
segmentation analog of `evaluate_tracking.py`, same question asked of
Milestone 8's detector: does letting a track coast through a few frames of
missed detection recover instances lost to occlusion.

Does NOT touch semantic mIoU. That number comes from the semantic head's
own per-pixel argmax, decoded independently at the target frame -- tracking
only ever changes which *instances* are reported, so it cannot move mIoU
whether it helps, hurts, or does nothing here.

Usage:
    python scripts/evaluate_tracking_segm.py configs/segmentation_deep_backbone_copy_paste.yaml \\
        experiments/segmentation_deep_backbone_copy_paste_.../best.pt [--data-root PATH] \\
        [--device cuda:0] [--window-radius 5] [--max-age 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data import splits  # noqa: E402
from surgical_ai.data.mask_utils import decode_instance_mask  # noqa: E402
from surgical_ai.data.segmentation_dataset import GraspSegmentationDataset  # noqa: E402
from surgical_ai.data.segmentation_targets import downsample_mask_nearest  # noqa: E402
from surgical_ai.evaluation.detection import compute_occlusion_fractions  # noqa: E402
from surgical_ai.evaluation.segmentation import (  # noqa: E402
    evaluate_instance_ap50,
    evaluate_occlusion_stratified_recall_segm,
)
from surgical_ai.inference.pipeline import build_frame_window, parse_case_and_frame_number, run_window_and_track_segm  # noqa: E402
from surgical_ai.models.segmenters.registry import build_segmenter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSp")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-radius", type=int, default=5)
    parser.add_argument("--max-age", type=int, default=1, help="Best-tuned value found for the detector, Milestone 8.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Tracker's association IoU threshold.")
    parser.add_argument("--min-confidence-to-coast", type=float, default=0.5)
    parser.add_argument("--boundary-margin-frac", type=float, default=0.05)
    parser.add_argument("--occlusion-corridor-iou-threshold", type=float, default=0.0)
    parser.add_argument("--occluded-max-age", type=int, default=5)
    parser.add_argument("--require-continuous-occlusion-evidence", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Matches decode_instances' own default.")
    parser.add_argument("--recall-score-threshold", type=float, default=0.3)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N val frames (debugging).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)

    image_size = config["data"].get("image_size", 384)
    output_stride = config["data"].get("output_stride", 4)
    out_hw = image_size // output_stride

    _, val_split = splits.resolve_train_val_split(config["data"]["split"])
    val_ds = GraspSegmentationDataset(args.data_root, val_split, image_size=image_size, output_stride=output_stride)
    class_names = val_ds.class_names_ordered()
    frames_root = val_ds.frames_root

    model = build_segmenter(config["model"]["name"], num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    occlusion_fractions = compute_occlusion_fractions(val_ds)

    # evaluate_occlusion_stratified_recall_segm indexes predictions positionally
    # against the *full* dataset.samples, so predictions must stay full-length
    # even under --limit -- untested frames are padded with empty predictions
    # (their real GT instances then correctly count as misses, not silently
    # dropped) rather than truncating the lists.
    n_samples = len(val_ds.samples) if args.limit is None else min(args.limit, len(val_ds.samples))
    direct_predictions: list[list] = []
    tracked_predictions: list[list] = []
    gt_instances: list[list] = []

    for image_idx in range(len(val_ds.samples)):
        file_name, anns = val_ds.samples[image_idx]
        if image_idx >= n_samples:
            direct_predictions.append([])
            tracked_predictions.append([])
            gt_instances.append([])
            continue

        case, frame_number = parse_case_and_frame_number(file_name)
        window = build_frame_window(frames_root, case, frame_number, args.window_radius, causal=True)
        if not any(n == frame_number for n, _ in window):
            direct_predictions.append([])
            tracked_predictions.append([])
            gt_instances.append([])
            continue

        direct, tracked = run_window_and_track_segm(
            model, device, window, frame_number,
            image_size=image_size, output_stride=output_stride,
            score_threshold=args.score_threshold, iou_threshold=args.iou_threshold, max_age=args.max_age,
            min_confidence_to_coast=args.min_confidence_to_coast, boundary_margin_frac=args.boundary_margin_frac,
            occlusion_corridor_iou_threshold=args.occlusion_corridor_iou_threshold,
            occluded_max_age=args.occluded_max_age,
            require_continuous_occlusion_evidence=args.require_continuous_occlusion_evidence,
        )
        direct_predictions.append(direct)
        tracked_predictions.append(tracked)

        gts = []
        for a in anns:
            native_mask = decode_instance_mask(a["segmentation"])
            resized = np.array(Image.fromarray(native_mask * 255).resize((image_size, image_size), Image.NEAREST)) > 0
            if not resized.any():
                continue
            small = downsample_mask_nearest(resized, output_stride, out_hw, out_hw)
            gts.append((small, val_ds._id_to_index[a["category_id"]]))
        gt_instances.append(gts)

        if image_idx % 100 == 0:
            print(f"{image_idx}/{n_samples}", flush=True)

    direct_ap50 = evaluate_instance_ap50(direct_predictions, gt_instances, class_names)
    tracked_ap50 = evaluate_instance_ap50(tracked_predictions, gt_instances, class_names)

    direct_recall = evaluate_occlusion_stratified_recall_segm(
        val_ds, direct_predictions, occlusion_fractions, output_stride,
        score_threshold=args.recall_score_threshold, iou_threshold=args.eval_iou_threshold,
    )
    tracked_recall = evaluate_occlusion_stratified_recall_segm(
        val_ds, tracked_predictions, occlusion_fractions, output_stride,
        score_threshold=args.recall_score_threshold, iou_threshold=args.eval_iou_threshold,
    )

    print("\n=== direct per-frame instances ===")
    print(f"AP50_segm={direct_ap50['map50']:.4f}")
    print(direct_recall.to_markdown())
    print(f"\n=== tracked (IOU-Tracker, window_radius={args.window_radius}, max_age={args.max_age}) ===")
    print(f"AP50_segm={tracked_ap50['map50']:.4f}")
    print(tracked_recall.to_markdown())


if __name__ == "__main__":
    main()
