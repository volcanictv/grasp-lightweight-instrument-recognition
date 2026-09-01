"""Evaluate tracking-by-detection's effect on occlusion-stratified recall
(Milestone 9's second planned component, docs/DECISIONS.md).

GraSP's annotated frames are ~35 frames apart within a case -- too sparse
for frame-to-frame tracking on their own. This script runs an already-
trained detector across a small window of the *raw* (unannotated, ~1fps)
frames surrounding each annotated val frame, tracks through that window
with a simple IOU-Tracker, and compares occlusion-stratified recall at the
annotated frame between the raw per-frame detections and the tracker's
state (which can carry a recently-lost detection forward through a few
frames of missed detection).

Usage:
    python scripts/evaluate_tracking.py configs/detection_copy_paste.yaml \\
        experiments/detection_copy_paste_.../best.pt [--data-root PATH] \\
        [--device cuda:0] [--window-radius 5] [--max-age 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data import splits  # noqa: E402
from surgical_ai.data.detection_dataset import GraspDetectionDataset  # noqa: E402
from surgical_ai.evaluation.detection import (  # noqa: E402
    compute_occlusion_fractions,
    dataset_to_coco_gt,
    evaluate_detection,
    evaluate_occlusion_stratified_recall,
)
from surgical_ai.inference.pipeline import build_frame_window, parse_case_and_frame_number, run_window_and_track  # noqa: E402
from surgical_ai.models.detectors.registry import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSp")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-radius", type=int, default=5)
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Tracker's association IoU threshold.")
    parser.add_argument(
        "--min-confidence-to-coast", type=float, default=0.5,
        help="Only tracks whose last real detection scored at least this let coast through a missed "
        "frame -- a low-confidence (likely noise) track coasting was found to hurt mAP@50 (docs/DECISIONS.md).",
    )
    parser.add_argument(
        "--boundary-margin-frac", type=float, default=0.05,
        help="Fraction of frame width/height counted as 'near the edge' for boundary-exit detection -- "
        "a track near an edge and moving further outward is dropped immediately instead of coasting, "
        "on the theory that it's genuinely leaving the frame, not just occluded. Pass 0 to disable.",
    )
    parser.add_argument(
        "--occlusion-corridor-iou-threshold", type=float, default=0.0,
        help="If > 0, a missing track's extrapolated position overlapping another real detection this "
        "frame (at least this IoU) is treated as plausible occlusion evidence, granting "
        "--occluded-max-age instead of --max-age for the rest of that gap. 0 disables this (default).",
    )
    parser.add_argument(
        "--occluded-max-age", type=int, default=5,
        help="Extended coasting lifetime granted to a track classified as likely-occluded by the "
        "occlusion-corridor check (only relevant if --occlusion-corridor-iou-threshold > 0).",
    )
    parser.add_argument(
        "--require-continuous-occlusion-evidence", action="store_true",
        help="Re-check the occlusion-corridor condition on every missed frame (not just the first) -- "
        "extended trust requires evidence to keep holding up, not just have held once at gap start.",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.05,
        help="Threshold for a detection to be fed into the tracker / included in predictions at all -- "
        "kept low (COCO-eval convention) so the full confidence range is available for mAP's precision-"
        "recall curve. The occlusion-recall table's own fixed 0.5 operating point is separate, see "
        "--recall-score-threshold.",
    )
    parser.add_argument(
        "--recall-score-threshold", type=float, default=0.5,
        help="Fixed score cutoff for the occlusion-stratified recall table, matching this project's "
        "established convention -- independent of --score-threshold.",
    )
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5, help="Occlusion-recall matching IoU.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N val frames (debugging).")
    parser.add_argument(
        "--non-causal", action="store_true",
        help="Also fetch frames after the target (verified to make no difference to the result -- the "
        "tracker's state is read out at the target frame before later frames are processed -- so this "
        "only exists to reproduce that verification; causal-only is the default and is what a real-time "
        "system actually has available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)

    _, val_split = splits.resolve_train_val_split(config["data"]["split"])
    val_ds = GraspDetectionDataset(args.data_root, val_split)
    class_names = val_ds.class_names_ordered()
    frames_root = val_ds.frames_root

    model = build_detector(config["model"]["name"], num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    occlusion_fractions = compute_occlusion_fractions(val_ds)

    n_samples = len(val_ds.samples) if args.limit is None else min(args.limit, len(val_ds.samples))
    raw_predictions: list[dict] = []
    tracked_predictions: list[dict] = []

    for image_idx in range(n_samples):
        file_name, _anns = val_ds.samples[image_idx]
        case, frame_number = parse_case_and_frame_number(file_name)
        window = build_frame_window(frames_root, case, frame_number, args.window_radius, causal=not args.non_causal)
        if not any(n == frame_number for n, _ in window):
            continue  # the annotated frame itself must be in its own window

        direct, tracked = run_window_and_track(
            model, device, window, frame_number,
            score_threshold=args.score_threshold, iou_threshold=args.iou_threshold, max_age=args.max_age,
            min_confidence_to_coast=args.min_confidence_to_coast, boundary_margin_frac=args.boundary_margin_frac,
            occlusion_corridor_iou_threshold=args.occlusion_corridor_iou_threshold,
            occluded_max_age=args.occluded_max_age,
            require_continuous_occlusion_evidence=args.require_continuous_occlusion_evidence,
        )
        for d in direct:
            x1, y1, x2, y2 = d["box"]
            raw_predictions.append(
                {"image_id": image_idx, "category_id": d["label"], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d["score"]}
            )
        for d in tracked:
            x1, y1, x2, y2 = d["box"]
            tracked_predictions.append(
                {"image_id": image_idx, "category_id": d["label"], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d["score"]}
            )

        if image_idx % 100 == 0:
            print(f"{image_idx}/{n_samples}", flush=True)

    raw_recall = evaluate_occlusion_stratified_recall(
        val_ds, raw_predictions, occlusion_fractions,
        score_threshold=args.recall_score_threshold, iou_threshold=args.eval_iou_threshold,
    )
    tracked_recall = evaluate_occlusion_stratified_recall(
        val_ds, tracked_predictions, occlusion_fractions,
        score_threshold=args.recall_score_threshold, iou_threshold=args.eval_iou_threshold,
    )

    coco_gt = dataset_to_coco_gt(val_ds)
    raw_map = evaluate_detection(coco_gt, raw_predictions, class_names)
    tracked_map = evaluate_detection(coco_gt, tracked_predictions, class_names)

    print("\n=== raw per-frame detections ===")
    print(f"mAP@50={raw_map.map50:.4f} mAP@50:95={raw_map.map50_95:.4f}")
    print(raw_recall.to_markdown())
    mode = "non-causal, verified identical to causal" if args.non_causal else "causal/real-time"
    print(
        "\n=== tracked (IOU-Tracker, window_radius=%d, max_age=%d, %s) ==="
        % (args.window_radius, args.max_age, mode)
    )
    print(f"mAP@50={tracked_map.map50:.4f} mAP@50:95={tracked_map.map50_95:.4f}")
    print(tracked_recall.to_markdown())


if __name__ == "__main__":
    main()
