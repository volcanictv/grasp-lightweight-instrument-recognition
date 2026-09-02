"""Evaluate the fold1 and fold2 Mask R-CNN checkpoints, individually and
ensembled, against the official test set -- legitimate because neither
checkpoint's training data (fold2 for the fold1 model, fold1 for the
fold2 model) overlaps with the 5 official-test cases at all (fold1 and
fold2 partition the 8 official-train cases exactly). See
src/surgical_ai/inference/ensemble.py and docs/DECISIONS.md.

Usage:
    python scripts/evaluate_maskrcnn_ensemble.py \\
        experiments/instance_segmentation_maskrcnn_20260901-171545/best.pt \\
        experiments/instance_segmentation_maskrcnn_fold2_20260901-170628/best.pt \\
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
    parser.add_argument("checkpoint_fold1", type=Path)
    parser.add_argument("checkpoint_fold2", type=Path)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Candidate threshold before fusion.")
    parser.add_argument("--fusion-iou-threshold", type=float, default=0.5)
    parser.add_argument("--recall-score-threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_model_detections(model: torch.nn.Module, image_tensor: torch.Tensor, score_threshold: float) -> list[dict]:
    output = model([image_tensor])[0]
    boxes = output["boxes"].cpu().tolist()
    labels = output["labels"].cpu().tolist()
    scores = output["scores"].cpu().tolist()
    masks = output["masks"].cpu().numpy()[:, 0]  # (N, H, W) float [0,1], pre-threshold
    dets = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < score_threshold:
            continue
        dets.append({"box": box, "label": label - 1, "score": score, "mask": mask})  # back to 0-indexed
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
    recall_str = ", ".join(
        f"{b}={hits[b] / counts[b] if counts[b] else float('nan'):.3f}" for b in ("isolated", "light", "heavy")
    )
    print(f"[{label}] AP50_segm={ap50['map50']:.4f}  occlusion recall: {recall_str}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    _, val_split = splits.resolve_train_val_split("official")
    val_ds = GraspDetectionDataset(args.data_root, val_split, include_masks=True, transform=build_detection_transforms(train=False))
    class_names = val_ds.class_names_ordered()
    occlusion_fractions = compute_occlusion_fractions(val_ds)

    model1 = build_detector("maskrcnn_mobilenet_v3", num_classes=len(class_names), pretrained=False).to(device)
    model1.load_state_dict(torch.load(args.checkpoint_fold1, map_location=device))
    model1.eval()

    model2 = build_detector("maskrcnn_mobilenet_v3", num_classes=len(class_names), pretrained=False).to(device)
    model2.load_state_dict(torch.load(args.checkpoint_fold2, map_location=device))
    model2.eval()

    n_samples = len(val_ds.samples) if args.limit is None else min(args.limit, len(val_ds.samples))
    preds_model1: list[list[tuple]] = []
    preds_model2: list[list[tuple]] = []
    preds_ensemble: list[list[tuple]] = []
    gt_instances: list[list[tuple]] = []

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, range(n_samples)), batch_size=1, shuffle=False, collate_fn=collate_fn,
    )
    for idx, (images, targets) in enumerate(loader):
        image_tensor = images[0].to(device)
        dets1 = collect_model_detections(model1, image_tensor, args.score_threshold)
        dets2 = collect_model_detections(model2, image_tensor, args.score_threshold)

        preds_model1.append([(d["mask"] >= 0.5, d["label"], d["score"]) for d in dets1])
        preds_model2.append([(d["mask"] >= 0.5, d["label"], d["score"]) for d in dets2])

        fused = weighted_fusion_merge([dets1, dets2], iou_threshold=args.fusion_iou_threshold)
        preds_ensemble.append([(d["mask"] >= 0.5, d["label"], d["score"]) for d in fused])

        image_idx = int(targets[0]["image_id"].item())
        _file_name, anns = val_ds.samples[image_idx]
        gts = []
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if gt_mask.any():
                gts.append((gt_mask, val_ds._id_to_index[a["category_id"]]))
        gt_instances.append(gts)

        if idx % 100 == 0:
            print(f"{idx}/{n_samples}", flush=True)

    # evaluate_predictions indexes positionally against the full val_ds.samples,
    # so pad untested indices (under --limit) with empty predictions rather than
    # truncate -- see the identical fix in evaluate_tracking_segm.py.
    for _ in range(len(val_ds.samples) - n_samples):
        preds_model1.append([])
        preds_model2.append([])
        preds_ensemble.append([])
        gt_instances.append([])

    evaluate_predictions(preds_model1, gt_instances, val_ds, occlusion_fractions, class_names, args.recall_score_threshold, "fold1 model alone")
    evaluate_predictions(preds_model2, gt_instances, val_ds, occlusion_fractions, class_names, args.recall_score_threshold, "fold2 model alone")
    evaluate_predictions(preds_ensemble, gt_instances, val_ds, occlusion_fractions, class_names, args.recall_score_threshold, "ensemble (WBF)")


if __name__ == "__main__":
    main()
