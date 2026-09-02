"""Root-cause decomposition of detection errors on a GraSP split, using an
already-trained detection checkpoint. Not a new headline metric -- a
diagnostic answering "when an instance is missed, what specifically
happened": no candidate box at all, a candidate box that existed but scored
below threshold, a candidate box claimed by a higher-scoring same-class
neighbor (the occlusion/NMS mechanism directly), or a candidate box that
landed on the wrong class. Same breakdown for false positives: background
hallucination vs. a real instrument located but misclassified vs. a
duplicate. See docs/error_analysis.md for the results this produced on the
official test set and docs/DECISIONS.md, 2026-09-02.

Read-only: loads a checkpoint, runs inference, prints tables. Does not
retrain or write to experiments/.

Usage:
    python scripts/analyze_detection_errors.py \\
        experiments/detection_weighted_loss_20260901-001115/best.pt \\
        [--split test] [--data-root PATH] [--score-thresh 0.5] [--iou-thresh 0.5]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from surgical_ai.data.detection_dataset import GraspDetectionDataset, build_detection_transforms, collate_fn
from surgical_ai.evaluation.detection import compute_occlusion_fractions, evaluate_detection, dataset_to_coco_gt
from surgical_ai.models.detectors.registry import build_detector
from surgical_ai.training.trainer import collect_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", default="fasterrcnn_mobilenet_v3")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--loc-iou-floor", type=float, default=0.1)
    return parser.parse_args()


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def occl_bucket(frac: float) -> str:
    if frac <= 0.0:
        return "isolated"
    return "heavy" if frac > 0.5 else "light"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    val_ds = GraspDetectionDataset(args.data_root, args.split, transform=build_detection_transforms(train=False))
    class_names = val_ds.class_names_ordered()
    coco_gt = dataset_to_coco_gt(val_ds)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = build_detector(args.model, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    predictions = collect_detections(model, val_loader, device)
    metrics = evaluate_detection(coco_gt, predictions, class_names)
    print(f"\nsanity check -- mAP@50 = {metrics.map50:.4f}, mAP@50:95 = {metrics.map50_95:.4f}\n")

    occlusion_fractions = compute_occlusion_fractions(val_ds)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        preds_by_image[p["image_id"]].append(p)
    for lst in preds_by_image.values():
        lst.sort(key=lambda p: -p["score"])

    fn_reasons = defaultdict(int)
    fn_reasons_by_class = defaultdict(lambda: defaultdict(int))
    fn_reasons_by_bucket = defaultdict(lambda: defaultdict(int))
    fp_reasons = defaultdict(int)
    fp_reasons_by_class = defaultdict(lambda: defaultdict(int))
    fp_confusion_pairs = defaultdict(int)
    n_tp = 0
    n_gt_total = 0

    for image_idx, (_file_name, anns) in enumerate(val_ds.samples):
        gts = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            label = val_ds._id_to_index[a["category_id"]] + 1
            frac = occlusion_fractions.get(a["id"], 0.0)
            gts.append({"id": a["id"], "label": label, "box": [x, y, w, h], "bucket": occl_bucket(frac), "matched": False})
        n_gt_total += len(gts)

        preds_all = preds_by_image.get(image_idx, [])
        preds_thresh = [p for p in preds_all if p["score"] >= args.score_thresh]

        matched_pred_ids = set()
        for pi, p in enumerate(preds_thresh):
            best_gt, best_iou = None, 0.0
            for g in gts:
                if g["matched"] or g["label"] != p["category_id"]:
                    continue
                iou = iou_xyxy(g["box"], p["bbox"])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
            if best_gt is not None and best_iou >= args.iou_thresh:
                best_gt["matched"] = True
                matched_pred_ids.add(pi)
                n_tp += 1

        for g in gts:
            if g["matched"]:
                continue
            same_label_preds = [p for p in preds_all if p["category_id"] == g["label"]]
            best_same_iou, best_same_pred = 0.0, None
            for p in same_label_preds:
                iou = iou_xyxy(g["box"], p["bbox"])
                if iou > best_same_iou:
                    best_same_iou, best_same_pred = iou, p

            best_any_iou, best_any_pred = 0.0, None
            for p in preds_all:
                iou = iou_xyxy(g["box"], p["bbox"])
                if iou > best_any_iou:
                    best_any_iou, best_any_pred = iou, p

            if best_same_iou >= args.iou_thresh:
                reason = "low_confidence" if best_same_pred["score"] < args.score_thresh else "stolen_by_neighbor"
            elif best_any_iou >= args.iou_thresh and best_any_pred["category_id"] != g["label"]:
                reason = "classification_error" if best_any_pred["score"] >= args.score_thresh else "classification_error_low_conf"
            elif args.loc_iou_floor <= best_same_iou < args.iou_thresh:
                reason = "localization_error"
            else:
                reason = "not_proposed"

            fn_reasons[reason] += 1
            fn_reasons_by_class[class_names[g["label"] - 1]][reason] += 1
            fn_reasons_by_bucket[g["bucket"]][reason] += 1

        for pi, p in enumerate(preds_thresh):
            if pi in matched_pred_ids:
                continue
            best_iou, best_gt = 0.0, None
            for g in gts:
                iou = iou_xyxy(g["box"], p["bbox"])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
            if best_iou >= args.iou_thresh:
                reason = "duplicate" if best_gt["label"] == p["category_id"] else "misclassified_fp"
                if reason == "misclassified_fp":
                    fp_confusion_pairs[(class_names[best_gt["label"] - 1], class_names[p["category_id"] - 1])] += 1
            elif args.loc_iou_floor <= best_iou < args.iou_thresh:
                reason = "loc_fp"
            else:
                reason = "background"
            fp_reasons[reason] += 1
            fp_reasons_by_class[class_names[p["category_id"] - 1]][reason] += 1

    n_fn = sum(fn_reasons.values())
    n_fp = sum(fp_reasons.values())
    print(f"total GT instances: {n_gt_total}, TP: {n_tp}, FN: {n_fn}, FP (score>={args.score_thresh}): {n_fp}\n")

    print("=== False negative (missed GT instance) root cause ===")
    for reason, count in sorted(fn_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<28}{count:>6}  ({100*count/n_fn:.1f}% of misses)")

    print("\n=== False negative reason x occlusion bucket ===")
    buckets = ["isolated", "light", "heavy"]
    reasons_order = ["not_proposed", "low_confidence", "stolen_by_neighbor", "classification_error", "classification_error_low_conf", "localization_error"]
    print(f"{'reason':<28}" + "".join(f"{b:>12}" for b in buckets))
    for reason in reasons_order:
        print(f"{reason:<28}" + "".join(f"{fn_reasons_by_bucket[b].get(reason, 0):>12}" for b in buckets))

    print("\n=== False negative reason x class ===")
    print(f"{'class':<28}" + "".join(f"{r:>16}" for r in reasons_order))
    for cname in class_names:
        print(f"{cname:<28}" + "".join(f"{fn_reasons_by_class[cname].get(r, 0):>16}" for r in reasons_order))

    print("\n=== False positive root cause ===")
    for reason, count in sorted(fp_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<28}{count:>6}  ({100*count/n_fp:.1f}% of FP)")

    print("\n=== misclassified_fp confusion pairs (true -> predicted) ===")
    for (true_n, pred_n), count in sorted(fp_confusion_pairs.items(), key=lambda kv: -kv[1]):
        print(f"  true={true_n:<28} predicted={pred_n:<28} count={count}")


if __name__ == "__main__":
    main()
