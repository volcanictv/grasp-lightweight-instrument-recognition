"""Tests a proposed abstention rule against error-log review (2026-09-04
supervisor feedback): instances whose mask covers less than 2% of the
frame are often barely visible to a human, let alone a model, and the
error logs show a disproportionate share of the ensemble's wrong
predictions land on exactly these instances. This script checks that
claim directly against the existing weighted ensemble rather than assuming
it -- no retraining, just a post-hoc split of the already-computed
predictions by mask-area fraction.

Reuses the same weighted-ensemble machinery as
evaluate_region_ensemble_weighted.py so the "no filter" number here should
reproduce that script's baseline exactly.

Usage:
    python scripts/evaluate_region_area_filter.py
    python scripts/evaluate_region_area_filter.py --area-threshold 0.02
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--weight-320", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--area-threshold", type=float, default=0.02, help="mask area / frame area cutoff below which an instance is abstained on")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "area_filter_results.json")
    return parser.parse_args()


def mask_area_fractions(ds: GraspRegionDataset) -> np.ndarray:
    fracs = np.zeros(len(ds.instances), dtype=np.float64)
    for i, (_file_name, segmentation, _box, _label_idx) in enumerate(ds.instances):
        h, w = segmentation["size"]
        mask = decode_instance_mask(segmentation)
        fracs[i] = mask.sum() / (h * w)
    return fracs


def subset_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            name: {"support": int(support[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, name in enumerate(class_names)
        },
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.config.read_text())
    members = ensemble_config["members"]
    weight_320 = args.weight_320 if args.weight_320 is not None else ensemble_config["weight_resnet50_320"]
    print(f"device: {device}, config={args.config}, weight(resnet50_320)={weight_320}, area_threshold={args.area_threshold}")

    class_names = None
    y_true = None
    area_frac = None
    all_probs = []
    for member in members:
        ds = GraspRegionDataset(
            args.data_root, args.split, transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            area_frac = mask_area_fractions(ds)

        model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs)
        all_probs.append(probs)

    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)

    small_mask = area_frac < args.area_threshold
    overall = subset_report(y_true, y_pred, class_names)
    small = subset_report(y_true[small_mask], y_pred[small_mask], class_names)
    normal = subset_report(y_true[~small_mask], y_pred[~small_mask], class_names)

    per_class_small_share = {}
    for i, name in enumerate(class_names):
        cls_total = int((y_true == i).sum())
        cls_small = int(((y_true == i) & small_mask).sum())
        per_class_small_share[name] = {
            "total_instances": cls_total,
            "small_instances": cls_small,
            "pct_small": (cls_small / cls_total * 100) if cls_total else 0.0,
        }

    result = {
        "area_threshold": args.area_threshold,
        "n_total": int(len(y_true)),
        "n_small": int(small_mask.sum()),
        "pct_small": float(small_mask.mean() * 100),
        "overall_unfiltered": overall,
        "small_subset": small,
        "normal_subset_if_abstaining_on_small": normal,
        "per_class_small_share": per_class_small_share,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\noverall (unfiltered): n={overall['n']} accuracy={overall['accuracy']:.4f} macro_f1={overall['macro_f1']:.4f}")
    print(f"small (<{args.area_threshold*100:.0f}% of frame): n={small['n']} ({result['pct_small']:.1f}% of test set) "
          f"accuracy={small.get('accuracy', float('nan')):.4f} macro_f1={small.get('macro_f1', float('nan')):.4f}")
    print(f"normal (>={args.area_threshold*100:.0f}% of frame, i.e. if abstaining below threshold): "
          f"n={normal['n']} accuracy={normal['accuracy']:.4f} macro_f1={normal['macro_f1']:.4f}")
    print("\nper-class share of instances below the area threshold:")
    for name, d in per_class_small_share.items():
        print(f"  {name:<28} {d['small_instances']:>4}/{d['total_instances']:<4} ({d['pct_small']:.1f}%)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
