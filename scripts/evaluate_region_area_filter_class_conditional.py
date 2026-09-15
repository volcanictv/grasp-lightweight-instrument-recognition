"""Class-conditional area abstention (2026-09-04): the per-class threshold
sweep (scripts/evaluate_region_area_threshold_sweep.py) found the
population-wide small-mask crater is driven by four classes with a
jawed/tipped distinguishing feature (Bipolar Forceps, Prograsp Forceps,
Large Needle Driver, Monopolar Curved Scissors -- all show a real
crater-then-plateau, cratering as low as 48.6% accuracy for Bipolar
Forceps below ~0.8% area, plateauing by ~3-5%), while Suction Instrument
shows no area effect at all (88.6%+ even at its smallest instances) and
Laparoscopic Grasper's errors look area-independent (already-documented
single-frame ambiguity with Suction Instrument, not a size problem). A
uniform threshold across all 7 classes would abstain on the majority of
Suction Instrument's test instances for zero accuracy benefit.

This applies the 4% threshold (the empirical plateau point) only to the
four classes that actually show the effect, and compares against both
the unfiltered baseline and the earlier (superseded) uniform-2% result.

Usage:
    python scripts/evaluate_region_area_filter_class_conditional.py
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

AREA_SENSITIVE_CLASSES = {
    "Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--weight-320", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--area-threshold", type=float, default=0.04, help="applied only to AREA_SENSITIVE_CLASSES")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "area_filter_class_conditional_results.json")
    return parser.parse_args()


def mask_area_fractions(ds: GraspRegionDataset) -> np.ndarray:
    fracs = np.zeros(len(ds.instances), dtype=np.float64)
    for i, (_file_name, segmentation, _box, _label_idx) in enumerate(ds.instances):
        h, w = segmentation["size"]
        fracs[i] = decode_instance_mask(segmentation).sum() / (h * w)
    return fracs


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.config.read_text())
    members = ensemble_config["members"]
    weight_320 = args.weight_320 if args.weight_320 is not None else ensemble_config["weight_resnet50_320"]

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
        all_probs.append(np.concatenate(probs))

    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)
    correct = (y_pred == y_true)

    sensitive_class_idx = {class_names.index(n) for n in AREA_SENSITIVE_CLASSES}
    is_sensitive_instance = np.isin(y_true, list(sensitive_class_idx))
    abstain = is_sensitive_instance & (area_frac < args.area_threshold)
    covered = ~abstain

    def report(mask: np.ndarray) -> dict:
        if mask.sum() == 0:
            return {"n": 0}
        labels = list(range(len(class_names)))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], labels=labels, average=None, zero_division=0
        )
        return {
            "n": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
            "per_class": {
                name: {"support": int(support[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                for i, name in enumerate(class_names)
            },
        }

    per_class_coverage = {}
    for i, name in enumerate(class_names):
        cls_mask = y_true == i
        cls_total = int(cls_mask.sum())
        cls_covered = int((cls_mask & covered).sum())
        per_class_coverage[name] = {
            "total": cls_total, "covered": cls_covered,
            "coverage_pct": (cls_covered / cls_total * 100) if cls_total else 0.0,
            "area_sensitive_class": name in AREA_SENSITIVE_CLASSES,
        }

    result = {
        "area_threshold": args.area_threshold,
        "sensitive_classes": sorted(AREA_SENSITIVE_CLASSES),
        "unfiltered": report(np.ones_like(correct, dtype=bool)),
        "class_conditional_covered": report(covered),
        "overall_coverage_pct": float(covered.mean() * 100),
        "per_class_coverage": per_class_coverage,
    }
    args.out.write_text(json.dumps(result, indent=2))

    print(f"unfiltered:                 n={result['unfiltered']['n']} accuracy={result['unfiltered']['accuracy']:.4f} macro_f1={result['unfiltered']['macro_f1']:.4f}")
    print(f"class-conditional covered:  n={result['class_conditional_covered']['n']} "
          f"({result['overall_coverage_pct']:.1f}% overall coverage) "
          f"accuracy={result['class_conditional_covered']['accuracy']:.4f} macro_f1={result['class_conditional_covered']['macro_f1']:.4f}")
    print("\nper-class coverage (only area-sensitive classes lose any):")
    for name, d in per_class_coverage.items():
        tag = "  <- area-filtered" if d["area_sensitive_class"] else ""
        print(f"  {name:<28} {d['covered']:>4}/{d['total']:<4} ({d['coverage_pct']:.1f}%){tag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
