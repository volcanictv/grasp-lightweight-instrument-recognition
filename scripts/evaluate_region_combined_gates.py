"""Final combined-protocol number: both adopted abstention gates applied
together (class-conditional area rule + confidence threshold, 2026-09-04)
-- each was validated separately; this reports what the classifier
actually ships as once both are stacked.

Usage:
    python scripts/evaluate_region_combined_gates.py
"""
from __future__ import annotations

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

AREA_SENSITIVE_CLASSES = {"Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors"}
AREA_THRESHOLD = 0.04
CONFIDENCE_THRESHOLD = 0.40


def main() -> None:
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_config = yaml.safe_load((REPO_ROOT / "configs" / "region_ensemble.yaml").read_text())
    members = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]

    class_names = None
    y_true = None
    area_frac = None
    all_probs = []
    for member in members:
        ds = GraspRegionDataset(
            data_root, "test", transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            area_frac = np.array([
                decode_instance_mask(seg).sum() / (seg["size"][0] * seg["size"][1])
                for _fn, seg, _box, _lbl in ds.instances
            ])

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
    max_conf = avg.max(axis=1)

    is_sensitive = np.isin(y_true, [class_names.index(n) for n in AREA_SENSITIVE_CLASSES])
    area_abstain = is_sensitive & (area_frac < AREA_THRESHOLD)
    confidence_abstain = max_conf < CONFIDENCE_THRESHOLD
    abstain = area_abstain | confidence_abstain
    covered = ~abstain

    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true[covered], y_pred[covered], labels=labels, average=None, zero_division=0
    )
    result = {
        "n_total": int(len(y_true)),
        "n_covered": int(covered.sum()),
        "coverage_pct": float(covered.mean() * 100),
        "n_abstained_area_only": int((area_abstain & ~confidence_abstain).sum()),
        "n_abstained_confidence_only": int((confidence_abstain & ~area_abstain).sum()),
        "n_abstained_both": int((area_abstain & confidence_abstain).sum()),
        "accuracy": float((y_pred[covered] == y_true[covered]).mean()),
        "macro_f1": float(f1_score(y_true[covered], y_pred[covered], average="macro", zero_division=0)),
        "per_class_coverage": {},
    }
    for i, name in enumerate(class_names):
        cls_mask = y_true == i
        cls_total = int(cls_mask.sum())
        cls_covered = int((cls_mask & covered).sum())
        result["per_class_coverage"][name] = {
            "total": cls_total, "covered": cls_covered,
            "coverage_pct": (cls_covered / cls_total * 100) if cls_total else 0.0,
            "f1": float(f1[i]), "recall": float(recall[i]),
        }

    (REPO_ROOT / "docs" / "combined_gates_results.json").write_text(json.dumps(result, indent=2))
    print(f"combined gates: n_covered={result['n_covered']} ({result['coverage_pct']:.2f}%) "
          f"accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
    print(f"  abstained by area only: {result['n_abstained_area_only']}")
    print(f"  abstained by confidence only: {result['n_abstained_confidence_only']}")
    print(f"  abstained by both: {result['n_abstained_both']}")
    print("\nper-class coverage under combined gates:")
    for name, d in result["per_class_coverage"].items():
        print(f"  {name:<28} {d['covered']:>4}/{d['total']:<4} ({d['coverage_pct']:.1f}%)  F1={d['f1']:.3f}")
    print(f"\nwrote docs/combined_gates_results.json")


if __name__ == "__main__":
    main()
