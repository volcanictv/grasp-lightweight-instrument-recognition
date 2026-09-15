"""Confidence-based abstention, as a targeted alternative/complement to
the area-based rule (2026-09-04): manual review of specific error-log
cases (CASE041/04830.jpg, CASE041/05145.jpg, CASE053/08575.jpg) found the
real "no model or human can tell" instances are near coin-flips in the
ensemble's own output (e.g. Bipolar Forceps 0.444 vs Prograsp Forceps
0.432) -- not necessarily small in area. One instance in that same
review (CASE041/05145.jpg, Bipolar Forceps, 13.45% of frame) was
confidently (90.2%) misclassified as Large Needle Driver despite being
large and clearly visible, which no area threshold could ever catch.

This sweeps a threshold on the ensemble's own max softmax probability
(low confidence = abstain) and separately checks ensemble member
disagreement (how often the 4 members' individual argmax predictions
don't unanimously agree), to see whether either signal can isolate the
genuinely hopeless cases at much lower coverage cost than the area rule
(target: >=98-99% coverage, per direct instruction).

Usage:
    python scripts/evaluate_region_confidence_filter.py
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
from sklearn.metrics import f1_score

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--weight-320", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "confidence_filter_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.config.read_text())
    members = ensemble_config["members"]
    weight_320 = args.weight_320 if args.weight_320 is not None else ensemble_config["weight_resnet50_320"]

    class_names = None
    y_true = None
    all_probs = []
    all_member_preds = []
    for member in members:
        ds = GraspRegionDataset(
            args.data_root, args.split, transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])

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
        all_member_preds.append(probs.argmax(axis=1))

    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)
    correct = (y_pred == y_true)
    max_conf = avg.max(axis=1)

    member_preds = np.stack(all_member_preds, axis=1)  # (N, 4)
    n_agree = (member_preds == y_pred[:, None]).sum(axis=1)  # how many of 4 members agree with the ensemble's own pick
    unanimous = n_agree == len(members)

    print(f"baseline: n={len(y_true)} accuracy={correct.mean():.4f} macro_f1={f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"fully unanimous (4/4 members agree): {unanimous.sum()} ({unanimous.mean()*100:.1f}%), accuracy within={correct[unanimous].mean():.4f}")
    print(f"not unanimous: {(~unanimous).sum()} ({(~unanimous).mean()*100:.1f}%), accuracy within={correct[~unanimous].mean():.4f}\n")

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    rows = []
    for t in thresholds:
        covered = max_conf >= t
        n_abstain = int((~covered).sum())
        row = {
            "confidence_threshold": t,
            "n_abstain": n_abstain,
            "coverage_pct": float(covered.mean() * 100),
            "accuracy_covered": float(correct[covered].mean()) if covered.any() else None,
            "macro_f1_covered": float(f1_score(y_true[covered], y_pred[covered], average="macro", zero_division=0)) if covered.any() else None,
            "accuracy_abstained_group": float(correct[~covered].mean()) if n_abstain else None,
        }
        rows.append(row)

    print("confidence-threshold sweep (abstain when max ensemble probability < t):")
    print("| t | n abstained | coverage | accuracy (covered) | macro-F1 (covered) | accuracy (abstained group) |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        acc_c = f"{row['accuracy_covered']:.4f}" if row["accuracy_covered"] is not None else "n/a"
        f1_c = f"{row['macro_f1_covered']:.4f}" if row["macro_f1_covered"] is not None else "n/a"
        acc_a = f"{row['accuracy_abstained_group']:.4f}" if row["accuracy_abstained_group"] is not None else "n/a"
        print(f"| {row['confidence_threshold']:.2f} | {row['n_abstain']} | {row['coverage_pct']:.2f}% | {acc_c} | {f1_c} | {acc_a} |")

    for t in (0.35, 0.40, 0.45):
        covered = max_conf >= t
        print(f"\nper-class coverage at t={t}:")
        for i, name in enumerate(class_names):
            cls_mask = y_true == i
            cls_total = int(cls_mask.sum())
            cls_covered = int((cls_mask & covered).sum())
            print(f"  {name:<28} {cls_covered:>4}/{cls_total:<4} ({cls_covered/cls_total*100 if cls_total else 0:.1f}%)")

    (args.out).write_text(json.dumps({
        "baseline_accuracy": float(correct.mean()),
        "baseline_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "unanimous_n": int(unanimous.sum()), "unanimous_accuracy": float(correct[unanimous].mean()),
        "not_unanimous_n": int((~unanimous).sum()), "not_unanimous_accuracy": float(correct[~unanimous].mean()),
        "confidence_sweep": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
