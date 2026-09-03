"""Weighted Task B ensemble, driven by configs/region_ensemble.yaml (the
member checkpoints and default weight) instead of hardcoded values, so the
weight can be changed on the fly for testing without touching code --
requested directly, 2026-09-03, after the fold1/fold2 confirmatory check
found the fine-tuned weight (0.40) doesn't clearly beat flat weighting on
held-out data (docs/DECISIONS.md). 0.40 is kept as the operational
default by deliberate choice, not because it's confirmed to generalize --
see the config file's own comment for the full disclosure. The validated,
reported-as-best number for this project is the flat ensemble (weight
0.25 each, macro-F1 0.8929); this script's default is what's actually run
day to day.

--weight-320 overrides the config's default for a quick one-off test
without editing the file. Pass 0.25 to reproduce flat averaging exactly.

Usage:
    python scripts/evaluate_region_ensemble_weighted.py
    python scripts/evaluate_region_ensemble_weighted.py --weight-320 0.25
    python scripts/evaluate_region_ensemble_weighted.py --config configs/region_ensemble.yaml --split fold1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--weight-320", type=float, default=None, help="override the config's weight for resnet50_320; remainder split equally across the other members")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.config.read_text())
    members = ensemble_config["members"]
    weight_320 = args.weight_320 if args.weight_320 is not None else ensemble_config["weight_resnet50_320"]
    print(f"device: {device}, config={args.config}, weight(resnet50_320)={weight_320}")

    class_names = None
    y_true = None
    all_probs = []
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
        print(f"  {member['label']}: standalone accuracy={(probs.argmax(axis=1)==y_true).mean():.4f}")

    # first config entry is the weighted member; the rest split the remainder equally
    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)

    acc = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)
    print(f"\naccuracy={acc:.4f} macro-F1={macro_f1:.4f}")
    for i, name in enumerate(class_names):
        print(f"  {name:<28} P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f}")


if __name__ == "__main__":
    main()
