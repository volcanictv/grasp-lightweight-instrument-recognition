"""Confirmatory check: does the resnet50_320-weighted ensemble combination
(weight=0.40, found by sweeping against official test,
docs/DECISIONS.md 2026-09-03) replicate on models trained fresh on fold1
and evaluated on fold1's own held-out val set? Answers whether the exact
weight value generalizes, or was tuned to official test's specific
composition -- see the same DECISIONS.md entry for the result (it did
not replicate; flat weighting reverted to the reported best).

Usage:
    python scripts/evaluate_fold1_ensemble_weight_sweep.py
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
from sklearn.metrics import f1_score

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

FOLD1_MEMBERS = [
    ("experiments/region_letterbox_resnet50_320_fold1_20260903-012254/best.pt", "resnet50", 320, True, "resnet50_320_fold1"),
    ("experiments/region_letterbox_resnet50_fold1_20260903-012222/best.pt", "resnet50", 224, True, "resnet50_224_fold1"),
    ("experiments/region_baseline_fold1_20260903-015022/best.pt", "mobilenet_v3_small", 224, False, "baseline_fold1"),
    ("experiments/region_letterbox_crop_fold1_20260903-015614/best.pt", "mobilenet_v3_small", 224, True, "letterbox_crop_fold1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    class_names = None
    y_true = None
    all_probs = []
    for ckpt, model_name, size, letterbox, label in FOLD1_MEMBERS:
        # eval split "fold1" resolves to val=fold1 (never gradient-descent-trained
        # by these fold1-trained checkpoints, only used for their own checkpoint selection)
        ds = GraspRegionDataset(args.data_root, "fold1", transform=build_transforms(size, train=False), letterbox=letterbox)
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])

        model = build_model(model_name, num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / ckpt, map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs)
        all_probs.append(probs)
        print(f"{label}: standalone accuracy={(probs.argmax(axis=1)==y_true).mean():.4f}")

    print(f"\nn instances in fold1 val set: {len(y_true)}")
    print(f"\n{'weight_320':>12}{'accuracy':>12}{'macro-F1':>12}")
    for w320 in args.weights:
        w_rest = (1 - w320) / 3
        avg = w320 * all_probs[0] + w_rest * sum(all_probs[1:])
        y_pred = avg.argmax(axis=1)
        acc = (y_pred == y_true).mean()
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"{w320:>12.2f}{acc:>12.4f}{macro_f1:>12.4f}")


if __name__ == "__main__":
    main()
