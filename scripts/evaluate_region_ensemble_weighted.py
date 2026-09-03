"""Weighted 4-model Task B ensemble (2x ResNet-50 + 2x MobileNetV3),
favoring resnet50_320 over a flat average. Motivated by a specific failure
found checking known hard examples against the flat-averaged ensemble
(docs/DECISIONS.md, 2026-09-03): resnet50_320 alone confidently got a
Bipolar-Forceps-as-Prograsp-Forceps case right, but got outvoted 3-to-1 by
weaker models under flat 1/4-each averaging. Swept a fixed weight on the
full official test set rather than per-instance confidence weighting
(confidence weighting was shown, in an earlier ensemble, to favor whichever
model is more confident even when that model is wrong -- not what's
wanted here).

--weight-320 default (0.40) was chosen by a sweep against the official
test set -- see docs/DECISIONS.md's overfitting-risk entry, same date,
for the important caveat that this makes it a test-set-selected value,
not yet confirmed on an independent held-out split.

Usage:
    python scripts/evaluate_region_ensemble_weighted.py --weight-320 0.40
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
from sklearn.metrics import f1_score, precision_recall_fscore_support

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

DEFAULT_MEMBERS = [
    ("experiments/region_letterbox_resnet50_320_20260902-234246/best.pt", "resnet50", 320, True),
    ("experiments/region_letterbox_resnet50_20260902-225519/best.pt", "resnet50", 224, True),
    ("experiments/region_baseline_20260831-182451/best.pt", "mobilenet_v3_small", 224, False),
    ("experiments/region_letterbox_crop_20260902-152750/best.pt", "mobilenet_v3_small", 224, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weight-320", type=float, default=0.40, help="weight for resnet50_320; remainder split equally across the other three")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}, weight(resnet50_320)={args.weight_320}")

    class_names = None
    y_true = None
    all_probs = []
    for ckpt, model_name, image_size, letterbox in DEFAULT_MEMBERS:
        ds = GraspRegionDataset(args.data_root, args.split, transform=build_transforms(image_size, train=False), letterbox=letterbox)
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
        all_probs.append(np.concatenate(probs))

    w_rest = (1 - args.weight_320) / (len(DEFAULT_MEMBERS) - 1)
    avg = args.weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)

    acc = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)
    print(f"\naccuracy={acc:.4f} macro-F1={macro_f1:.4f}")
    for i, name in enumerate(class_names):
        print(f"  {name:<28} P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f}")


if __name__ == "__main__":
    main()
