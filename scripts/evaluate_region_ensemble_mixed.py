"""General N-way Task B ensemble across possibly-different architectures
and crop preprocessing, generalizing evaluate_region_ensemble.py (which
assumed two MobileNetV3-Small checkpoints) now that a ResNet-50 checkpoint
exists (docs/DECISIONS.md, 2026-09-02, the session's biggest single-model
win). Tests whether averaging it with the existing MobileNetV3 checkpoints
beats ResNet-50 alone, the same "architecture diversity" question already
asked of the MobileNetV3-only ensemble -- checked rather than assumed to
transfer to a mix that includes a meaningfully heavier model.

Usage:
    python scripts/evaluate_region_ensemble_mixed.py \\
        --member experiments/region_letterbox_resnet50_20260902-225519/best.pt:resnet50:letterbox \\
        --member experiments/region_baseline_20260831-182451/best.pt:mobilenet_v3_small:plain \\
        --member experiments/region_letterbox_crop_20260902-152750/best.pt:mobilenet_v3_small:letterbox
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", action="append", required=True, dest="members", help="checkpoint:model_name:plain|letterbox, repeatable")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> float:
    acc = (y_pred == y_true).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== {name}: accuracy={acc:.4f} macro-F1={macro_f1:.4f} ===")
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)
    for i, cname in enumerate(class_names):
        print(f"  {cname:<28} F1={f1[i]:.3f}")
    return macro_f1


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    specs = []
    for m in args.members:
        parts = m.split(":")
        if len(parts) == 4:
            ckpt, model_name, crop_mode, image_size = parts
            image_size = int(image_size)
        else:
            ckpt, model_name, crop_mode = parts
            image_size = args.image_size
        specs.append((Path(ckpt), model_name, crop_mode == "letterbox", image_size))

    class_names = None
    y_true = None
    all_probs = []
    for ckpt, model_name, letterbox, image_size in specs:
        ds = GraspRegionDataset(args.data_root, args.split, transform=build_transforms(image_size, train=False), letterbox=letterbox)
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])

        model = build_model(model_name, num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        missing, _unexpected = model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        assert not missing, f"missing keys loading {ckpt}: {missing}"
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs)
        all_probs.append(probs)
        acc = (probs.argmax(axis=1) == y_true).mean()
        print(f"loaded {ckpt.parent.name} ({model_name}, {'letterbox' if letterbox else 'plain'}): standalone accuracy={acc:.4f}")

    print(f"\n{'='*70}\nEnsemble combinations\n{'='*70}")
    best_macro_f1, best_combo = 0.0, None
    n = len(all_probs)
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            avg_probs = np.mean([all_probs[i] for i in combo], axis=0)
            y_pred = avg_probs.argmax(axis=1)
            name = " + ".join(specs[i][0].parent.name for i in combo)
            macro_f1 = report(name, y_true, y_pred, class_names)
            if macro_f1 > best_macro_f1:
                best_macro_f1, best_combo = macro_f1, name

    print(f"\nBest combination: {best_combo} (macro-F1={best_macro_f1:.4f})")


if __name__ == "__main__":
    main()
