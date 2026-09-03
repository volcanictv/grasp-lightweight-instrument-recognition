"""Aspect-ratio-conditional weighting for the Task B ensemble, instead of
the blind 50/50 average `evaluate_region_ensemble.py` uses. Motivated by a
concrete failure found while planning further fixes (docs/DECISIONS.md,
2026-09-02): for one elongated crop, `region_letterbox_crop` alone got the
right answer (Large Needle Driver, 0.498 confidence) but
`region_baseline` was confidently wrong (Bipolar Forceps, 0.903) -- a flat
average lets the confident-wrong model win (0.648 Bipolar Forceps).
Confidence-weighting would make this worse, not better, since it would
trust the *more* confident (wrong) model even more. What actually matches
the diagnosis: `region_letterbox_crop` exists specifically to fix
elongated crops, so its vote should count for more exactly when the crop
is elongated -- a deterministic, known-at-inference-time property, not a
confidence heuristic.

weight_b = clip((aspect_ratio - 2.0) / 2.0, 0.5, 1.0) -- 0.5 (equal
weighting, matching the existing ensemble) for near-square crops
(ratio<=2), scaling up to fully trusting region_letterbox_crop for very
elongated ones (ratio>=4).

Usage:
    python scripts/evaluate_region_ensemble_aspect_weighted.py \\
        --checkpoint-a experiments/region_baseline_20260831-182451/best.pt \\
        --checkpoint-b experiments/region_letterbox_crop_20260902-152750/best.pt
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--ratio-low", type=float, default=2.0)
    parser.add_argument("--ratio-high", type=float, default=4.0)
    return parser.parse_args()


def predict_probs(checkpoint: Path, ds: GraspRegionDataset, num_classes: int, device: torch.device) -> np.ndarray:
    model = build_model("mobilenet_v3_small", num_classes=num_classes, pretrained=False, freeze_backbone=False).to(device)
    missing, _unexpected = model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    assert not missing, f"missing keys loading {checkpoint}: {missing}"
    model.eval()

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    all_probs = []
    with torch.no_grad():
        for images, _labels in loader:
            logits = model(images.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    acc = (y_pred == y_true).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n=== {name}: accuracy={acc:.4f} macro-F1={macro_f1:.4f} ===")
    for i, cname in enumerate(class_names):
        print(f"  {cname:<28} P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    ds_a = GraspRegionDataset(args.data_root, args.split, transform=build_transforms(args.image_size, train=False), letterbox=False)
    ds_b = GraspRegionDataset(args.data_root, args.split, transform=build_transforms(args.image_size, train=False), letterbox=True)
    class_names = ds_a.class_names_ordered()
    y_true = np.array([lbl for _fn, _seg, _box, lbl in ds_a.instances])

    probs_a = predict_probs(args.checkpoint_a, ds_a, len(class_names), device)
    probs_b = predict_probs(args.checkpoint_b, ds_b, len(class_names), device)

    aspect_ratios = np.array([max(w, h) / max(1, min(w, h)) for _fn, _seg, (x, y, w, h), _lbl in ds_a.instances])
    weight_b = np.clip((aspect_ratios - args.ratio_low) / (args.ratio_high - args.ratio_low), 0.0, 1.0) * 0.5 + 0.5
    weight_a = 1.0 - weight_b

    probs_flat = (probs_a + probs_b) / 2
    probs_weighted = weight_a[:, None] * probs_a + weight_b[:, None] * probs_b

    report("Flat 50/50 ensemble (existing)", y_true, probs_flat.argmax(axis=1), class_names)
    report("Aspect-ratio-weighted ensemble", y_true, probs_weighted.argmax(axis=1), class_names)

    elongated = aspect_ratios >= args.ratio_high
    print(f"\n--- On the {elongated.sum()} most elongated crops (ratio >= {args.ratio_high}) only ---")
    report("  flat", y_true[elongated], probs_flat.argmax(axis=1)[elongated], class_names)
    report("  weighted", y_true[elongated], probs_weighted.argmax(axis=1)[elongated], class_names)


if __name__ == "__main__":
    main()
