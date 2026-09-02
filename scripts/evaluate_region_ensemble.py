"""Averages predictions from two Task B checkpoints trained with different
crop preprocessing. Motivated by docs/error_analysis.md/DECISIONS.md,
2026-09-02: `region_baseline` (plain stretch-to-square) and
`region_letterbox_crop` (pad-to-square) make different mistakes -- the
letterbox model fixed the diagnosed aspect-ratio confusion pairs but
measurably hurt the two smallest classes (Clip Applier, Laparoscopic
Grasper), the plain model didn't. Same "architecture/preprocessing
diversity beats single-model tuning" pattern this project already found for
the segmentation ensemble (docs/DECISIONS.md, four-way Mask R-CNN + SAM2
ensemble) -- tested here for Task B instead of assumed to transfer.

Read-only: loads two checkpoints, runs inference, prints comparison tables.

Usage:
    python scripts/evaluate_region_ensemble.py \\
        --checkpoint-a experiments/region_baseline_20260831-182451/best.pt --letterbox-a=false \\
        --checkpoint-b experiments/region_letterbox_crop_20260902-152750/best.pt --letterbox-b=true
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
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--letterbox-a", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--letterbox-b", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--crop-mode-a", default="bbox")
    parser.add_argument("--crop-mode-b", default="bbox")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def predict_probs(checkpoint: Path, ds: GraspRegionDataset, class_names: list[str], device: torch.device) -> np.ndarray:
    model = build_model("mobilenet_v3_small", num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
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

    ds_a = GraspRegionDataset(
        args.data_root, args.split, transform=build_transforms(args.image_size, train=False),
        letterbox=args.letterbox_a, crop_mode=args.crop_mode_a,
    )
    ds_b = GraspRegionDataset(
        args.data_root, args.split, transform=build_transforms(args.image_size, train=False),
        letterbox=args.letterbox_b, crop_mode=args.crop_mode_b,
    )
    assert len(ds_a) == len(ds_b), "datasets must have the same instances in the same order to ensemble"
    class_names = ds_a.class_names_ordered()
    y_true = np.array([lbl for _fn, _seg, _box, lbl in ds_a.instances])

    probs_a = predict_probs(args.checkpoint_a, ds_a, class_names, device)
    probs_b = predict_probs(args.checkpoint_b, ds_b, class_names, device)
    probs_ensemble = (probs_a + probs_b) / 2

    report("Model A alone", y_true, probs_a.argmax(axis=1), class_names)
    report("Model B alone", y_true, probs_b.argmax(axis=1), class_names)
    y_pred_ensemble = probs_ensemble.argmax(axis=1)
    report("Ensemble (averaged softmax)", y_true, y_pred_ensemble, class_names)

    cm = confusion_matrix(y_true, y_pred_ensemble, labels=list(range(len(class_names))))
    print("\n--- Ensemble confusion matrix (rows=true, cols=predicted) ---")
    print("true\\pred".ljust(14) + "".join(f"{n[:10]:>12}" for n in class_names))
    for i, name in enumerate(class_names):
        print(name[:13].ljust(14) + "".join(f"{cm[i, j]:>12}" for j in range(len(class_names))))


if __name__ == "__main__":
    main()
