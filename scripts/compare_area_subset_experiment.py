"""Compares a candidate checkpoint against a baseline checkpoint on the
small-mask-area subset specifically (not the whole split) -- the metric
`docs/DECISIONS.md`'s area-oversampling entries (2026-09-04/05) use to
judge whether a fix actually helps the targeted population, since
aggregate accuracy/macro-F1 can hide a fix that trades one area-sensitive
class against another.

Usage:
    python scripts/compare_area_subset_experiment.py \\
        --baseline-checkpoint experiments/region_letterbox_resnet50_320_fold1_.../best.pt \\
        --candidate-checkpoint experiments/region_area_oversample_classcond_fold1_.../best.pt \\
        --split fold1 --label "class-conditional area oversample"
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

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--split", default="fold1")
    parser.add_argument("--area-threshold", type=float, default=0.04)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def predict(checkpoint: Path, model_name: str, image_size: int, split: str, data_root: Path, device: torch.device):
    ds = GraspRegionDataset(data_root, split, transform=build_transforms(image_size, train=False), letterbox=True)
    class_names = ds.class_names_ordered()
    y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
    area_frac = np.array([
        decode_instance_mask(seg).sum() / (seg["size"][0] * seg["size"][1])
        for _fn, seg, _box, _lbl in ds.instances
    ])

    model = build_model(model_name, num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    preds = []
    with torch.no_grad():
        for images, _labels in loader:
            logits = model(images.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return class_names, y_true, y_pred, area_frac


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    base_names, base_y_true, base_y_pred, base_area = predict(
        args.baseline_checkpoint, args.model, args.image_size, args.split, args.data_root, device
    )
    cand_names, cand_y_true, cand_y_pred, _cand_area = predict(
        args.candidate_checkpoint, args.model, args.image_size, args.split, args.data_root, device
    )
    assert base_names == cand_names and (base_y_true == cand_y_true).all(), "class order or instance order mismatch"

    small = base_area < args.area_threshold
    n_small = int(small.sum())
    print(f"split={args.split}  small-area subset (< {args.area_threshold*100:.0f}%): n={n_small}/{len(base_y_true)}\n")

    base_small_acc = float((base_y_pred[small] == base_y_true[small]).mean())
    cand_small_acc = float((cand_y_pred[small] == cand_y_true[small]).mean())
    print(f"baseline small-area accuracy:                {base_small_acc:.4f}")
    print(f"{args.label} small-area accuracy: {cand_small_acc:.4f}  (delta {cand_small_acc - base_small_acc:+.4f})\n")

    print("per-class F1 on the small-area subset:")
    labels = list(range(len(base_names)))
    _bp, _br, base_f1, base_support = precision_recall_fscore_support(
        base_y_true[small], base_y_pred[small], labels=labels, average=None, zero_division=0
    )
    _cp, _cr, cand_f1, _cand_support = precision_recall_fscore_support(
        cand_y_true[small], cand_y_pred[small], labels=labels, average=None, zero_division=0
    )
    for i, name in enumerate(base_names):
        print(f"  {name:<28} support={base_support[i]:<4} baseline={base_f1[i]:.3f}  candidate={cand_f1[i]:.3f}  delta={cand_f1[i] - base_f1[i]:+.3f}")

    base_macro = float(f1_score(base_y_true[small], base_y_pred[small], average="macro", zero_division=0))
    cand_macro = float(f1_score(cand_y_true[small], cand_y_pred[small], average="macro", zero_division=0))
    print(f"\nmacro-F1 on small-area subset: baseline={base_macro:.4f}  candidate={cand_macro:.4f}  delta={cand_macro - base_macro:+.4f}")


if __name__ == "__main__":
    main()
