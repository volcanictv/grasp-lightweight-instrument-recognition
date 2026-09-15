"""Compares a candidate fix against the fold1 baseline on both aggregate
metrics and the specific residual confusion pairs it's meant to help
(2026-09-04 overnight investigation) -- aggregate macro-F1 alone can hide
whether a change actually helped the targeted pairs or just moved noise
around elsewhere.

Usage:
    python scripts/compare_fold1_experiment.py \\
        --baseline-checkpoint experiments/region_letterbox_resnet50_320_fold1_20260903-012254/best.pt \\
        --baseline-image-size 320 \\
        --candidate-checkpoint experiments/region_letterbox_resnet50_384_fold1_.../best.pt \\
        --candidate-image-size 384 \\
        --label "384px resolution"
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
from sklearn.metrics import confusion_matrix, f1_score

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

TARGET_PAIRS = [
    ("Prograsp Forceps", "Bipolar Forceps"),
    ("Bipolar Forceps", "Large Needle Driver"),
    ("Monopolar Curved Scissors", "Suction Instrument"),
    ("Large Needle Driver", "Bipolar Forceps"),
    ("Suction Instrument", "Prograsp Forceps"),
    ("Bipolar Forceps", "Prograsp Forceps"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-image-size", type=int, required=True)
    parser.add_argument("--baseline-model", default="resnet50")
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-image-size", type=int, required=True)
    parser.add_argument("--candidate-model", default="resnet50")
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", default="fold1")
    parser.add_argument("--letterbox", action="store_true", default=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(checkpoint: Path, image_size: int, letterbox: bool, split: str, data_root: Path, device: torch.device, model_name: str = "resnet50"):
    ds = GraspRegionDataset(data_root, split, transform=build_transforms(image_size, train=False), letterbox=letterbox)
    class_names = ds.class_names_ordered()
    y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])

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

    acc = float((y_pred == y_true).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return class_names, y_true, y_pred, acc, macro_f1, cm


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    base_names, base_y_true, base_y_pred, base_acc, base_f1, base_cm = evaluate(
        args.baseline_checkpoint, args.baseline_image_size, args.letterbox, args.split, args.data_root, device, args.baseline_model
    )
    cand_names, cand_y_true, cand_y_pred, cand_acc, cand_f1, cand_cm = evaluate(
        args.candidate_checkpoint, args.candidate_image_size, args.letterbox, args.split, args.data_root, device, args.candidate_model
    )
    assert base_names == cand_names and (base_y_true == cand_y_true).all(), "class order or instance order mismatch"

    print(f"baseline (320px):        accuracy={base_acc:.4f} macro_f1={base_f1:.4f}")
    print(f"candidate ({args.label}): accuracy={cand_acc:.4f} macro_f1={cand_f1:.4f}")
    print(f"delta: accuracy={cand_acc - base_acc:+.4f} macro_f1={cand_f1 - base_f1:+.4f}\n")

    print("per-class F1:")
    for i, name in enumerate(base_names):
        base_cls_f1 = f1_score(base_y_true == i, base_y_pred == i, zero_division=0)
        cand_cls_f1 = f1_score(cand_y_true == i, cand_y_pred == i, zero_division=0)
        print(f"  {name:<28} baseline={base_cls_f1:.3f}  candidate={cand_cls_f1:.3f}  delta={cand_cls_f1 - base_cls_f1:+.3f}")

    print("\ntargeted confusion pairs (count, lower is better):")
    for true_name, pred_name in TARGET_PAIRS:
        i, j = base_names.index(true_name), base_names.index(pred_name)
        base_ct, cand_ct = int(base_cm[i, j]), int(cand_cm[i, j])
        print(f"  {true_name} -> {pred_name}: baseline={base_ct}  candidate={cand_ct}  delta={cand_ct - base_ct:+d}")


if __name__ == "__main__":
    main()
