"""Confirms (or refutes) whether the contrastive fine-tune
(`scripts/finetune_grasper_suction_contrastive.py`) actually moved the
Laparoscopic Grasper / Suction Instrument confusion, not just the
aggregate macro-F1 the training log already reports per epoch. Compares
the base checkpoint against the fine-tuned one on the same official test
set: full confusion matrix (the Grasper->Suction cell specifically), and
the known hard example (CASE053/04165.jpg, docs/DECISIONS.md 2026-09-03)
that a wide-window temporal search already failed to fix.

Usage:
    python scripts/evaluate_grasper_suction_finetune.py \\
        --finetuned experiments/region_grasper_suction_contrastive/best.pt
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
    parser.add_argument("--base", type=Path, default=REPO_ROOT / "experiments" / "region_letterbox_resnet50_320_20260902-234246" / "best.pt")
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--hard-example", default="CASE053/04165.jpg")
    return parser.parse_args()


def evaluate_checkpoint(checkpoint: Path, ds, class_names, device, image_size) -> tuple[np.ndarray, np.ndarray]:
    model = build_model("resnet50", num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    preds, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            logits = model(images.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def find_hard_example_probs(checkpoint: Path, ds, class_names, device, hard_example: str):
    """A frame can hold multiple annotated instruments -- return every
    instance whose file matches, not just the first, so the caller can
    identify which one is the actually-diagnosed Grasper/Suction case."""
    indices = [i for i, (fn, *_r) in enumerate(ds.instances) if hard_example in fn]
    if not indices:
        return None
    model = build_model("resnet50", num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()
    results = []
    for idx in indices:
        image, label = ds[idx]
        with torch.no_grad():
            probs = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
        results.append((probs, label.item()))
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    ds = GraspRegionDataset(
        args.data_root, args.split, transform=build_transforms(args.image_size, train=False), letterbox=True,
    )
    class_names = ds.class_names_ordered()
    grasper_idx = class_names.index("Laparoscopic Grasper")
    suction_idx = class_names.index("Suction Instrument")

    for label, checkpoint in [("base", args.base), ("finetuned", args.finetuned)]:
        y_pred, y_true = evaluate_checkpoint(checkpoint, ds, class_names, device, args.image_size)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc = (y_pred == y_true).mean()
        _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        print(f"\n=== {label}: {checkpoint} ===")
        print(f"accuracy={acc:.4f} macro-F1={macro_f1:.4f} grasper_f1={f1_per_class[grasper_idx]:.4f} suction_f1={f1_per_class[suction_idx]:.4f}")
        print(f"true Grasper -> pred Suction: {cm[grasper_idx, suction_idx]} / {cm[grasper_idx].sum()} true Grasper instances")
        print(f"true Suction -> pred Grasper: {cm[suction_idx, grasper_idx]} / {cm[suction_idx].sum()} true Suction instances")

        results = find_hard_example_probs(checkpoint, ds, class_names, device, args.hard_example)
        if results is None:
            print(f"hard example {args.hard_example} not found in {args.split} split")
        else:
            for probs, true_label in results:
                pred_label = class_names[probs.argmax()]
                print(f"hard example {args.hard_example} [true={class_names[true_label]}]: pred={pred_label} "
                      f"P(Grasper)={probs[grasper_idx]:.3f} P(Suction)={probs[suction_idx]:.3f}")


if __name__ == "__main__":
    main()
