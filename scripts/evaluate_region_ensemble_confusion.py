"""Confusion matrix for the actual reported weighted ensemble (2026-09-04
feedback: report it normalized, not raw counts -- raw counts conflate a
class's confusion rate with its support). Prior confusion-pair numbers in
docs/error_analysis.md predate the 4-model ensemble and come from a single
Task B model; this regenerates them against the current ensemble so the
top confused pairs used for follow-up work (e.g. Grad-CAM) reflect what's
actually reported.

Usage:
    python scripts/evaluate_region_ensemble_confusion.py
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
from sklearn.metrics import confusion_matrix

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model
from surgical_ai.utils.visualization import plot_multiclass_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--weight-320", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "figures")
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
        all_probs.append(np.concatenate(probs))

    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_multiclass_confusion_matrix(cm, class_names, args.out_dir / "region_ensemble_confusion_raw.png", normalize=False)
    plot_multiclass_confusion_matrix(cm, class_names, args.out_dir / "region_ensemble_confusion_normalized.png", normalize=True)

    row_sums = cm.sum(axis=1)
    pairs = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append({
                "true": true_name, "predicted": pred_name, "count": int(cm[i, j]),
                "pct_of_true_class": float(cm[i, j] / row_sums[i] * 100) if row_sums[i] else 0.0,
            })
    pairs.sort(key=lambda p: -p["count"])

    (args.out_dir.parent / "region_ensemble_confusion.json").write_text(json.dumps({
        "class_names": class_names, "confusion_matrix": cm.tolist(), "top_confused_pairs": pairs[:10],
    }, indent=2))

    print(f"accuracy={float((y_pred == y_true).mean()):.4f}")
    print("\ntop confused pairs (current weighted ensemble, official test):")
    print("| true class | predicted as | count | % of true class |")
    print("|---|---|---|---|")
    for p in pairs[:8]:
        print(f"| {p['true']} | {p['predicted']} | {p['count']} | {p['pct_of_true_class']:.1f}% |")
    print(f"\nwrote {args.out_dir / 'region_ensemble_confusion_raw.png'}")
    print(f"wrote {args.out_dir / 'region_ensemble_confusion_normalized.png'}")
    print(f"wrote {args.out_dir.parent / 'region_ensemble_confusion.json'}")


if __name__ == "__main__":
    main()
