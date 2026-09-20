"""Whole-frame (Task A) baseline: accuracy on the official test set, from the saved
checkpoint of the weighted-loss + strong-augmentation run (docs/findings.md,
macro-F1 0.708). The run's manifest only recorded mean AP and macro-F1, so the
accuracy is measured here from the checkpoint: exact-match accuracy (every
instrument in the frame right) and per-label accuracy, at the 0.5 sigmoid
threshold the run's own evaluation uses. The rerun's macro-F1 (0.701) is within
0.007 of the recorded 0.708; the gap is a few rare-class frames flipping.

Usage:
    python scripts/eval_taskA_accuracy.py --run experiments/imbalance_weighted_loss_augmentation_20260831-173704
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
from sklearn.metrics import f1_score

from surgical_ai.data.dataset import GraspMultiLabelDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "final_pipeline" / "taskA_accuracy.json")
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text())
    cfg = manifest["config"]
    ds = GraspMultiLabelDataset(args.data_root, "test", transform=build_transforms(cfg["data"]["image_size"], train=False))
    model = build_model(cfg["model"]["name"], num_classes=ds.num_classes, pretrained=False, freeze_backbone=False).to(args.device)
    model.load_state_dict(torch.load(args.run / "best.pt", map_location=args.device), strict=False)
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    scores, truth = [], []
    with torch.no_grad():
        for x, y in loader:
            scores.append(torch.sigmoid(model(x.to(args.device))).cpu().numpy())
            truth.append(y.numpy())
    scores, truth = np.concatenate(scores), np.concatenate(truth).astype(int)
    pred = (scores >= 0.5).astype(int)
    out = {
        "n_frames": int(len(truth)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "exact_match_accuracy": float((pred == truth).all(axis=1).mean()),
        "per_label_accuracy": float((pred == truth).mean()),
        "recorded_macro_f1": manifest["final_metrics"]["macro_f1"],
    }
    print(json.dumps(out, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
