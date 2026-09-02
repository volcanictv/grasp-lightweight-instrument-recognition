"""Per-class decision threshold tuning for Task A (multi-label frame
classification), as a post-hoc calibration fix that touches neither
training dynamics nor pos_weight -- the promising alternative flagged
after focal loss's negative result (docs/error_analysis.md, docs/DECISIONS.md
2026-09-02).

Honesty note this script makes explicit rather than hiding: this
project's Task A checkpoints were trained with `data.split: official`
(train on the 8 official train cases, select/report on the 5 official
test cases), so there is no leakage-free held-out split left for this
specific checkpoint -- fold1/fold2 are carved from the same 8 train cases
it already saw in full. Thresholds are therefore tuned on the TRAIN
split's own predictions (in-sample), then applied to the official test
split. An "oracle" column (thresholds tuned directly on test) is printed
for comparison only, to show how much of the theoretical headroom the
in-sample thresholds actually capture -- never to be reported as an
achievable result, since tuning on the evaluation set is exactly the
leakage this project's splits policy exists to prevent.

Usage:
    python scripts/tune_per_class_thresholds.py \\
        experiments/imbalance_weighted_loss_augmentation_20260831-173704/best.pt \\
        [--only "Suction Instrument"]
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

from surgical_ai.data.dataset import GraspMultiLabelDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--tune-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--only", nargs="+", default=None, help="apply the tuned threshold only to these class names; every other class keeps 0.5")
    return parser.parse_args()


@torch.no_grad()
def get_scores(model, data_root: Path, split: str, image_size: int, device: torch.device):
    ds = GraspMultiLabelDataset(data_root, split, transform=build_transforms(image_size, train=False))
    class_names = ds.class_names_ordered()
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    all_scores, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels), class_names


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, f1_score(labels, (scores >= 0.5).astype(int), zero_division=0)
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = build_model("mobilenet_v3_small", num_classes=7, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_score_tune, y_true_tune, class_names = get_scores(model, args.data_root, args.tune_split, args.image_size, device)
    y_score_eval, y_true_eval, _ = get_scores(model, args.data_root, args.eval_split, args.image_size, device)

    tuned_thresholds = np.full(len(class_names), 0.5)
    for c, name in enumerate(class_names):
        if args.only and name not in args.only:
            continue
        t, train_f1 = best_threshold(y_score_tune[:, c], y_true_tune[:, c])
        tuned_thresholds[c] = t
        print(f"{name:<28} {args.tune_split}-optimal threshold={t:.2f} ({args.tune_split} F1 at that threshold={train_f1:.3f})")

    y_pred_flat = (y_score_eval >= 0.5).astype(int)
    _, _, f1_flat, _ = precision_recall_fscore_support(y_true_eval, y_pred_flat, average=None, zero_division=0)

    y_pred_tuned = (y_score_eval >= tuned_thresholds[None, :]).astype(int)
    _, _, f1_tuned, _ = precision_recall_fscore_support(y_true_eval, y_pred_tuned, average=None, zero_division=0)

    oracle_thresholds = np.array([best_threshold(y_score_eval[:, c], y_true_eval[:, c])[0] for c in range(len(class_names))])
    y_pred_oracle = (y_score_eval >= oracle_thresholds[None, :]).astype(int)
    _, _, f1_oracle, _ = precision_recall_fscore_support(y_true_eval, y_pred_oracle, average=None, zero_division=0)

    print(f"\n{'class':<28}{'flat 0.5 F1':>14}{'tuned F1':>12}{'oracle F1 (not achievable, reference only)':>44}")
    for c, name in enumerate(class_names):
        print(f"{name:<28}{f1_flat[c]:>14.3f}{f1_tuned[c]:>12.3f}{f1_oracle[c]:>44.3f}")
    print(f"{'macro-F1':<28}{f1_flat.mean():>14.3f}{f1_tuned.mean():>12.3f}{f1_oracle.mean():>44.3f}")


if __name__ == "__main__":
    main()
