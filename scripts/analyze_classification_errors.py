"""Root-cause error analysis for Task A (multi-label frame classification)
and Task B (region classification) on a GraSP split, using already-trained
checkpoints. Answers two questions the aggregate metrics in
docs/findings.md don't: does Task A's recall track how many instruments
co-occur in the frame, and which specific classes does Task B actually
confuse with which (not just which classes have low F1). See
docs/error_analysis.md for the results this produced and
docs/DECISIONS.md, 2026-09-02.

Read-only: loads checkpoints, runs inference, prints tables.

Usage:
    python scripts/analyze_classification_errors.py \\
        --task-a-checkpoint experiments/imbalance_weighted_loss_augmentation_20260831-173704/best.pt \\
        --task-b-checkpoint experiments/region_baseline_20260831-182451/best.pt \\
        [--split test] [--data-root PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from surgical_ai.data.dataset import GraspMultiLabelDataset
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-a-checkpoint", type=Path, default=None)
    parser.add_argument("--task-b-checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--task-b-letterbox", action="store_true",
        help="match GraspRegionDataset(letterbox=True) -- use when --task-b-checkpoint was trained with data.letterbox_crop: true",
    )
    return parser.parse_args()


def load_classifier(name: str, num_classes: int, checkpoint: Path, device: torch.device):
    model = build_model(name, num_classes=num_classes, pretrained=False, freeze_backbone=False).to(device)
    # Checkpoints saved before models/classifiers/common.py's head-aliasing fix (docs/DECISIONS.md)
    # carry extra head.* keys duplicating classifier.*/fc.* weights -- strict=False ignores those
    # with no information loss (they're exact duplicates, not missing weights).
    missing, _unexpected = model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    assert not missing, f"missing keys loading {checkpoint}: {missing}"
    model.eval()
    return model


def run_task_a(args: argparse.Namespace, device: torch.device) -> None:
    print("=" * 70)
    print("TASK A -- multi-label frame classification error analysis")
    print("=" * 70)

    ds = GraspMultiLabelDataset(args.data_root, args.split, transform=build_transforms(args.image_size, train=False))
    class_names = ds.class_names_ordered()
    num_classes = len(class_names)

    model = load_classifier("mobilenet_v3_small", num_classes, args.task_a_checkpoint, device)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    y_pred = (y_score >= args.threshold).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"sanity check -- macro-F1 = {macro_f1:.4f}\n")

    n_instruments_per_frame = y_true.sum(axis=1).astype(int)

    print("--- Recall per class, stratified by how many instruments co-occur in that frame ---")
    print(f"{'class':<28}{'n=1':>16}{'n=2':>16}{'n=3+':>16}")
    for c, cname in enumerate(class_names):
        row = f"{cname:<28}"
        for cond in [n_instruments_per_frame == 1, n_instruments_per_frame == 2, n_instruments_per_frame >= 3]:
            mask = cond & (y_true[:, c] == 1)
            n = mask.sum()
            row += f"{'n/a':>16}" if n == 0 else f"{y_pred[mask, c].mean():>10.3f}({n})"
        print(row)

    print("\n--- Overall (all classes pooled) recall vs. number of co-occurring instruments ---")
    for k in sorted(set(n_instruments_per_frame.tolist())):
        mask_frames = n_instruments_per_frame == k
        present = y_true[mask_frames] == 1
        if present.sum() == 0:
            continue
        recall = y_pred[mask_frames][present].mean()
        print(f"  {k} instrument(s) in frame: recall={recall:.3f}  (n_frames={int(mask_frames.sum())})")

    print("\n--- False-negative co-occurrence: when class X is missed, what else is in the frame? ---")
    base_rate = y_true.mean(axis=0)
    for c, cname in enumerate(class_names):
        fn_mask = (y_true[:, c] == 1) & (y_pred[:, c] == 0)
        n_fn = fn_mask.sum()
        if n_fn < 5:
            continue
        co_rate = y_true[fn_mask].mean(axis=0)
        lift = [(class_names[j], co_rate[j], base_rate[j]) for j in range(num_classes) if j != c and base_rate[j] > 0]
        lift.sort(key=lambda t: -(t[1] / t[2]))
        top = lift[:3]
        print(f"  {cname} (n_fn={n_fn}): " + ", ".join(f"{n}={r:.2f} (base {b:.2f})" for n, r, b in top))


def run_task_b(args: argparse.Namespace, device: torch.device) -> None:
    print("\n" + "=" * 70)
    print("TASK B -- region classification error analysis")
    print("=" * 70)

    ds = GraspRegionDataset(
        args.data_root, args.split, transform=build_transforms(args.image_size, train=False),
        letterbox=args.task_b_letterbox,
    )
    class_names = ds.class_names_ordered()

    model = load_classifier("mobilenet_v3_small", len(class_names), args.task_b_checkpoint, device)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = (y_pred == y_true).mean()
    print(f"sanity check -- accuracy = {acc:.4f}\n")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("--- Full confusion matrix (rows=true, cols=predicted) ---")
    print("true\\pred".ljust(14) + "".join(f"{n[:10]:>12}" for n in class_names))
    for i, name in enumerate(class_names):
        print(name[:13].ljust(14) + "".join(f"{cm[i, j]:>12}" for j in range(len(class_names))))

    print("\n--- Top confusion pairs (true -> predicted, off-diagonal) ---")
    pairs = [(cm[i, j], class_names[i], class_names[j]) for i in range(len(class_names)) for j in range(len(class_names)) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    for count, true_name, pred_name in pairs[:15]:
        n_true = cm[class_names.index(true_name)].sum()
        print(f"  {true_name} -> predicted {pred_name}: {count} ({100*count/n_true:.1f}% of all {true_name})")

    correct = y_pred == y_true
    areas = np.array([w * h for _fn, _seg, (x, y, w, h), _lbl in ds.instances])
    quartiles = np.percentile(areas, [25, 50, 75])
    buckets_area = np.digitize(areas, quartiles)
    print("\n--- Error rate by instance bbox area (crop size) ---")
    for q in range(4):
        mask = buckets_area == q
        if mask.sum() == 0:
            continue
        print(f"  area quartile {q+1} (median={np.median(areas[mask]):.0f}px^2, n={mask.sum()}): error rate={1 - correct[mask].mean():.3f}")

    cases = np.array([fn.split("/")[0] for fn, _seg, _box, _lbl in ds.instances])
    case_errors = defaultdict(lambda: [0, 0])
    for case, is_correct in zip(cases, correct):
        case_errors[case][1] += 1
        if not is_correct:
            case_errors[case][0] += 1
    print("\n--- Error concentration by case ---")
    for case in sorted(case_errors, key=lambda c: -case_errors[c][0] / case_errors[c][1]):
        errs, n = case_errors[case]
        print(f"  {case:<10}n={n:<6}errors={errs:<6}error_rate={errs/n:.3f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}\n")
    if args.task_a_checkpoint:
        run_task_a(args, device)
    if args.task_b_checkpoint:
        run_task_b(args, device)


if __name__ == "__main__":
    main()
