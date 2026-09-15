"""Where does accuracy actually fall off a cliff as a function of mask
area, rather than assuming a round number like 2% (2026-09-04 feedback:
find the real breakpoint empirically, don't guess it, so the abstention
cutoff isn't arbitrary).

Two complementary views, both computed once from the existing weighted
ensemble (no retraining needed -- this is a property of the data/model as
they already stand):

1. Rolling-window accuracy: sort test instances by mask-area fraction,
   slide a fixed-size window over them, plot windowed accuracy against
   the window's mean area fraction. Robust to bin-count sparsity near the
   low end where most of the signal is, unlike fixed-width bins.
2. Fixed-band accuracy: non-overlapping bands (0-0.5%, 0.5-1%, ... ) with
   n and accuracy per band, for a readable table alongside the plot.

Also reports the cumulative "abstain below t" coverage/accuracy curve
(same shape as evaluate_region_area_filter.py, but swept across many t
instead of one), and breaks the rolling curve out per-class for the two
classes supervisors flagged (Suction Instrument, Laparoscopic Grasper) to
check whether they crater at the same point as everything else or are
just naturally small without being equally hard.

Usage:
    python scripts/evaluate_region_area_threshold_sweep.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--weight-320", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=150, help="rolling-window size, in instances")
    parser.add_argument("--step", type=int, default=20, help="rolling-window stride, in instances")
    parser.add_argument("--max-area-pct", type=float, default=8.0, help="zoom the plot's x-axis to [0, this] percent -- most of the action is well below 8%")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "figures")
    return parser.parse_args()


def mask_area_fractions(ds: GraspRegionDataset) -> np.ndarray:
    fracs = np.zeros(len(ds.instances), dtype=np.float64)
    for i, (_file_name, segmentation, _box, _label_idx) in enumerate(ds.instances):
        h, w = segmentation["size"]
        fracs[i] = decode_instance_mask(segmentation).sum() / (h * w)
    return fracs


def rolling_accuracy(area_pct: np.ndarray, correct: np.ndarray, window: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(area_pct)
    area_sorted = area_pct[order]
    correct_sorted = correct[order]
    n = len(area_sorted)
    centers, accs, ns = [], [], []
    for start in range(0, max(1, n - window + 1), step):
        end = min(start + window, n)
        centers.append(float(area_sorted[start:end].mean()))
        accs.append(float(correct_sorted[start:end].mean()))
        ns.append(end - start)
    return np.array(centers), np.array(accs), np.array(ns)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.config.read_text())
    members = ensemble_config["members"]
    weight_320 = args.weight_320 if args.weight_320 is not None else ensemble_config["weight_resnet50_320"]

    class_names = None
    y_true = None
    area_frac = None
    all_probs = []
    for member in members:
        ds = GraspRegionDataset(
            args.data_root, args.split, transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            area_frac = mask_area_fractions(ds)

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
    correct = (y_pred == y_true).astype(np.float64)
    area_pct = area_frac * 100

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. rolling-window curve, overall
    centers, accs, ns = rolling_accuracy(area_pct, correct, args.window, args.step)

    # per-class rolling curves, all classes -- which ones actually drive the
    # population-wide crater matters for whether abstention should be
    # uniform or class-conditional (2026-09-04: found Suction Instrument
    # does not crater at all, so a blanket rule would cost it coverage for
    # zero benefit -- check every class rather than assume the pattern
    # generalizes beyond the two initially flagged).
    flagged = {}
    for cls_name in class_names:
        idx = class_names.index(cls_name)
        mask = y_true == idx
        if mask.sum() >= args.window:
            c, a, n = rolling_accuracy(area_pct[mask], correct[mask], min(args.window, mask.sum()), max(1, args.step // 2))
            flagged[cls_name] = (c, a, n)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(centers, accs, color="#1f3a5f", linewidth=2, label=f"overall (window={args.window})")
    for cls_name, (c, a, _n) in flagged.items():
        ax.plot(c, a, linewidth=1.3, linestyle="--", label=cls_name, alpha=0.85)
    ax.axhline(float(correct.mean()), color="gray", linestyle=":", linewidth=1, label=f"overall accuracy ({correct.mean():.3f})")
    ax.set_xlim(0, args.max_area_pct)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("mask area, % of frame (rolling-window mean)")
    ax.set_ylabel("rolling-window accuracy")
    ax.set_title("Region-classifier accuracy vs. instance mask area")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "area_threshold_rolling_accuracy.png", dpi=150)
    plt.close(fig)

    # 2. fixed non-overlapping bands, for a readable table
    band_edges = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 100.0]
    band_rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        in_band = (area_pct >= lo) & (area_pct < hi)
        n = int(in_band.sum())
        band_rows.append({
            "band_low_pct": lo, "band_high_pct": hi, "n": n,
            "accuracy": float(correct[in_band].mean()) if n else None,
        })

    # 3. cumulative "abstain below t" coverage/accuracy curve
    cumulative_ts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    cumulative_rows = []
    for t in cumulative_ts:
        covered = area_pct >= t
        cumulative_rows.append({
            "threshold_pct": t, "coverage_pct": float(covered.mean() * 100),
            "accuracy_covered": float(correct[covered].mean()) if covered.any() else None,
            "accuracy_abstained": float(correct[~covered].mean()) if (~covered).any() else None,
        })

    result = {
        "rolling": {"area_pct_centers": centers.tolist(), "accuracy": accs.tolist(), "window_n": ns.tolist()},
        "rolling_per_class": {
            name: {"area_pct_centers": c.tolist(), "accuracy": a.tolist(), "window_n": n.tolist()}
            for name, (c, a, n) in flagged.items()
        },
        "fixed_bands": band_rows,
        "cumulative_abstain_below": cumulative_rows,
    }
    (args.out_dir.parent / "area_threshold_sweep.json").write_text(json.dumps(result, indent=2))

    print("fixed non-overlapping bands:")
    print("| band | n | accuracy |")
    print("|---|---|---|")
    for row in band_rows:
        acc_str = f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "n/a"
        print(f"| {row['band_low_pct']:.1f}-{row['band_high_pct']:.1f}% | {row['n']} | {acc_str} |")

    print("\ncumulative: accuracy if abstaining below threshold t, and coverage lost:")
    print("| t | coverage | accuracy (t and above) | accuracy (below t, abstained group) |")
    print("|---|---|---|---|")
    for row in cumulative_rows:
        acc_c = f"{row['accuracy_covered']:.3f}" if row["accuracy_covered"] is not None else "n/a"
        acc_a = f"{row['accuracy_abstained']:.3f}" if row["accuracy_abstained"] is not None else "n/a"
        print(f"| {row['threshold_pct']:.1f}% | {row['coverage_pct']:.1f}% | {acc_c} | {acc_a} |")

    print(f"\nwrote {args.out_dir / 'area_threshold_rolling_accuracy.png'}")
    print(f"wrote {args.out_dir.parent / 'area_threshold_sweep.json'}")


if __name__ == "__main__":
    main()
