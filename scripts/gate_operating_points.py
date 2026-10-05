"""Epistemic-score operating points of an evidential ensemble's single-pass logits on the official test instruments: the share flagged at a
threshold, and the threshold that flags a given number (the reported headline tracked the 833 most uncertain of 2,861).

Usage (titanxp): python scripts/gate_operating_points.py --logits experiments/gtbox_sam/final/logits_s44.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from surgical_ai.evaluation.evidential import variance_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--budgets", type=int, nargs="*", default=[539, 833])
    args = ap.parse_args()
    z = np.load(args.logits)
    s1 = variance_scores(alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"]))["epistemic"][z["done"]]
    print(f"{len(s1)} instruments; flagged at tau {args.tau:.2e}: {int((s1 >= args.tau).sum())} ({100 * (s1 >= args.tau).mean():.1f}%)")
    ranked = np.sort(s1)[::-1]
    for b in args.budgets:
        print(f"top {b} ({100 * b / len(s1):.1f}%): the threshold is {ranked[b - 1]:.3e}")


if __name__ == "__main__":
    main()
