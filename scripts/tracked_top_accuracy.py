"""Instance accuracy and macro-F1 when the N most uncertain instruments (evidential S1, per seed) take the fused label of a saved tracked run and the rest keep the single-pass label.
For runs whose accuracy was not stored (for example the augmented-view control, tracked dir tracked_tta[_s43|_s44]).

Usage: python scripts/tracked_top_accuracy.py --dir experiments/gtbox_sam/final --tracked tracked_tta --top 941 --out result.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from surgical_ai.evaluation.evidential import variance_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--tracked", required=True)
    ap.add_argument("--top", type=int, default=941)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]
    runs = []
    for seed in (42, 43, 44):
        z = np.load(args.dir / f"logits_s{seed}.npz")
        y, done = z["y"], z["done"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        base, s1 = mu.argmax(axis=1), variance_scores(alpha)["epistemic"]
        tracked = load_tracked(args.dir / (args.tracked + ("" if seed == 42 else f"_s{seed}")))
        pred = base.copy()
        top = [int(i) for i in np.argsort(-s1, kind="stable")[: args.top] if done[i]]
        for i in top:
            if i in tracked:
                pred[i] = fuse(tracked[i], weights)[0]
        runs.append({"seed": seed, "accuracy": float((pred[done] == y[done]).mean()), "macro_f1": float(f1_score(y[done], pred[done], average="macro", labels=range(7))),
                     "missing_tracks": sum(i not in tracked for i in top)})
    out = {"accuracy": float(np.mean([r["accuracy"] for r in runs])), "macro_f1": float(np.mean([r["macro_f1"] for r in runs])), "per_seed": runs}
    print(f"{args.tracked} top {args.top}: acc {100 * out['accuracy']:.2f} F1 {100 * out['macro_f1']:.2f} missing {[r['missing_tracks'] for r in runs]}")
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
