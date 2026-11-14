"""Per-class recall of the single pass and of the uncertainty-triggered revisit (S1 >= tau, tracked label for the gated instruments), errors fixed and correct labels broken, mean over seeds.
Usage: python scripts/revisit_per_class.py --dir experiments/gtbox_sam/gtmask --out result.json
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

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.evaluation.evidential import variance_scores

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]
    rows = []
    for seed in (42, 43, 44):
        z = np.load(args.dir / f"logits_s{seed}.npz")
        y, done = z["y"], z["done"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        base = (alpha / alpha.sum(axis=1, keepdims=True)).argmax(axis=1)
        s1 = variance_scores(alpha)["epistemic"]
        tracked: dict = {}
        for d in TRACK_DIRS[seed]:
            tracked.update(load_tracked(args.dir / d))
        pred = base.copy()
        flag = (s1 >= args.tau) & done
        for i in np.where(flag)[0]:
            pred[i] = fuse(tracked[int(i)], weights)[0]
        recall = lambda p: [float((p[(y == c) & done] == c).mean()) for c in range(7)]
        rows.append({"seed": seed, "support": [int(((y == c) & done).sum()) for c in range(7)], "recall_single": recall(base), "recall_revisit": recall(pred),
                     "revisited": int(flag.sum()), "fixed": int(((base != y) & (pred == y) & done).sum()), "broken": int(((base == y) & (pred != y) & done).sum()),
                     "still_wrong": int(((pred != y) & done).sum()), "confusions_after": {f"{CLASSES[a]} -> {CLASSES[b]}": int(((y == a) & (pred == b) & done).sum()) for a in range(7) for b in range(7) if a != b and ((y == a) & (pred == b) & done).sum() >= 5}})
    out = {"classes": CLASSES, "recall_single": [float(np.mean([r["recall_single"][c] for r in rows])) for c in range(7)], "recall_revisit": [float(np.mean([r["recall_revisit"][c] for r in rows])) for c in range(7)],
           "support": rows[0]["support"], **{k: float(np.mean([r[k] for r in rows])) for k in ("revisited", "fixed", "broken", "still_wrong")}, "per_seed": rows}
    for c in range(7):
        print(f"{CLASSES[c]:<28} n={out['support'][c]:>4}  recall {100 * out['recall_single'][c]:.1f} -> {100 * out['recall_revisit'][c]:.1f}")
    print({k: round(out[k], 1) for k in ("revisited", "fixed", "broken", "still_wrong")})
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
