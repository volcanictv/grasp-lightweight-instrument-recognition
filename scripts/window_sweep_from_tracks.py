"""Temporal-window and causality sweep for the 941 most uncertain instruments, from the saved +-10 s refine-all tracks, with no new tracking or classifier passes.

SAM2 propagates the keyframe mask forwards and backwards independently, so the +-k s window is the subset of the +-10 s track with |offset| <= k, and the causal "last k" window is the subset with
-k <= offset <= 0. The per-frame logits of a track are stored in the order of its valid (non-empty-mask) offsets; the offsets come from the saved masks. Reports instance accuracy and macro-F1 of the
fused labels of the top-N instruments by the evidential score S1 (the rest keep the single-pass label), mean over seeds.

Usage (titanxp, repo root): python scripts/window_sweep_from_tracks.py --dir experiments/gtbox_sam/final --out result.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.evaluation.evidential import variance_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--top", type=int, default=941)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--causal-dir", default=None, help="name of a separate causal tracking run (past 20 frames; <name>, <name>_s43, <name>_s44); the causal windows are cut from it. Default: cut from the +-10 s tracks")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]

    def read_offsets(dirs: list[str]) -> dict[int, list[int]]:
        found: dict[int, list[int]] = {}
        for d in dirs:
            for path in sorted((args.dir / d).glob("masks_shard*.pkl")):
                for idx, frames in pickle.loads(path.read_bytes())["masks"].items():
                    found[idx] = [off for off in sorted(frames) if off == 0 or mask_codec.area(frames[off][1]) > 0]
        return found

    offsets = read_offsets(TRACK_DIRS[42])
    causal_offsets = read_offsets([args.causal_dir]) if args.causal_dir else offsets

    results: dict = {}
    for seed in args.seeds:
        z = np.load(args.dir / f"logits_s{seed}.npz")
        y, done = z["y"], z["done"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        base = mu.argmax(axis=1)
        s1 = variance_scores(alpha)["epistemic"]
        tracked: dict = {}
        for d in TRACK_DIRS[seed]:
            tracked.update(load_tracked(args.dir / d))
        causal_tracked = tracked
        if args.causal_dir:
            causal_tracked = load_tracked(args.dir / (args.causal_dir + ("" if seed == 42 else f"_s{seed}")))
        top = [int(i) for i in np.argsort(-s1, kind="stable")[: args.top] if done[i]]
        missing = sum(i not in tracked for i in top) + (sum(i not in causal_tracked for i in top) if args.causal_dir else 0)
        mismatch = sum(i in tracked and len(offsets.get(i, [])) != tracked[i].shape[0] for i in top)
        if args.causal_dir:
            mismatch += sum(i in causal_tracked and len(causal_offsets.get(i, [])) != causal_tracked[i].shape[0] for i in top)
        specs = [("none", 0)] + [(m, k) for m in ("sym", "causal") for k in args.windows]
        for mode, k in specs:
            pred = base.copy()
            trk, offs = (causal_tracked, causal_offsets) if mode == "causal" else (tracked, offsets)
            for i in top:
                if mode == "none" or i not in trk or len(offs.get(i, [])) != trk[i].shape[0]:
                    continue
                keep = [p for p, off in enumerate(offs[i]) if (abs(off) <= k if mode == "sym" else -k <= off <= 0)]
                pred[i] = fuse(trk[i][keep], weights)[0]
            results.setdefault(f"{mode}{k}" if mode != "none" else "none", []).append(
                {"seed": seed, "accuracy": float((pred[done] == y[done]).mean()),
                 "macro_f1": float(f1_score(y[done], pred[done], average="macro", labels=range(7))), "missing_tracks": missing, "offset_mismatch": mismatch})
    out = {name: {"accuracy": float(np.mean([r["accuracy"] for r in rs])), "macro_f1": float(np.mean([r["macro_f1"] for r in rs])), "per_seed": rs} for name, rs in results.items()}
    for name, r in out.items():
        print(f"{name:<10} acc {100 * r['accuracy']:.2f}  F1 {100 * r['macro_f1']:.2f}  missing {[x['missing_tracks'] for x in r['per_seed']]} mismatch {[x['offset_mismatch'] for x in r['per_seed']]}")
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
