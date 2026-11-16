"""Errors fixed (wrong single-pass label, right revisited label) and introduced (right, then wrong), per seed, mean over seeds, for (a) the S ordering at a 45% share on the refine-all tracks,
(b) the 941 most uncertain instruments with the symmetric +-10 s window and the causal last-10 window.
Usage: python scripts/revisit_error_counts.py --dir experiments/gtbox_sam/gtmask --causal-dir tracked_causal"""
import argparse, pickle, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src")); sys.path.insert(0, str(REPO_ROOT / "scripts"))
import numpy as np
from pycocotools import mask as mc
from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.evaluation.evidential import variance_scores
ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, required=True); ap.add_argument("--causal-dir", required=True); a = ap.parse_args()
W = CONFIGS["four"]
def offsets_of(dirs):
    out = {}
    for d in dirs:
        for p in sorted((a.dir / d).glob("masks_shard*.pkl")):
            for i, fr in pickle.loads(p.read_bytes())["masks"].items(): out[i] = [o for o in sorted(fr) if o == 0 or mc.area(fr[o][1]) > 0]
    return out
off_s, off_c = offsets_of(TRACK_DIRS[42]), offsets_of([a.causal_dir])
rows = {"S45": [], "sym10": [], "causal10": []}
for seed in (42, 43, 44):
    z = np.load(a.dir / f"logits_s{seed}.npz"); y, done = z["y"], z["done"]; al = alpha_mix(lambda k: z["det_" + k], W); base = (al / al.sum(1, keepdims=True)).argmax(1); s1 = variance_scores(al)["epistemic"]
    tr = {}
    for d in TRACK_DIRS[seed]: tr.update(load_tracked(a.dir / d))
    trc = load_tracked(a.dir / (a.causal_dir + ("" if seed == 42 else f"_s{seed}")))
    def count(idx, trk, offs, keep):
        pred = base.copy()
        for i in idx:
            sel = [p for p, o in enumerate(offs[i]) if keep(o)] if keep else list(range(trk[i].shape[0]))
            pred[i] = fuse(trk[i][sel], W)[0]
        m = np.zeros(len(y), bool); m[list(idx)] = True
        return int(((base != y) & (pred == y) & m).sum()), int(((base == y) & (pred != y) & m).sum())
    order = [int(i) for i in np.argsort(-s1, kind="stable") if done[i] and i in tr]
    rows["S45"].append(count(order[: int(round(0.45 * len(order)))], tr, off_s, None))
    top = [int(i) for i in np.argsort(-s1, kind="stable")[:941] if done[i]]
    rows["sym10"].append(count(top, tr, off_s, lambda o: abs(o) <= 10)); rows["causal10"].append(count(top, trc, off_c, lambda o: -10 <= o <= 0))
for k, v in rows.items(): print(f"{k:<9} per seed (fixed, introduced) {v}  mean fixed {np.mean([x[0] for x in v]):.1f}, introduced {np.mean([x[1] for x in v]):.1f}")
