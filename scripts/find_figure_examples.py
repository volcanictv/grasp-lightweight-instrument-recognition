"""Lists candidate instruments for the paper's Figure 1: the keyframe class (single pass, headline seed 44) is wrong, the instrument is in the gated set (top 833),
and the fused class over the tracked frames is right. Reads only the saved results of the final run; no GPU needed.

Usage (titanxp, surgical environment, repo root):
    python scripts/find_figure_examples.py --out experiments/figure_examples/candidates.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=REPO_ROOT / "experiments" / "gtbox_sam" / "final")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--budget", type=int, default=833)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    z = np.load(args.run / f"logits_s{args.seed}.npz")
    y, done = z["y"], z["done"]
    alpha = alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"])
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    base = mu.argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    gated = np.zeros(len(y), bool)
    gated[np.argsort(-s1, kind="stable")[: args.budget]] = True
    tdir = args.run / ("tracked" if args.seed == 42 else f"tracked_s{args.seed}")
    tracked = load_tracked(tdir)
    masks = {}
    for p in sorted((args.run / "tracked").glob("masks_shard*.pkl")):
        masks.update(pickle.loads(p.read_bytes())["masks"])

    rows = []
    for i in np.where(gated & done & (base != y))[0]:
        i = int(i)
        if i not in tracked or i not in masks:
            continue
        fused, share = fuse(tracked[i], CONFIGS["four"])
        if fused != y[i]:
            continue
        file_name, _seg, box, _label = ds.instances[i]
        x, yy, w, h = box
        rows.append({"index": i, "file": file_name, "box": [int(x), int(yy), int(w), int(h)], "true": CLASSES[y[i]], "single_frame": CLASSES[base[i]],
                     "single_frame_confidence": float(mu[i].max()), "s1": float(s1[i]), "fused": CLASSES[fused], "fused_share": share,
                     "track_frames": int(tracked[i].shape[0]), "offsets": sorted(masks[i].keys())[:1] + sorted(masks[i].keys())[-1:]})
    rows.sort(key=lambda r: (-r["s1"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} candidates (keyframe wrong, gated, fused right)")
    for r in rows[:25]:
        print(f"{r['index']:5d} {r['file']} box {r['box']} true {r['true']:<26} single {r['single_frame']:<26} conf {r['single_frame_confidence']:.2f} "
              f"S1 {r['s1']:.1e} frames {r['track_frames']} fused share {r['fused_share']:.2f}")
    pairs = Counter = {}
    for r in rows:
        k = (r["true"], r["single_frame"])
        pairs[k] = pairs.get(k, 0) + 1
    print("confusions fixed:", sorted(pairs.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    main()
