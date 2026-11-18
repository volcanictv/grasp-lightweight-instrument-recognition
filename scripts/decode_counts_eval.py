"""Counts behind the inline numbers of the recognition paper, with the argmax readout and the frame-decode readout side by side (GraSP official test, ground-truth masks, three seeds).

For each readout: errors fixed and introduced by the revisit (tau gate), by the 45% evidential gate, by the symmetric and the causal +-10 window (941 most uncertain instruments), the number of instruments
still wrong after the revisit, and the per-class recall before and after the revisit. 'Fixed' = wrong without the revisit and right with it, 'introduced' = the reverse; means over seeds.

    python scripts/decode_counts_eval.py --dir experiments/gtbox_sam/gtmask --causal-dir tracked_causal --shift -0.5 --out decode_counts.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from frame_decode_eval import decode, fused_distribution
from gtbox_sam_final_eval import load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores

NAMES = ["BF", "PF", "LND", "MCS", "SI", "CA", "LG"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--causal-dir", default=None)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--share", type=float, default=0.45)
    ap.add_argument("--top", type=int, default=941)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    train_labels = np.array([i[3] for i in GraspRegionDataset(args.data_root, "train", letterbox=True).instances])
    prior = np.bincount(train_labels, minlength=7) / len(train_labels)
    shift = -args.shift * np.log(prior)
    zero = np.zeros(7)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)

    def read_offsets(dirs: list[str]) -> dict[int, list[int]]:
        found: dict[int, list[int]] = {}
        for d in dirs:
            for path in sorted((args.dir / d).glob("masks_shard*.pkl")):
                for idx, fr in pickle.loads(path.read_bytes())["masks"].items():
                    found[idx] = [off for off in sorted(fr) if off == 0 or mask_codec.area(fr[off][1]) > 0]
        return found

    offsets = read_offsets(TRACK_DIRS[42])
    causal_offsets = read_offsets([args.causal_dir]) if args.causal_dir else offsets
    res: dict = {"argmax": defaultdict(list), "decode": defaultdict(list)}
    recall: dict = {"argmax": {"before": [], "after": []}, "decode": {"before": [], "after": []}}
    for seed in args.seeds:
        z = np.load(args.dir / f"logits_s{seed}.npz")
        y, done = z["y"], z["done"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        s1 = variance_scores(alpha)["epistemic"]
        tracked: dict = {}
        for d in TRACK_DIRS[seed]:
            tracked.update(load_tracked(args.dir / d))
        causal_tracked = load_tracked(args.dir / (args.causal_dir + ("" if seed == 42 else f"_s{seed}"))) if args.causal_dir else tracked
        usable = done & np.array([i in tracked for i in range(len(y))])
        pool = np.where(usable)[0]
        fused = {int(i): fused_distribution(tracked[int(i)], weights) for i in pool}

        def with_flag(flag: np.ndarray) -> np.ndarray:
            p = mu.copy()
            for i in np.where(flag & usable)[0]:
                p[i] = fused[int(i)]
            return p

        def with_window(mode: str, k: int = 10) -> np.ndarray:
            trk, offs = (causal_tracked, causal_offsets) if mode == "causal" else (tracked, offsets)
            p = mu.copy()
            for i in [int(i) for i in np.argsort(-s1, kind="stable")[: args.top] if done[i]]:
                if i not in trk or len(offs.get(i, [])) != trk[i].shape[0]:
                    continue
                keep = [q for q, off in enumerate(offs[i]) if (abs(off) <= k if mode == "sym" else -k <= off <= 0)]
                p[i] = fused_distribution(trk[i][keep], weights)
            return p

        order = pool[np.argsort(-s1[pool], kind="stable")]
        g45 = np.zeros(len(y), bool)
        g45[order[: int(round(args.share * len(pool)))]] = True
        variants = {"tau": with_flag(s1 >= args.tau), "gate45": with_flag(g45), "sym10": with_window("sym"), "causal10": with_window("causal")}
        for mode, sh in (("argmax", zero), ("decode", shift)):
            read = (lambda p: p.argmax(1)) if mode == "argmax" else (lambda p: decode(p, by_frame, sh))
            before = read(mu)
            for name, p in variants.items():
                after = read(p)
                res[mode][f"{name}_fixed"].append(int(((before != y) & (after == y) & done).sum()))
                res[mode][f"{name}_introduced"].append(int(((before == y) & (after != y) & done).sum()))
                res[mode][f"{name}_still_wrong"].append(int(((after != y) & done).sum()))
                if name == "tau":
                    recall[mode]["before"].append([float((before[(y == c) & done] == c).mean()) for c in range(7)])
                    recall[mode]["after"].append([float((after[(y == c) & done] == c).mean()) for c in range(7)])
        print("seed", seed, "done", flush=True)
    out = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()}
    for m in recall:
        out[m]["recall_before"] = {n: 100 * float(np.mean([r[c] for r in recall[m]["before"]])) for c, n in enumerate(NAMES)}
        out[m]["recall_after"] = {n: 100 * float(np.mean([r[c] for r in recall[m]["after"]])) for c, n in enumerate(NAMES)}
    out["per_seed"] = {m: {k: v for k, v in d.items()} for m, d in res.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for m in ("argmax", "decode"):
        print(m, json.dumps({k: round(v, 1) for k, v in out[m].items() if not isinstance(v, dict)}))
        print(m, "recall before", {n: round(v, 1) for n, v in out[m]["recall_before"].items()})
        print(m, "recall after ", {n: round(v, 1) for n, v in out[m]["recall_after"].items()})


if __name__ == "__main__":
    main()
