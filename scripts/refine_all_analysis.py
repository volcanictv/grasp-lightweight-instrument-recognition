"""Analysis of the refine-all run (docs/reports/gtbox_sam_protocol.md, second addendum of 2026-10-07): every instrument of the official test split is tracked, so the effort curve
can be measured up to 100% and the gating signals compared on real refinement outcomes. CPU only; scoring is that of scripts/gtbox_sam_final_eval.py (same painting, fusion and metrics).

For each classifier seed (42, 43, 44; the tracks of the registered final run plus the tracks of the refine-all run) it reports
  - the three scores of the single pass, of refining every instrument, and of refining the registered gate's share (29%, top 833) and the held-out threshold's share,
  - the gain as a function of the share refined for the gates: evidential epistemic S1 (the registered gate), one minus the largest belief, the entropy of the belief, a random order
    (mean of several permutations) and the best possible order (instruments whose label tracking fixes first, then the ones it does not change, then the ones it breaks),
  - the share of the refine-all gain reached at 19%, 29% and 31% refined.
Results are the means of the three seeds where all three are run (the endpoints and the operating points); the curves over gates are run for every seed.

    python scripts/refine_all_analysis.py --out docs/reports/refine_all_analysis.json
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
from gtbox_sam_final_eval import fuse, load_tracked
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

FRACTIONS = [0.05, 0.10, 0.19, 0.29, 0.31, 0.42, 0.60, 0.80, 1.00]
TRACK_DIRS = {42: ["tracked", "tracked_rest"], 43: ["tracked_s43", "tracked_rest_s43"], 44: ["tracked_s44", "tracked_rest_s44"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "experiments/gtbox_sam/final")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--random-perms", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.7e-5, help="the held-out gate threshold of the shipped pipeline")
    ap.add_argument("--curve-seeds", type=int, nargs="*", default=[42])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    weights = CONFIGS["four"]
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    masks = pickle.loads((args.dir / "masks.pkl").read_bytes())
    sam = {i: mask_codec.decode(r).astype(bool) for i, r in masks.items()}
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames = [(f, ix) for f, ix in by_frame.items() if all(i in masks for i in ix)]
    gts = {f: paint(tuple(ds.instances[ix[0]][1]["size"]), [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(ds.instances[i][3]) + 1, 0.0) for i in ix]) for f, ix in frames}

    def score(pred: np.ndarray, conf: np.ndarray, done: np.ndarray) -> dict:
        res = []
        for f, ix in frames:
            shape = tuple(ds.instances[ix[0]][1]["size"])
            insts = [(sam[i], int(pred[i]) + 1, float(conf[i])) for i in ix if done[i]]
            res.append(frame_class_ious(paint(shape, insts), gts[f]))
        out = aggregate(res)
        return {k: 100 * float(out[k]) for k in ("mIoU", "IoU", "mcIoU")}

    report: dict = {"fractions": FRACTIONS, "seeds": {}}
    rng = np.random.default_rng(0)
    for seed, dirs in TRACK_DIRS.items():
        z = np.load(args.dir / f"logits_s{seed}.npz")
        y, done = z["y"], z["done"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        base, base_conf = mu.argmax(axis=1), mu.max(axis=1)
        s1 = variance_scores(alpha)["epistemic"]
        tracked: dict = {}
        for d in dirs:
            tracked.update(load_tracked(args.dir / d))
        have = np.array([i in tracked for i in range(len(y))])
        usable = done & have
        n_ok, n_done = int(usable.sum()), int(done.sum())
        fused = {i: fuse(tracked[i], weights) for i in np.where(usable)[0]}
        tpred, tconf = base.copy(), base_conf.copy()
        for i, (p, c) in fused.items():
            tpred[i], tconf[i] = p, c

        def labels(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            f = flag & usable
            return np.where(f, tpred, base), np.where(f, tconf, base_conf)

        info = {"instruments": int(len(y)), "classified": n_done, "with_tracks": n_ok,
                "accuracy_single": float((base[done] == y[done]).mean()), "accuracy_all_refined": float((labels(usable)[0][done] == y[done]).mean())}
        single = score(base, base_conf, done)
        full = score(*labels(usable), done)
        info["single"], info["refine_all"] = single, full
        rank = np.argsort(-s1, kind="stable")
        rank = np.array([i for i in rank if usable[i]])

        def at_share(order: np.ndarray, share: float) -> dict:
            flag = np.zeros(len(y), bool)
            flag[order[: int(round(share * len(order)))]] = True
            return score(*labels(flag), done)

        info["gate_points"] = {f"{int(100 * f)}%": at_share(rank, f) for f in (0.19, 0.29, 0.31, 0.42)}
        flag_tau = (s1 >= args.tau) & usable
        info["held_out_threshold"] = {"share": float(flag_tau.sum() / n_ok), **score(*labels(flag_tau), done)}
        report["seeds"][str(seed)] = info
        if seed not in args.curve_seeds:
            print(f"seed {seed}: single {single['mIoU']:.2f}, refine all {full['mIoU']:.2f}", flush=True)
            continue

        belief_gap = 1.0 - mu.max(axis=1)
        entropy = -(mu * np.log(np.clip(mu, 1e-12, None))).sum(axis=1)
        right_before, right_after = base == y, tpred == y
        key = np.where(~right_before & right_after, 0, np.where(right_before == right_after, 1, 2)) + (np.arange(len(y)) * 0.0)
        orders = {
            "evidential S1": rank,
            "1 - largest belief": np.array(sorted(np.where(usable)[0], key=lambda i: -belief_gap[i])),
            "entropy of belief": np.array(sorted(np.where(usable)[0], key=lambda i: -entropy[i])),
            "best possible": np.array(sorted(np.where(usable)[0], key=lambda i: (key[i], -s1[i]))),
        }
        curves: dict[str, dict] = {}
        for name, order in orders.items():
            curves[name] = {f"{int(100 * f)}%": at_share(order, f) for f in FRACTIONS}
            print(f"seed {seed} {name}: " + ", ".join(f"{int(100 * f)}%={curves[name][f'{int(100 * f)}%']['mIoU']:.2f}" for f in FRACTIONS), flush=True)
        perms = [rng.permutation(np.where(usable)[0]) for _ in range(args.random_perms)]
        rand = {}
        for f in FRACTIONS:
            runs = [at_share(p, f) for p in perms]
            rand[f"{int(100 * f)}%"] = {k: float(np.mean([r[k] for r in runs])) for k in ("mIoU", "IoU", "mcIoU")}
        curves["random"] = rand
        info["curves"] = curves
        # share of the refine-all gain reached by each gate at the registered shares
        gain = full["mIoU"] - single["mIoU"]
        info["share_of_full_gain"] = {name: {p: (c[p]["mIoU"] - single["mIoU"]) / gain for p in ("19%", "29%", "31%", "42%") if p in c} for name, c in curves.items()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
