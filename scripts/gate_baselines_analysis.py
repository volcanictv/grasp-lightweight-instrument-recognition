"""Gate baselines for the refinement, on the refine-all tracks (every test instrument tracked): which ordering of the instruments should decide where the second look is spent?

Orderings compared (all refine the first share of instruments in the ordering and keep the single-pass label for the rest; the fusion, painting and metrics are those of scripts/refine_all_analysis.py and
scripts/gtbox_sam_final_eval.py):
  S                 the evidential epistemic score (the paper's gate)
  largest belief    one minus the largest belief of the same ensemble
  softmax           1 - max-softmax and the entropy of the softmax ensemble (cross-entropy members of scripts/make_ce_official_configs.py; needs scripts/baselines_on_test.py extract)
  MC dropout        vote disagreement and mutual information (80 passes, same members)
  class prior       instruments predicted as the weakest classes first (classes ordered by their single-pass IoU, then by S inside a class): no uncertainty at all
  small box         smallest given boxes first;   border box: boxes touching the image border first (then smaller first)
  random            mean of 3 permutations;       best possible: instruments whose label tracking fixes first, then unchanged, then the ones it breaks
Shares: 10, 20, 30, 45, 60, 100 percent. 3 seeds (the signals of a seed belong to that seed; geometry and class prior use that seed's single-pass labels).

    python scripts/gate_baselines_analysis.py --out docs/reports/gtbox_sam/gate_baselines.json
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
from scipy.special import softmax

from calibration_baselines import entropy
from evidential_analyze import LABELS, WEIGHTS
from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "experiments/gtbox_sam/final")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--shares", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.45, 0.60, 1.00])
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
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
    area = np.array([ds.instances[i][2][2] * ds.instances[i][2][3] for i in range(len(ds.instances))], dtype=float)
    border = np.zeros(len(ds.instances), bool)
    for i, (_f, seg, (x, y, w, h), _l) in enumerate(ds.instances):
        H, W = seg["size"]
        border[i] = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
    rng = np.random.default_rng(0)

    def score(pred: np.ndarray, conf: np.ndarray, done: np.ndarray) -> tuple[dict, dict]:
        res = []
        for f, ix in frames:
            shape = tuple(ds.instances[ix[0]][1]["size"])
            res.append(frame_class_ious(paint(shape, [(sam[i], int(pred[i]) + 1, float(conf[i])) for i in ix if done[i]]), gts[f]))
        out = aggregate(res)
        return {k: 100 * float(out[k]) for k in ("mIoU", "IoU", "mcIoU")}, out.get("per_class_iou", {})

    report: dict = {"shares": args.shares, "seeds": {}}
    for seed in args.seeds:
        z = np.load(args.dir / f"logits_s{seed}.npz")
        y, done = z["y"], z["done"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        base, base_conf = mu.argmax(axis=1), mu.max(axis=1)
        s1 = variance_scores(alpha)["epistemic"]
        tracked: dict = {}
        for d in TRACK_DIRS[seed]:
            tracked.update(load_tracked(args.dir / d))
        usable = done & np.array([i in tracked for i in range(len(y))])
        tpred, tconf = base.copy(), base_conf.copy()
        for i in np.where(usable)[0]:
            tpred[i], tconf[i] = fuse(tracked[int(i)], weights)
        pool = np.where(usable)[0]

        def labels(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            f = flag & usable
            return np.where(f, tpred, base), np.where(f, tconf, base_conf)

        single, per_class = score(base, base_conf, done)
        full, _ = score(*labels(usable), done)
        # class prior: instruments predicted as the weakest classes (single-pass IoU, ascending) first, S inside a class
        weak_order = sorted(per_class, key=lambda c: per_class[c]) if per_class else list(range(1, 8))
        class_rank = {int(c) - 1: r for r, c in enumerate(weak_order)}
        prior_key = np.array([class_rank.get(int(base[i]), 99) for i in range(len(y))], dtype=float)
        signals: dict[str, np.ndarray] = {"S": s1, "largest belief": 1 - mu.max(1), "belief entropy": entropy(mu),
                                          "class prior": -(prior_key * 1e6) + s1 / (s1.max() + 1e-12),
                                          "small box": -area, "border box": border * 1e12 - area}
        ce_path = args.dir / f"ce_test_s{seed}.npz"
        if ce_path.exists():
            ce = np.load(ce_path)
            pos = {int(i): k for k, i in enumerate(ce["index"])}
            take = np.array([pos.get(i, -1) for i in range(len(y))])
            def full_arr(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
                out = np.full(len(y), fill)
                ok = take >= 0
                out[ok] = a[take[ok]]
                return out
            logits = [ce["det_" + m] for m in LABELS]
            p_soft = sum(w * softmax(l, axis=1) for w, l in zip(WEIGHTS, logits))
            mc = [softmax(ce["mc_" + m], axis=2) for m in LABELS]
            p_mc = sum(w * x.mean(axis=0) for w, x in zip(WEIGHTS, mc))
            votes = np.concatenate([x.argmax(axis=2) for x in mc], axis=0)
            vshare = np.stack([(votes == c).sum(axis=0) for c in range(7)], axis=1).max(axis=1) / votes.shape[0]
            mean_h = sum(w * entropy(x).mean(axis=0) for w, x in zip(WEIGHTS, mc))
            signals.update({"softmax max-prob": full_arr(1 - p_soft.max(1)), "softmax entropy": full_arr(entropy(p_soft)), "MC vote disagreement": full_arr(1 - vshare),
                            "MC mutual information": full_arr(entropy(p_mc) - mean_h)})
        right_before, right_after = base == y, tpred == y
        best_key = np.where(~right_before & right_after, 0, np.where(right_before == right_after, 1, 2)) + 0.0
        orders = {name: pool[np.argsort(-sig[pool], kind="stable")] for name, sig in signals.items()}
        orders["best possible"] = pool[np.lexsort((-s1[pool], best_key[pool]))]
        curves: dict[str, dict] = {}
        for name, order in orders.items():
            curves[name] = {}
            for sh in args.shares:
                flag = np.zeros(len(y), bool)
                flag[order[: int(round(sh * len(order)))]] = True
                curves[name][f"{int(round(100 * sh))}%"] = score(*labels(flag), done)[0]
            print(f"seed {seed} {name:22s}" + " ".join(f"{int(round(100 * s))}%={curves[name][f'{int(round(100 * s))}%']['mIoU']:.2f}" for s in args.shares), flush=True)
        perms = [rng.permutation(pool) for _ in range(args.perms)]
        curves["random"] = {}
        for sh in args.shares:
            runs = []
            for p in perms:
                flag = np.zeros(len(y), bool)
                flag[p[: int(round(sh * len(p)))]] = True
                runs.append(score(*labels(flag), done)[0])
            curves["random"][f"{int(round(100 * sh))}%"] = {k: float(np.mean([r[k] for r in runs])) for k in ("mIoU", "IoU", "mcIoU")}
        report["seeds"][str(seed)] = {"single": single, "refine_all": full, "curves": curves, "ce_signals_present": ce_path.exists()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
