"""Frame-decode readout for the variants behind the paper's gate, window and views tables (GraSP official test, ground-truth masks, three classifier seeds).

Every variant is a different way to build the per-instance class probabilities; the readout is the same one for all of them: the shipped argmax (for checking against the earlier tables) and the
frame decode with a prior shift (the final readout, scripts/frame_decode_eval.py). Variants:
    none                 single-pass ensemble
    revisit (tau)        the shipped gate S >= tau, confidence-weighted fusion of the +-10 s track
    refine all           every instrument revisited
    gate 45%             45% of the instruments revisited, ordered by: random (3 permutations, as scripts/gate_baselines_analysis.py), MC-dropout vote disagreement, softmax max-prob, evidential S
    window sym k / causal k   the 941 most uncertain instruments, fusion of the +-k s window / the last k frames (accuracy and macro-F1 only, as in the window table)
    tta                  the 941 most uncertain instruments, fusion of the 21 augmented keyframe views (tracked_tta runs of scripts/tta_control.py)

    python scripts/decode_variants_eval.py --dir experiments/gtbox_sam/final --shift -0.5 --out decode_variants.json
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
from sklearn.metrics import f1_score

from evidential_analyze import LABELS, WEIGHTS
from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from frame_decode_eval import decode, fused_distribution
from gtbox_sam_final_eval import load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--share", type=float, default=0.45)
    ap.add_argument("--top", type=int, default=941)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--causal-dir", default=None, help="separate causal tracking run (past 20 frames; <name>, <name>_s43, <name>_s44), as in scripts/window_sweep_from_tracks.py")
    ap.add_argument("--no-paint", action="store_true", help="accuracy and macro-F1 only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]

    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    train_labels = np.array([i[3] for i in GraspRegionDataset(args.data_root, "train", letterbox=True).instances])
    prior = np.bincount(train_labels, minlength=7) / len(train_labels)
    shift = -args.shift * np.log(prior)
    zero = np.zeros(7)
    masks = pickle.loads((args.dir / "masks.pkl").read_bytes())
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames = [(f, ix) for f, ix in by_frame.items() if all(i in masks for i in ix)]
    y_all = np.array([int(i[3]) for i in ds.instances])
    gts = {f: paint(tuple(ds.instances[ix[0]][1]["size"]), [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y_all[i]) + 1, 0.0) for i in ix]) for f, ix in frames}

    def read_offsets(dirs: list[str]) -> dict[int, list[int]]:
        found: dict[int, list[int]] = {}
        for d in dirs:
            for path in sorted((args.dir / d).glob("masks_shard*.pkl")):
                for idx, fr in pickle.loads(path.read_bytes())["masks"].items():
                    found[idx] = [off for off in sorted(fr) if off == 0 or mask_codec.area(fr[off][1]) > 0]
        return found

    offsets = read_offsets(TRACK_DIRS[42])
    causal_offsets = read_offsets([args.causal_dir]) if args.causal_dir else offsets
    rng = np.random.default_rng(0)
    results: dict = {}

    def record(name: str, seed: int, prob: np.ndarray, y: np.ndarray, done: np.ndarray, paint_it: bool, with_argmax_paint: bool = False) -> None:
        def score(pred: np.ndarray, do_paint: bool) -> dict:
            out = {"acc": 100 * float((pred[done] == y[done]).mean()), "F1": 100 * float(f1_score(y[done], pred[done], average="macro", labels=range(7)))}
            if do_paint:
                res = [frame_class_ious(paint(tuple(ds.instances[ix[0]][1]["size"]), [(mask_codec.decode(masks[i]).astype(bool), int(pred[i]) + 1, 1.0) for i in ix if done[i]]), gts[f]) for f, ix in frames]
                agg = aggregate(res)
                out.update({k: 100 * float(agg[k]) for k in ("mIoU", "IoU", "mcIoU")})
            return out
        rec = {"argmax": score(prob.argmax(1), paint_it and with_argmax_paint), "decode": score(decode(prob, by_frame, shift), paint_it)}
        results.setdefault(name, {})[str(seed)] = rec
        print(f"seed {seed} {name:34s} argmax acc {rec['argmax']['acc']:.2f} | decode " + " ".join(f"{k} {v:.2f}" for k, v in rec["decode"].items()), flush=True)

    for seed in args.seeds:
        z = np.load(args.dir / f"logits_s{seed}.npz")
        y, done = z["y"], z["done"]
        alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        s1 = variance_scores(alpha)["epistemic"]
        tracked: dict = {}
        for d in TRACK_DIRS[seed]:
            tracked.update(load_tracked(args.dir / d))
        usable = done & np.array([i in tracked for i in range(len(y))])
        pool = np.where(usable)[0]
        fused = {int(i): fused_distribution(tracked[int(i)], weights) for i in pool}

        def probs(flag: np.ndarray) -> np.ndarray:
            p = mu.copy()
            for i in np.where(flag & usable)[0]:
                p[i] = fused[int(i)]
            return p

        paint_it = not args.no_paint
        record("none", seed, mu, y, done, paint_it, True)
        record("revisit (tau)", seed, probs(s1 >= args.tau), y, done, paint_it, True)
        record("refine all", seed, probs(usable), y, done, paint_it)

        # gate orderings at the common share
        signals = {"evidential S": s1}
        ce_path = args.dir / f"ce_test_s{seed}.npz"
        if ce_path.exists():
            ce = np.load(ce_path)
            pos = {int(i): k for k, i in enumerate(ce["index"])}
            take = np.array([pos.get(i, -1) for i in range(len(y))])

            def full_arr(a: np.ndarray) -> np.ndarray:
                out = np.zeros(len(y))
                ok = take >= 0
                out[ok] = a[take[ok]]
                return out
            p_soft = sum(w * softmax(ce["det_" + m], axis=1) for w, m in zip(WEIGHTS, LABELS))
            votes = np.concatenate([softmax(ce["mc_" + m], axis=2).argmax(axis=2) for m in LABELS], axis=0)
            vshare = np.stack([(votes == c).sum(axis=0) for c in range(7)], axis=1).max(axis=1) / votes.shape[0]
            signals["softmax max-prob"] = full_arr(1 - p_soft.max(1))
            signals["MC vote disagreement"] = full_arr(1 - vshare)
        n_gate = int(round(args.share * len(pool)))
        for name, sig in signals.items():
            order = pool[np.argsort(-sig[pool], kind="stable")]
            flag = np.zeros(len(y), bool)
            flag[order[:n_gate]] = True
            record(f"gate {int(100 * args.share)}% {name}", seed, probs(flag), y, done, paint_it)
        for pi in range(3):
            perm = rng.permutation(pool)
            flag = np.zeros(len(y), bool)
            flag[perm[:n_gate]] = True
            record(f"gate {int(100 * args.share)}% random{pi}", seed, probs(flag), y, done, paint_it)

        # temporal window and causality on the most uncertain instruments (accuracy and macro-F1)
        top = [int(i) for i in np.argsort(-s1, kind="stable")[: args.top] if done[i]]
        causal_tracked = tracked
        if args.causal_dir:
            causal_tracked = load_tracked(args.dir / (args.causal_dir + ("" if seed == 42 else f"_s{seed}")))
        for mode in ("sym", "causal"):
            trk, offs = (causal_tracked, causal_offsets) if mode == "causal" else (tracked, offsets)
            for k in args.windows:
                p = mu.copy()
                for i in top:
                    if i not in trk or len(offs.get(i, [])) != trk[i].shape[0]:
                        continue
                    keep = [q for q, off in enumerate(offs[i]) if (abs(off) <= k if mode == "sym" else -k <= off <= 0)]
                    p[i] = fused_distribution(trk[i][keep], weights)
                record(f"window {mode} {k}", seed, p, y, done, False)
        record("window none", seed, mu, y, done, False)

        # augmented views of the keyframe instead of neighbouring frames
        tta_dir = args.dir / ("tracked_tta" + ("" if seed == 42 else f"_s{seed}"))
        if tta_dir.exists():
            tta = load_tracked(tta_dir)
            p = mu.copy()
            for i in top:
                if i in tta:
                    p[i] = fused_distribution(tta[i], weights)
            record("views: 21 augmented keyframe views", seed, p, y, done, paint_it)
            p = mu.copy()
            for i in top:
                if i in tracked:
                    p[i] = fused[i]
            record("views: propagated neighbours top941", seed, p, y, done, paint_it, True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"shift": args.shift, "tau": args.tau, "results": results}, indent=1))

    summary = {}
    for name, per_seed in results.items():
        summary[name] = {mode: {k: float(np.mean([r[mode][k] for r in per_seed.values() if k in r[mode]])) for k in per_seed[next(iter(per_seed))][mode]} for mode in ("argmax", "decode")}
    args.out.write_text(json.dumps({"shift": args.shift, "tau": args.tau, "results": results, "mean_over_seeds": summary}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
