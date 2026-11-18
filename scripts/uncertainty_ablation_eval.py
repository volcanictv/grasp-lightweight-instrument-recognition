"""Where does the gain come from: the revisit, the uncertainty that triggers it, or the readout? (GraSP official test, ground-truth masks, three classifier seeds.)

Variants (per seed), each read out with plain argmax and with the frame decode + prior shift:
    no revisit                 single-pass ensemble
    revisit, uncertainty gate  the shipped gate S >= tau (about 38% of the instruments)
    revisit, random gate       the same number of instruments, chosen at random (5 permutations)
    revisit all                every instrument
Reports accuracy, macro-F1, mIoU, IoU, mcIoU (means over seeds; random: mean over permutations and seeds), and for the main contrasts the difference with a 95% frame bootstrap interval
(the same resampled frames for every variant; differences averaged over seeds) and the difference per held-out case.

    python scripts/uncertainty_ablation_eval.py --dir experiments/gtbox_sam/gtmask --out uncertainty_ablation.json
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
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from frame_decode_eval import decode, fused_distribution
from gtbox_sam_final_eval import load_tracked
from refine_all_analysis import TRACK_DIRS
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

KEYS = ("mIoU", "IoU", "mcIoU")
PERMS = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    train_labels = np.array([i[3] for i in GraspRegionDataset(args.data_root, "train", letterbox=True).instances])
    shift = -args.shift * np.log(np.bincount(train_labels, minlength=7) / len(train_labels))
    zero = np.zeros(7)
    masks = pickle.loads((args.dir / "masks.pkl").read_bytes())
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames = [(f, ix) for f, ix in by_frame.items() if all(i in masks for i in ix)]
    cases = np.array([f.split("/")[0] for f, _ in frames])
    y_all = np.array([int(i[3]) for i in ds.instances])
    gts = {f: paint(tuple(ds.instances[ix[0]][1]["size"]), [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y_all[i]) + 1, 0.0) for i in ix]) for f, ix in frames}
    rng = np.random.default_rng(0)

    # per[(seed, variant, readout)] = {"res": per-frame stats, "acc", "F1"}
    per: dict = {}

    def score(prob: np.ndarray, y: np.ndarray, done: np.ndarray, readout: str) -> dict:
        pred = prob.argmax(1) if readout == "argmax" else decode(prob, by_frame, shift)
        res = [frame_class_ious(paint(tuple(ds.instances[ix[0]][1]["size"]), [(mask_codec.decode(masks[i]).astype(bool), int(pred[i]) + 1, 1.0) for i in ix if done[i]]), gts[f]) for f, ix in frames]
        return {"res": res, "acc": 100 * float((pred[done] == y[done]).mean()), "F1": 100 * float(f1_score(y[done], pred[done], average="macro", labels=range(7)))}

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

        gate = (s1 >= args.tau) & usable
        n_gate = int(gate.sum())
        variants = {"none": probs(np.zeros(len(y), bool)), "uncertainty": probs(gate), "all": probs(usable)}
        for pi in range(PERMS):
            flag = np.zeros(len(y), bool)
            flag[rng.permutation(pool)[:n_gate]] = True
            variants[f"random{pi}"] = probs(flag)
        for name, p in variants.items():
            for readout in ("argmax", "decode"):
                per[(seed, name, readout)] = score(p, y, done, readout)
        print(f"seed {seed}: revisited {n_gate} of {len(pool)} ({100 * n_gate / len(pool):.1f}%)", flush=True)

    def metrics(res: list) -> dict:
        agg = aggregate(res)
        return {k: 100 * float(agg[k]) for k in KEYS}

    def variant_names(group: str) -> list[str]:
        return [f"random{i}" for i in range(PERMS)] if group == "random" else [group]

    table: dict = {}
    for group in ("none", "uncertainty", "random", "all"):
        for readout in ("argmax", "decode"):
            rows = [{**metrics(per[(s, v, readout)]["res"]), "acc": per[(s, v, readout)]["acc"], "F1": per[(s, v, readout)]["F1"]} for s in args.seeds for v in variant_names(group)]
            table[f"{group}|{readout}"] = {k: float(np.mean([r[k] for r in rows])) for k in (*KEYS, "acc", "F1")}
    print("\nvariant | readout        mIoU    IoU  mcIoU    acc     F1")
    for name, r in table.items():
        print(f"{name:22s} {r['mIoU']:6.2f} {r['IoU']:6.2f} {r['mcIoU']:6.2f} {r['acc']:6.2f} {r['F1']:6.2f}")

    # paired frame bootstrap of the contrasts, differences averaged over seeds (and over random permutations)
    contrasts = {
        "revisit(uncertainty) - none, argmax": (("uncertainty", "argmax"), ("none", "argmax")),
        "revisit(uncertainty) - none, decode": (("uncertainty", "decode"), ("none", "decode")),
        "decode - argmax, no revisit": (("none", "decode"), ("none", "argmax")),
        "decode - argmax, with revisit": (("uncertainty", "decode"), ("uncertainty", "argmax")),
        "uncertainty gate - random gate (same count), argmax": (("uncertainty", "argmax"), ("random", "argmax")),
        "uncertainty gate - random gate (same count), decode": (("uncertainty", "decode"), ("random", "decode")),
        "uncertainty gate - revisit all, decode": (("uncertainty", "decode"), ("all", "decode")),
    }
    n = len(frames)
    boots = [rng.integers(0, n, n) for _ in range(args.boots)]

    def delta(a: tuple, b: tuple, sample: np.ndarray | None) -> dict:
        out = {k: [] for k in KEYS}
        for s in args.seeds:
            for va in variant_names(a[0]):
                for vb in variant_names(b[0]):
                    if a[0] == "random" and b[0] == "random":
                        continue
                    if "random" in (a[0], b[0]) and va.startswith("random") and vb.startswith("random"):
                        continue
                    ra, rb = per[(s, va, a[1])]["res"], per[(s, vb, b[1])]["res"]
                    if sample is not None:
                        ra, rb = [ra[i] for i in sample], [rb[i] for i in sample]
                    ma, mb = metrics(ra), metrics(rb)
                    for k in KEYS:
                        out[k].append(ma[k] - mb[k])
        return {k: float(np.mean(v)) for k, v in out.items()}

    report: dict = {"table": table, "contrasts": {}, "per_case": {}}
    print("\ncontrast (mean difference over seeds; 95% frame-bootstrap interval)")
    for name, (a, b) in contrasts.items():
        point = delta(a, b, None)
        draws = [delta(a, b, s) for s in boots[: args.boots]]
        report["contrasts"][name] = {k: {"diff": point[k], "lo": float(np.percentile([d[k] for d in draws], 2.5)), "hi": float(np.percentile([d[k] for d in draws], 97.5))} for k in KEYS}
        print(f"{name:55s} " + "  ".join(f"{k} {point[k]:+.2f} [{report['contrasts'][name][k]['lo']:+.2f}, {report['contrasts'][name][k]['hi']:+.2f}]" for k in KEYS), flush=True)
    # per held-out case: difference of the three metrics for the two headline contrasts
    for name in ("revisit(uncertainty) - none, decode", "uncertainty gate - random gate (same count), decode"):
        a, b = contrasts[name]
        report["per_case"][name] = {}
        for c in sorted(set(cases)):
            sel = np.where(cases == c)[0]
            d = delta(a, b, sel)
            report["per_case"][name][c] = {**d, "frames": int(len(sel))}
        print(f"\nper case: {name}")
        for c, d in report["per_case"][name].items():
            print(f"  {c} ({d['frames']} frames): " + "  ".join(f"{k} {d[k]:+.2f}" for k in KEYS))
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
