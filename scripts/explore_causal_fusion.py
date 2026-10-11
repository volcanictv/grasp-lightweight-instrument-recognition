"""Exploratory, test-set only: can the causal pipeline's labels be improved without any new tracking, from the saved per-frame logits of the three classifier seeds?
Variants (all reported, none selected here): the registered fusion, the three seeds pooled into one 12-member evidential ensemble (alpha averaged over seeds, so the gate uses the pooled
score too), evidence summed over frames, plain mean of the frame mean-vectors, and the keyframe counted twice. Scoring is the registered one (unclipped masks).
Because it can only be scored on the test cases, whatever it finds is an exploratory variant, reported as such and never a replacement for the registered headline.

Usage (titanxp, surgical environment, repo root):
    python scripts/explore_causal_fusion.py --variant final --out docs/reports/gtbox_sam/explore_causal_fusion_final.json
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

from evidential_seeds_e2e_eval import CONFIGS, ORDER
from gtbox_sam_final_eval import load_tracked
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

W = CONFIGS["four"]


def alpha_of(frames: np.ndarray) -> np.ndarray:
    """(frames, members, classes) logits -> mixed Dirichlet parameters per frame."""
    return sum(W[k] * alpha_from_logits(frames[:, ORDER.index(k), :]) for k in ORDER)


def fuse_rule(alpha_frames: np.ndarray, rule: str) -> tuple[int, float]:
    """The label and its share of the combined mass (the share orders overlapping masks in the scoring, as in the registered evaluation)."""
    mu = alpha_frames / alpha_frames.sum(axis=1, keepdims=True)
    if rule == "registered":
        c = ((mu.max(axis=1)[:, None]) * mu).sum(axis=0)
    elif rule == "evidence_sum":
        c = alpha_frames.sum(axis=0)
    elif rule == "mean_mu":
        c = mu.mean(axis=0)
    elif rule == "keyframe_x2":  # the keyframe is the last frame of a causal track: count it twice
        w = np.ones(len(mu))
        w[-1] = 2.0
        c = ((w * mu.max(axis=1))[:, None] * mu).sum(axis=0)
    else:
        raise ValueError(rule)
    return int(c.argmax()), float(c.max() / c.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="final", help="experiments/gtbox_sam/<variant> holds masks.pkl, logits_s*.npz and tracked_causal_sam2[_s43|_s44]")
    ap.add_argument("--budget", type=int, default=833)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    D = REPO_ROOT / "experiments" / "gtbox_sam" / args.variant
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    masks = pickle.loads((D / "masks.pkl").read_bytes())
    seeds = (42, 43, 44)
    z = {s: np.load(D / f"logits_s{s}.npz") for s in seeds}
    y, done = z[42]["y"], z[42]["done"]
    tracked = {s: load_tracked(D / ("tracked_causal_sam2" if s == 42 else f"tracked_causal_sam2_s{s}")) for s in seeds}
    sam = {i: mask_codec.decode(r).astype(bool) for i, r in masks.items()}
    by_frame: dict[str, list[int]] = defaultdict(list)
    for i, inst in enumerate(ds.instances):
        by_frame[inst[0]].append(i)

    def single_alpha(s: int) -> np.ndarray:
        return sum(W[k] * alpha_from_logits(z[s][f"det_{k}"]) for k in ORDER)

    alpha_s = {s: single_alpha(s) for s in seeds}
    alpha_pool = sum(alpha_s.values()) / len(seeds)

    def score(pred: np.ndarray, conf: np.ndarray) -> dict:
        frames = []
        for fn, idx in by_frame.items():
            if any(i not in masks for i in idx):
                continue
            shape = tuple(ds.instances[idx[0]][1]["size"])
            gt = paint(shape, [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y[i]) + 1, 0.0) for i in idx])
            frames.append(frame_class_ious(paint(shape, [(sam[i], int(pred[i]) + 1, float(conf[i])) for i in idx if i in sam and done[i]]), gt))
        out = aggregate(frames)
        return {k: out[k] for k in ("mIoU", "IoU", "mcIoU")} | {"instance_accuracy": float((pred[done] == y[done]).mean())}

    def run(name: str, gate_alpha: np.ndarray, track_alpha, rule: str) -> dict:
        s1 = variance_scores(gate_alpha)["epistemic"]
        flag = np.zeros(len(y), bool)
        flag[np.argsort(-s1, kind="stable")[: args.budget]] = True
        mu = gate_alpha / gate_alpha.sum(axis=1, keepdims=True)
        pred, conf, missing = mu.argmax(axis=1).copy(), mu.max(axis=1).copy(), 0
        for i in np.where(flag & done)[0]:
            a = track_alpha(int(i))
            if a is None:
                missing += 1
            else:
                pred[i], conf[i] = fuse_rule(a, rule)
        r = score(pred, conf) | {"tracks_missing": missing}
        print(f"{name:34s} {100 * r['mIoU']:6.2f} {100 * r['IoU']:6.2f} {100 * r['mcIoU']:6.2f}  acc {r['instance_accuracy']:.4f}  missing {missing}", flush=True)
        return r

    def per_seed(s: int):
        return lambda i: alpha_of(tracked[s][i]) if i in tracked[s] else None

    def pooled(i: int):
        if not all(i in tracked[s] for s in seeds):
            return None
        shapes = {tracked[s][i].shape for s in seeds}
        if len(shapes) != 1:
            return None
        return sum(alpha_of(tracked[s][i]) for s in seeds) / len(seeds)

    results: dict = {"variant": args.variant, "budget": args.budget, "note": "exploratory, scored on the test cases only; the registered headline is unchanged"}
    for rule in ("registered", "evidence_sum", "mean_mu", "keyframe_x2"):
        rs = [run(f"seed {s}, {rule}", alpha_s[s], per_seed(s), rule) for s in seeds]
        results[f"3-seed mean, {rule}"] = {k: float(np.mean([r[k] for r in rs])) for k in ("mIoU", "IoU", "mcIoU", "instance_accuracy")} | {"seeds": rs}
        results[f"pooled 12 members, {rule}"] = run(f"pooled 12 members, {rule}", alpha_pool, pooled, rule)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print("\nrule                        3-seed mean (mIoU IoU mcIoU)      pooled 12 members")
    for rule in ("registered", "evidence_sum", "mean_mu", "keyframe_x2"):
        a, b = results[f"3-seed mean, {rule}"], results[f"pooled 12 members, {rule}"]
        print(f"{rule:20s} {100 * a['mIoU']:6.2f} {100 * a['IoU']:6.2f} {100 * a['mcIoU']:6.2f}        {100 * b['mIoU']:6.2f} {100 * b['IoU']:6.2f} {100 * b['mcIoU']:6.2f}")
    print("TAPIS                 86.61  83.38  77.42")


if __name__ == "__main__":
    main()
