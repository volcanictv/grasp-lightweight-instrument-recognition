"""Post-hoc frame-level decoding of the ensemble + revisit probabilities with ground-truth masks.

Per instance probabilities are the single-pass ensemble mean, replaced by the confidence-weighted fused distribution of the
track for revisited instruments (same fusion as gtbox_sam_final_eval.fuse). Variants differ only in how labels are read out:
    argmax            the shipped readout
    frame             joint maximum log-probability per frame, every class except LND used at most once per frame (a structural
                      fact of the data: LND is the only class that repeats in 211 of 215 duplicate test frames, 414 of 421 train)
    frame + shift g   same, with log P + g * (-log prior) added per class; g < 0 favours frequent classes (rare-class precision)
Writes per-instance probabilities too, so later steps need no re-tracking.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, ORDER, alpha_mix
from gtbox_sam_final_eval import load_tracked
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

LND = 2


def fused_distribution(det_frames: np.ndarray, weights: dict) -> np.ndarray:
    alpha = sum(w * alpha_from_logits(det_frames[:, ORDER.index(k), :]) for k, w in weights.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    combined = (mu.max(axis=1)[:, None] * mu).sum(axis=0)
    return combined / combined.sum()


def decode(prob: np.ndarray, by_frame: dict, shift: np.ndarray) -> np.ndarray:
    logp = np.log(prob + 1e-9) + shift[None, :]
    out = logp.argmax(axis=1)
    for indices in by_frame.values():
        if len(indices) < 2:
            continue
        best, best_score = None, -np.inf
        for combo in itertools.product(range(prob.shape[1]), repeat=len(indices)):
            if any(n > 1 and c != LND for c, n in Counter(combo).items()):
                continue
            score = sum(logp[i, c] for i, c in zip(indices, combo))
            if score > best_score:
                best, best_score = combo, score
        for i, c in zip(indices, best):
            out[i] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--tracked-dir", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--shifts", type=float, nargs="*", default=[-0.5, -0.25, 0.25])
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    weights = CONFIGS["four"]
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    train_labels = np.array([i[3] for i in GraspRegionDataset(args.data_root, "train", letterbox=True).instances])
    prior = np.bincount(train_labels, minlength=7) / len(train_labels)
    masks = pickle.loads(args.masks.read_bytes())
    z = np.load(args.logits)
    y, done = z["y"], z["done"]
    alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    s1 = variance_scores(alpha)["epistemic"]
    tracked = load_tracked(args.tracked_dir)
    flag = (s1 >= args.tau) & done
    prob, revisited = mu.copy(), np.zeros(len(y), bool)
    for i in np.where(flag)[0]:
        if int(i) in tracked:
            prob[i], revisited[i] = fused_distribution(tracked[int(i)], weights), True

    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    sam = {idx: mask_codec.decode(rle).astype(bool) for idx, rle in masks.items()}
    gts = {}
    for file_name, indices in by_frame.items():
        shape = tuple(ds.instances[indices[0]][1]["size"])
        gts[file_name] = paint(shape, [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y[i]) + 1, 0.0) for i in indices])

    def score(pred: np.ndarray) -> dict:
        frames = []
        for file_name, indices in by_frame.items():
            if any(i not in masks for i in indices):
                continue
            shape = tuple(ds.instances[indices[0]][1]["size"])
            insts = [(sam[i], int(pred[i]) + 1, 1.0) for i in indices if done[i]]
            frames.append(frame_class_ious(paint(shape, insts), gts[file_name]))
        out = aggregate(frames)
        keep = done
        out["acc"] = float((pred[keep] == y[keep]).mean())
        out["macro_f1"] = float(f1_score(y[keep], pred[keep], average="macro", labels=range(7)))
        out["per_class_f1"] = [float(v) for v in f1_score(y[keep], pred[keep], average=None, labels=range(7))]
        out["fixed"] = int(((prob.argmax(1) != y) & (pred == y)).sum())
        out["broken"] = int(((prob.argmax(1) == y) & (pred != y)).sum())
        return out

    results = {"revisited": int(revisited.sum()), "instruments": int(len(y))}
    zero = np.zeros(7)
    results["argmax"] = score(prob.argmax(1))
    results["single pass argmax"] = score(mu.argmax(1))
    results["frame"] = score(decode(prob, by_frame, zero))
    for g in args.shifts:
        results[f"frame shift {g:+.2f}"] = score(decode(prob, by_frame, -g * np.log(prior)))
    pred_final = decode(prob, by_frame, zero)
    wrong = (prob.argmax(1) != y) & done
    results["errors_by_tracked"] = {"wrong_tracked": int((wrong & revisited).sum()), "wrong_untracked": int((wrong & ~revisited).sum())}
    fp = {}
    for c, name in enumerate(["BF", "PF", "LND", "MCS", "SI", "CA", "LG"]):
        m = (prob.argmax(1) == c) & (y != c) & done
        fp[name] = {"false_positives": int(m.sum()), "of_which_revisited": int((m & revisited).sum())}
    results["false_positives_after_revisit"] = fp
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    np.savez(args.out.with_suffix(".npz"), prob=prob, mu=mu, y=y, revisited=revisited, s1=s1, done=done)
    print(f"{'variant':<24}{'mIoU':>8}{'IoU':>8}{'mcIoU':>8}{'acc':>8}{'F1':>8}{'fixed':>7}{'broken':>7}")
    for name, r in results.items():
        if isinstance(r, dict) and "mIoU" in r:
            print(f"{name:<24}{100 * r['mIoU']:>8.2f}{100 * r['IoU']:>8.2f}{100 * r['mcIoU']:>8.2f}{100 * r['acc']:>8.2f}{100 * r['macro_f1']:>8.2f}{r['fixed']:>7}{r['broken']:>7}")
    print(json.dumps(results["false_positives_after_revisit"]))


if __name__ == "__main__":
    main()
