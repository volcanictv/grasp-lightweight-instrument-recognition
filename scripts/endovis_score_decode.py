"""EndoVis 2018 (ISINet split) validation with annotation masks: gate, fused revisit, frame decode, and the three semantic IoUs.

Inputs are the outputs of the EndoVis pipeline for one classifier seed: the single-pass member logits of every validation instrument (`single_ens_s<seed>.npz`)
and the member logits of the SAM2 tracks of the flagged instruments (`tracked_s<seed>.npz`). The instruments with a track take the confidence-weighted fused
belief of their track, the others the single-pass mean belief. The frame decode then labels the instruments of each frame jointly; its duplicate penalties come
from a statistics file (`decode_stats/endovis18_train_plus_2017.json` of the handoff repository, or any file of that format), i.e. from the training frames.

Usage:
    python scripts/endovis_score_decode.py --instances ~/endovis_isinet/instances.pkl --single single_ens_s42.npz --tracked tracked_s42.npz \\
        --stats decode_stats/endovis18_train_plus_2017.json --out score_s42.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec

from evidential_seeds_e2e_eval import CONFIGS, ORDER, alpha_mix
from surgical_ai.evaluation.evidential import alpha_from_logits
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

NAMES = ["BF", "PF", "LND", "MCS", "UP", "SI", "CA"]  # class order of the EndoVis members


def fused_beliefs(single: np.lib.npyio.NpzFile, tracked: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    weights = CONFIGS["four"]
    alpha = alpha_mix(lambda k: single[f"det_{k}"], weights)
    mu = alpha / alpha.sum(1, keepdims=True)
    prob = mu.copy()
    for key in tracked.files:
        if key.startswith("det_"):
            a = sum(w * alpha_from_logits(tracked[key][:, ORDER.index(m), :]) for m, w in weights.items())
            m = a / a.sum(1, keepdims=True)
            combined = (m.max(1)[:, None] * m).sum(0)
            prob[int(key[4:])] = combined / combined.sum()
    return mu, prob


def decode_frames(prob: np.ndarray, frames: dict, penalty: np.ndarray) -> np.ndarray:
    logp = np.log(prob + 1e-9)
    out = logp.argmax(1)
    for idx in frames.values():
        if len(idx) < 2:
            continue
        best, best_score = None, -np.inf
        for combo in itertools.product(range(prob.shape[1]), repeat=len(idx)):
            score = sum(logp[i, c] for i, c in zip(idx, combo))
            score += sum((n - 1) * penalty[c] for c, n in collections.Counter(combo).items() if n > 1)
            if score > best_score:
                best, best_score = combo, score
        out[idx] = best
    return out


def score(pred: np.ndarray, y: np.ndarray, val: list[dict], frames: dict) -> dict:
    per_frame = []
    for idx in frames.values():
        shape = tuple(val[idx[0]]["rle"]["size"])
        gt = paint(shape, [(mask_codec.decode(val[i]["rle"]).astype(bool), int(y[i]) + 1, 0.0) for i in idx])
        pr = paint(shape, [(mask_codec.decode(val[i]["rle"]).astype(bool), int(pred[i]) + 1, 1.0) for i in idx])
        per_frame.append(frame_class_ious(pr, gt))
    agg = aggregate(per_frame)
    return {"mIoU": 100 * agg["mIoU"], "IoU": 100 * agg["IoU"], "mcIoU": 100 * agg["mcIoU"], "accuracy": 100 * float((pred == y).mean()),
            "per_class_iou": {NAMES[c]: 100 * agg["per_class_iou"].get(c + 1, float("nan")) for c in range(7)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--single", type=Path, required=True)
    ap.add_argument("--tracked", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True, help="decode statistics (soft rule): frames_with_class and frames_with_duplicate per class")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    val = pickle.loads(args.instances.read_bytes())["val"]
    single, tracked = np.load(args.single), np.load(args.tracked)
    y = single["y"]
    assert (y == np.array([r["cls"] for r in val])).all(), "instances.pkl and the logits are not the same validation set"
    stats = json.loads(args.stats.read_text())
    assert stats["rule"] == "soft" and stats["class_names"][4] == "Ultrasound Probe"
    penalty = np.log((np.asarray(stats["frames_with_duplicate"], float) + 1) / (np.asarray(stats["frames_with_class"], float) + 2))
    frames: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for i, r in enumerate(val):
        frames[(r["seq"], r["frame"])].append(i)
    mu, prob = fused_beliefs(single, tracked)
    result = {"tracked": sum(k.startswith("det_") for k in tracked.files)}
    for name, pred in (("single pass", mu.argmax(1)), ("revisit, argmax", prob.argmax(1)), ("revisit + frame decode", decode_frames(prob, frames, penalty))):
        result[name] = score(pred, y, val, frames)
        r = result[name]
        print(f"{name:<24} mIoU {r['mIoU']:.2f}  IoU {r['IoU']:.2f}  mcIoU {r['mcIoU']:.2f}  accuracy {r['accuracy']:.2f}")
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
