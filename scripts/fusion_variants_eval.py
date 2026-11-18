"""Instance-level comparison of per-track fusion rules over the saved per-frame evidential outputs (GT masks, tau gate).

Only the rule that turns a track's per-frame distributions into one distribution changes; the gate, the tracks, the members and the
readout (argmax, or frame decode with a prior shift) are those of frame_decode_eval.py. Also reports, for revisited instruments of
the lagging classes, how many frames read the true class, to show whether the evidence exists before fusion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, ORDER, alpha_mix
from frame_decode_eval import decode
from gtbox_sam_final_eval import load_tracked
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

NAMES = ["BF", "PF", "LND", "MCS", "SI", "CA", "LG"]


def frame_alphas(det_frames: np.ndarray, weights: dict) -> np.ndarray:
    return sum(w * alpha_from_logits(det_frames[:, ORDER.index(k), :]) for k, w in weights.items())


def norm(v: np.ndarray) -> np.ndarray:
    return v / v.sum()


def fusions(alpha: np.ndarray) -> dict[str, np.ndarray]:
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    conf = mu.max(axis=1)
    return {
        "conf-weighted mean (current)": norm((conf[:, None] * mu).sum(0)),
        "plain mean": norm(mu.mean(0)),
        "evidence sum": norm(1.0 + (alpha - 1.0).sum(0)),
        "log mean": norm(np.exp(np.log(mu + 1e-9).mean(0))),
        "max pool": norm(mu.max(0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--tracked-dir", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    weights = CONFIGS["four"]
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    train_labels = np.array([i[3] for i in GraspRegionDataset(args.data_root, "train", letterbox=True).instances])
    prior = np.bincount(train_labels, minlength=7) / len(train_labels)
    z = np.load(args.logits)
    y, done = z["y"], z["done"]
    alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    s1 = variance_scores(alpha)["epistemic"]
    tracked = load_tracked(args.tracked_dir)
    revisit = [int(i) for i in np.where((s1 >= args.tau) & done)[0] if int(i) in tracked]
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)

    per_track = {i: fusions(frame_alphas(tracked[i], weights)) for i in revisit}
    results: dict = {}
    for name in next(iter(per_track.values())):
        prob = mu.copy()
        for i in revisit:
            prob[i] = per_track[i][name]
        row = {}
        for label, pred in (("argmax", prob.argmax(1)), ("decode+shift", decode(prob, by_frame, -args.shift * np.log(prior)))):
            keep = done
            recall = [float(((pred == c) & (y == c))[keep].sum() / max(1, (y == c)[keep].sum())) for c in range(7)]
            row[label] = {"acc": float((pred[keep] == y[keep]).mean()), "macro_f1": float(f1_score(y[keep], pred[keep], average="macro", labels=range(7))),
                          "recall": dict(zip(NAMES, recall))}
        results[name] = row

    diag = {}
    for c in (2, 5, 6):
        wrong = [i for i in revisit if y[i] == c and per_track[i]["conf-weighted mean (current)"].argmax() != c]
        if not wrong:
            continue
        al = {i: frame_alphas(tracked[i], weights) for i in wrong}
        frac = [float((a.argmax(1) == c).mean()) for a in al.values()]
        pmax = [float((a / a.sum(1, keepdims=True))[:, c].max()) for a in al.values()]
        diag[NAMES[c]] = {"revisited_errors": len(wrong), "mean_frames": float(np.mean([len(a) for a in al.values()])),
                          "mean_share_of_frames_reading_true_class": float(np.mean(frac)),
                          "errors_with_any_frame_reading_true_class": int(sum(f > 0 for f in frac)),
                          "mean_max_frame_prob_of_true_class": float(np.mean(pmax))}
    results["diagnostic_revisited_errors"] = diag
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"{'fusion':<30}{'acc':>7}{'F1':>7}{'LG rec':>8}{'LND rec':>8}{'| dec+shift: acc':>16}{'F1':>7}{'LG rec':>8}{'CA rec':>8}")
    for name, r in results.items():
        if name == "diagnostic_revisited_errors":
            continue
        a, d = r["argmax"], r["decode+shift"]
        print(f"{name:<30}{100 * a['acc']:>7.2f}{100 * a['macro_f1']:>7.2f}{100 * a['recall']['LG']:>8.1f}{100 * a['recall']['LND']:>8.1f}"
              f"{100 * d['acc']:>16.2f}{100 * d['macro_f1']:>7.2f}{100 * d['recall']['LG']:>8.1f}{100 * d['recall']['CA']:>8.1f}")
    print(json.dumps(diag))


if __name__ == "__main__":
    main()
