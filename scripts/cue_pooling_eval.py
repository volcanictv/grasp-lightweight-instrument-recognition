"""Class-specific track pooling: for classes recognized by one localized cue that is visible in only some frames (LG: the fenestrated jaw; CA: the clip jaws), the fused probability of that class is the mean of
its top-k per-frame probabilities along the track instead of the confidence-weighted mean; all other classes keep the shipped fusion. Everything else (gate tau, tracks, members, frame decode + prior shift)
is the shipped pipeline. Settings are compared on the five cross-validation folds (ground-truth masks, tracks of scripts/campaign_tasks_gtmask.py) and then once on the official test split (3 seeds).

Rule fixed in advance: a setting is only adopted if accuracy and macro-F1 both improve on the mean over folds and in at least 4 of the 5 folds; the test split is then reported, not used for the choice.

    python scripts/cue_pooling_eval.py --cv-root experiments/gtcv5 --test-dir experiments/gtbox_sam/gtmask --out cue_pooling.json
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
from refine_all_analysis import TRACK_DIRS
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

LG, CA = 6, 5
SETTINGS = ["base"] + [f"{'+'.join(n)}:k{k}" for n in (("LG",), ("CA",), ("LG", "CA")) for k in (1, 2, 3)]


def pooled(det: np.ndarray, weights: dict, cue_classes: list[int], k: int) -> np.ndarray:
    alpha = sum(w * alpha_from_logits(det[:, ORDER.index(m), :]) for m, w in weights.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    combined = (mu.max(axis=1)[:, None] * mu).sum(axis=0)
    combined = combined / combined.sum()
    for c in cue_classes:
        top = np.sort(mu[:, c])[::-1][: min(k, len(mu))]
        combined[c] = top.mean()
    return combined / combined.sum()


def parse(setting: str) -> tuple[list[int], int]:
    if setting == "base":
        return [], 1
    names, k = setting.split(":k")
    return [{"LG": LG, "CA": CA}[n] for n in names.split("+")], int(k)


def evaluate(ds, z, tracked, weights, tau, shift) -> dict:
    y, done = z["y"], z["done"]
    alpha = alpha_mix(lambda m: z[f"det_{m}"], weights)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    s1 = variance_scores(alpha)["epistemic"]
    flag = np.where((s1 >= tau) & done)[0]
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    out = {}
    for setting in SETTINGS:
        cues, k = parse(setting)
        prob = mu.copy()
        for i in flag:
            if int(i) in tracked:
                prob[i] = pooled(tracked[int(i)], weights, cues, k)
        pred = decode(prob, by_frame, shift)
        m = done
        rec = {"acc": 100 * float((pred[m] == y[m]).mean()), "F1": 100 * float(f1_score(y[m], pred[m], average="macro", labels=range(7)))}
        for c, n in ((LG, "LG"), (CA, "CA")):
            tp = int(((pred == c) & (y == c) & m).sum())
            rec[f"{n}_recall"] = 100 * tp / max(1, int(((y == c) & m).sum()))
            rec[f"{n}_precision"] = 100 * tp / max(1, int(((pred == c) & m).sum()))
        out[setting] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-root", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]
    cv: dict = {}
    for k in range(5):
        fo = args.cv_root / f"cv5_f{k}" / "final"
        ds = GraspRegionDataset(args.data_root, f"cv5_f{k}_test", letterbox=True)
        train = np.array([i[3] for i in GraspRegionDataset(args.data_root, f"cv5_f{k}_train", letterbox=True).instances])
        shift = -args.shift * np.log(np.bincount(train, minlength=7) / len(train))
        cv[k] = evaluate(ds, np.load(fo / "logits.npz"), load_tracked(fo / "tracked"), weights, args.tau, shift)
        print(f"fold {k} done", flush=True)
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    train = np.array([i[3] for i in GraspRegionDataset(args.data_root, "train", letterbox=True).instances])
    shift = -args.shift * np.log(np.bincount(train, minlength=7) / len(train))
    test: dict = {}
    for seed in (42, 43, 44):
        tracked: dict = {}
        for d in TRACK_DIRS[seed]:
            tracked.update(load_tracked(args.test_dir / d))
        test[seed] = evaluate(ds, np.load(args.test_dir / f"logits_s{seed}.npz"), tracked, weights, args.tau, shift)
        print(f"test seed {seed} done", flush=True)
    cols = ["acc", "F1", "LG_recall", "LG_precision", "CA_recall", "CA_precision"]
    print("\nCV folds (mean of 5)               " + " ".join(f"{c:>12}" for c in cols) + "   folds better in acc&F1")
    report = {"cv": {}, "test": {}}
    for s in SETTINGS:
        mean = {c: float(np.mean([cv[k][s][c] for k in cv])) for c in cols}
        better = sum(cv[k][s]["acc"] > cv[k]["base"]["acc"] and cv[k][s]["F1"] > cv[k]["base"]["F1"] for k in cv)
        report["cv"][s] = {**mean, "folds_better": int(better)}
        print(f"{s:34s} " + " ".join(f"{mean[c]:12.2f}" for c in cols) + f"   {better}/5")
    print("\ntest (mean of 3 seeds)")
    for s in SETTINGS:
        mean = {c: float(np.mean([test[sd][s][c] for sd in test])) for c in cols}
        report["test"][s] = mean
        print(f"{s:34s} " + " ".join(f"{mean[c]:12.2f}" for c in cols))
    args.out.write_text(json.dumps({"per_fold": {str(k): v for k, v in cv.items()}, "per_seed": {str(k): v for k, v in test.items()}, **report}, indent=1))


if __name__ == "__main__":
    main()
