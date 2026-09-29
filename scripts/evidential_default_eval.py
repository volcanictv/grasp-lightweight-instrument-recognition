"""Scores experiments 1 and 2 of docs/DECISIONS.md 2026-09-29 (evidential default).

Experiment 1: nested top-k sets by S1 (epistemic trace) of the seed-42 evidential official-split
ensemble, tracked predictions from the 2026-09-28 run plus the newly tracked instances, final
accuracy and macro-F1 at budgets 525, 649, 833 (and 539, the fold1-threshold set), next to the
vote pipeline at the same counts. Experiment 2: base accuracy, macro-F1 and S1 AUROC against the
ensemble's own errors for seeds 42, 43, 44, mean and SD.

Usage (any machine with the npz files):
    python scripts/evidential_default_eval.py --extract-dir experiments_edl/extract \
        --gate-dir experiments/evidential_gate_switch --default-dir experiments/evidential_default \
        --out docs/reports/evidential_default/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}
LABELS = list(WEIGHTS)
VOTE_PIPELINE = {525: (0.9581, 0.9295), 649: (0.9595, 0.9309), 833: (0.9623, 0.9351)}


def ensemble_alpha(d) -> np.ndarray:
    return sum(v * alpha_from_logits(d[f"det_{k}"]) for k, v in WEIGHTS.items())


def track_predict(det_frames: np.ndarray) -> int:
    w = np.array([WEIGHTS[l] for l in LABELS])
    alpha = sum(w[m] * alpha_from_logits(det_frames[:, m, :]) for m in range(len(LABELS)))
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    combined = (mu.max(axis=1)[:, None] * mu).sum(axis=0)
    return int(combined.argmax())


def exp1(args) -> dict:
    d = np.load(args.extract_dir / "E_grasp_official_s42.npz")
    y = d["y"]
    alpha = ensemble_alpha(d)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    base = mu.argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    order = np.argsort(-s1, kind="stable")

    tracked = {}
    files = [args.gate_dir / f"official_frames_shard{s}.npz" for s in (0, 1)]
    files += [args.default_dir / f"exp1_frames_shard{s}.npz" for s in (0, 1)]
    for p in files:
        fd = np.load(p)
        for key in fd.files:
            if key.startswith("det_"):
                tracked[int(key[4:])] = track_predict(fd[key])
    print(f"tracked instances available: {len(tracked)}")

    rows = []
    for k in (525, 539, 649, 833):
        top = order[:k]
        missing = [int(i) for i in top if int(i) not in tracked]
        assert not missing, f"top-{k} lacks tracking for {len(missing)} instances"
        pred = base.copy()
        for i in top:
            pred[i] = tracked[int(i)]
        flag = np.zeros(len(y), bool)
        flag[top] = True
        row = {"budget": k, "accuracy": float((pred == y).mean()),
               "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(7)))),
               "fixed": int(((base != y) & (pred == y) & flag).sum()),
               "broken": int(((base == y) & (pred != y) & flag).sum()),
               "errors_left": int((pred != y).sum()),
               "errors_inside_set": int((base[top] != y[top]).sum()),
               "share_of_base_errors_inside": float((base[top] != y[top]).sum() / (base != y).sum())}
        if k in VOTE_PIPELINE:
            row["vote_pipeline_accuracy"], row["vote_pipeline_macro_f1"] = VOTE_PIPELINE[k]
            row["delta_accuracy_vs_vote"] = row["accuracy"] - VOTE_PIPELINE[k][0]
        rows.append(row)
    return {"n": int(len(y)), "base_accuracy": float((base == y).mean()),
            "base_macro_f1": float(f1_score(y, base, average="macro", labels=list(range(7)))),
            "base_errors": int((base != y).sum()), "rows": rows}


def exp2(args) -> dict:
    rows = []
    for seed in (42, 43, 44):
        p = args.extract_dir / f"E_grasp_official_s{seed}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        y = d["y"]
        alpha = ensemble_alpha(d)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        pred = mu.argmax(axis=1)
        s1 = variance_scores(alpha)["epistemic"]
        err = pred != y
        rows.append({"seed": seed, "n": int(len(y)), "accuracy": float((~err).mean()),
                     "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(7)))),
                     "auroc_s1": float(roc_auc_score(err, s1)), "errors": int(err.sum()),
                     "flagged_at_tau_1.7e-5": int((s1 >= 1.7e-05).sum())})
    out = {"rows": rows}
    if len(rows) > 1:
        for key in ("accuracy", "macro_f1", "auroc_s1"):
            v = np.array([r[key] for r in rows])
            out[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "experiments_edl" / "extract")
    ap.add_argument("--gate-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_gate_switch")
    ap.add_argument("--default-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_default")
    ap.add_argument("--which", choices=["1", "2", "both"], default="both")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = {}
    if args.which in ("1", "both"):
        result["experiment_1"] = exp1(args)
        for r in result["experiment_1"]["rows"]:
            print(r)
    if args.which in ("2", "both"):
        result["experiment_2"] = exp2(args)
        print(json.dumps(result["experiment_2"], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
