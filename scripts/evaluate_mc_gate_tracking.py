"""Scores the end-to-end tracking comparison: confidence gate vs MC Dropout gate
on the deep-dropout ensemble (docs/DECISIONS.md 2026-09-19; gates fixed by
scripts/compute_tracking_gates.py before any tracking outcome existed).

For each gate the final prediction is: the SAM2-tracked avg-softmax prediction
for flagged instances, the ensemble's own single-frame prediction for the rest
-- the same construction as the existing 0.9343 -> 0.9567 result. Because
tracking an instance does not depend on which gate picked it, one tracked run
over the union of both gates' flagged instances scores every gate.

Reports accuracy, macro-F1, per-class F1, fix/regression counts, and a paired
comparison of the two gates (exact McNemar test on the instances where their
final predictions differ in correctness, plus a paired instance bootstrap).

Usage:
    python scripts/evaluate_mc_gate_tracking.py [--allow-partial]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_mcgate")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_mcgate" / "results.json")
    parser.add_argument("--allow-partial", action="store_true", help="dry run on incomplete tracking output")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y_true = cache["y_true"]
    p_det = sum(w * torch.softmax(torch.from_numpy(cache[f"det_{m['label']}"]), dim=1).numpy() for w, m in zip(weights, members))
    y_pred = p_det.argmax(axis=1)

    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    ds = GraspRegionDataset(data_root, "test", transform=build_transforms(224, train=False))
    names = ds.class_names_ordered()
    case_of = np.array([fn.split("/")[0] for fn, _s, _b, _l in ds.instances])
    name_to_idx = {n: i for i, n in enumerate(names)}

    gates = json.loads((args.dir / "gates.json").read_text())
    tracked: dict[int, dict] = {}
    for shard in (0, 1):
        path = args.dir / f"tracked_shard{shard}.json"
        if path.exists():
            for r in json.loads(path.read_text())["instances"]:
                tracked[r["index"]] = r
    union = set(gates["conf_flagged_indices"]) | set(gates["mc_flagged_indices"])
    missing = union - set(tracked)
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} flagged instances have no tracked result (e.g. {sorted(missing)[:5]}); use --allow-partial for a dry run")
    if missing:
        print(f"WARNING dry run: {len(missing)} of {len(union)} flagged instances not tracked yet; they keep the baseline prediction\n")

    mismatch = [i for i, r in tracked.items() if name_to_idx[r["single_frame_pred"]] != y_pred[i]]
    print(f"tracked instances: {len(tracked)}; single-frame prediction differs from cached ensemble on {len(mismatch)}")

    def final_pred(flag_idx: list[int]) -> np.ndarray:
        pred = y_pred.copy()
        for i in flag_idx:
            if i in tracked:
                pred[i] = name_to_idx[tracked[i]["avg_softmax_pred"]]
        return pred

    conf_idx, mc_idx = gates["conf_flagged_indices"], gates["mc_flagged_indices"]
    both = sorted(set(conf_idx) & set(mc_idx))
    configs = {
        "ensemble alone": [],
        f"+ confidence gate (<{gates['confidence_gate']['threshold']:.2f}, n={len(conf_idx)})": conf_idx,
        f"+ MC Dropout gate (>={gates['mc_dropout_gate']['threshold']:.2f}, n={len(mc_idx)})": mc_idx,
        f"+ either gate (n={len(union)})": sorted(union),
        f"+ both gates agree (n={len(both)})": both,
    }
    preds = {k: final_pred(v) for k, v in configs.items()}
    labels = list(range(len(names)))
    results = {"n_total": int(len(y_true)), "rows": {}}
    print(f"\n{'configuration':<44}{'accuracy':>9}{'macro-F1':>10}{'tracked':>9}{'fixed':>7}{'broken':>8}")
    for k, pred in preds.items():
        flag = configs[k]
        fixed = int(sum(1 for i in flag if y_pred[i] != y_true[i] and pred[i] == y_true[i]))
        broken = int(sum(1 for i in flag if y_pred[i] == y_true[i] and pred[i] != y_true[i]))
        acc, f1 = float((pred == y_true).mean()), float(f1_score(y_true, pred, average="macro", labels=labels))
        per_class = dict(zip(names, f1_score(y_true, pred, average=None, labels=labels).round(4).tolist()))
        results["rows"][k] = {"accuracy": acc, "macro_f1": f1, "n_tracked": len(flag), "fixed": fixed, "broken": broken, "per_class_f1": per_class}
        print(f"{k:<44}{acc:>9.4f}{f1:>10.4f}{len(flag):>9}{fixed:>7}{broken:>8}")

    kc, km = list(configs)[1], list(configs)[2]
    ok_c, ok_m = preds[kc] == y_true, preds[km] == y_true
    only_c, only_m = int((ok_c & ~ok_m).sum()), int((ok_m & ~ok_c).sum())
    p = binomtest(only_m, only_c + only_m, 0.5).pvalue if only_c + only_m else 1.0
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y_true), len(y_true))
        diffs.append(ok_m[i].mean() - ok_c[i].mean())
    results["gate_comparison"] = {
        "mc_minus_conf_accuracy": float(ok_m.mean() - ok_c.mean()), "bootstrap_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "correct_only_under_conf_gate": only_c, "correct_only_under_mc_gate": only_m, "mcnemar_exact_p": float(p),
        "per_case_accuracy": {c: {"conf": float(ok_c[case_of == c].mean()), "mc": float(ok_m[case_of == c].mean())} for c in sorted(set(case_of))},
    }
    g = results["gate_comparison"]
    print(f"\nMC gate minus confidence gate: accuracy {g['mc_minus_conf_accuracy']:+.4f}  95% bootstrap CI [{g['bootstrap_ci95'][0]:+.4f}, {g['bootstrap_ci95'][1]:+.4f}]")
    print(f"instances correct only under the confidence gate: {only_c}, only under the MC gate: {only_m}; exact McNemar p = {p:.3f}")
    print("per-case accuracy (conf / MC):", {c: (round(v['conf'], 3), round(v['mc'], 3)) for c, v in g["per_case_accuracy"].items()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
