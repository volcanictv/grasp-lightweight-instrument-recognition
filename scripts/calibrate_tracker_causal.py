"""Fold1 calibration of the evidential gate threshold and the causal look-back k, per tracker.

Same threshold rule as evidential_three_member_yolo.py (maximise accuracy, then the highest threshold
within 0.0010 of the best), applied to the last k past frames plus the annotated frame, for each k.
k is the shortest look-back whose calibrated fold1 accuracy is within 0.0010 of the best k, so a
longer window is only chosen when it buys accuracy. Fold1 only: the official test is never read here.

Usage:
    python scripts/calibrate_tracker_causal.py --tracker edgetam=<dir> --tracker yolo=<dir> --ks 3,5,10,15,20 \\
        --out calibration_causal.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score

import evidential_three_member_yolo as t
from compare_trackers_causal import load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", action="append", required=True, help="name=directory")
    ap.add_argument("--ks", default="3,5,10,15,20")
    ap.add_argument("--extract", type=Path, default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_fold1_s42.npz")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "evidential" / "ens_E_fold1_s42_lam0p01a10.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    order = t.member_order(args.config)
    y, base, s1 = t.base_scores(args.extract)
    grid = sorted({round(float(x), 6) for x in np.percentile(s1, t.PERCENTILES)}, reverse=True)
    need = np.where(s1 >= min(grid))[0]
    result: dict = {"base_accuracy": float((base == y).mean()), "grid": grid, "trackers": {}}
    for spec in args.tracker:
        name, directory = spec.split("=", 1)
        tr = load(Path(directory))
        missing = [i for i in need if int(i) not in tr]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} of {len(need)} grid instances have no tracking yet")
        per_k = {}
        for k in (int(x) for x in args.ks.split(",")):
            tracked = {}
            for i in need:
                logits, c = tr[int(i)]
                tracked[int(i)] = t.track_predict(logits[max(0, c - k):c + 1], order)
            rows = {}
            for tau in grid:
                flag = s1 >= tau
                pred = t.apply_gate(base, tracked, flag)
                rows[str(tau)] = t.row(y, base, pred, flag, threshold=tau)
                rows[str(tau)]["per_class_f1"] = f1_score(y, pred, average=None, labels=range(7)).round(4).tolist()
            best = max(r["accuracy"] for r in rows.values())
            chosen = max(r["threshold"] for r in rows.values() if r["accuracy"] >= best - t.TOLERANCE)
            per_k[str(k)] = {"chosen_threshold": chosen, "chosen_row": rows[str(chosen)], "rows": rows}
        best_acc = max(v["chosen_row"]["accuracy"] for v in per_k.values())
        k_star = min(int(k) for k, v in per_k.items() if v["chosen_row"]["accuracy"] >= best_acc - t.TOLERANCE)
        result["trackers"][name] = {"chosen_k": k_star, "chosen_threshold": per_k[str(k_star)]["chosen_threshold"], "by_k": per_k}
        print(f"\n{name}: chosen k={k_star}, threshold {per_k[str(k_star)]['chosen_threshold']:.6f}")
        print(f"{'k':>4}{'tau':>10}{'tracked%':>9}{'acc':>8}{'macroF1':>9}{'fixed':>6}{'broken':>7}")
        for k, v in per_k.items():
            r = v["chosen_row"]
            print(f"{k:>4}{v['chosen_threshold']:>10.6f}{100 * r['share']:>9.1f}{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}{r['fixed']:>6}{r['broken']:>7}")
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
