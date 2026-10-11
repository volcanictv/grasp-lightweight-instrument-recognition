"""Reporting pass over the frozen gate evaluation (scripts/gate_calibration.py evaluate): per-case table, S1 vs max-belief overlap, pooling check, paired case bootstrap.
CPU only, reads gate_calibration.json and gate_outputs.pkl; changes no threshold.

    python scripts/gate_report.py --dir <folder with gate_calibration.json and gate_outputs.pkl> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bootstrap_cis import ci, frame_arrays, metrics  # noqa: E402  (existing bootstrap machinery: vectorised metrics with frame weights)
from surgical_ai.evaluation.semantic_iou import aggregate  # noqa: E402

M = ["mIoU", "IoU", "mcIoU"]
GATES = ["S1", "maxbelief"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--b", type=int, default=10000)
    args = ap.parse_args()
    fz = json.loads((args.dir / "gate_calibration.json").read_text())
    dump = pickle.loads((args.dir / "gate_outputs.pkl").read_bytes())
    fr = dump["frames"]
    names = [f[0] for f in fr["single"]]
    assert all([f[0] for f in fr[c]] == names for c in GATES), "frame lists differ between configurations"
    cases = np.array([n.split("/")[0] for n in names])
    case_list = sorted(set(cases.tolist()))
    arr = {c: frame_arrays(fr[c]) for c in fr}
    out: dict = {"seeds": "one classifier seed (42) per fold; the four fold checkpoint sets are independent; no seed averaging", "cases": case_list}

    # 3. pooling: all evaluation frames of the five cases in one pool (the repository's aggregate), against the equal-weight mean of the fold values
    pooled = {}
    for c in fr:
        agg = aggregate([(f[1], f[2]) for f in fr[c]])
        mine = metrics(arr[c], np.ones(len(names)))
        assert np.allclose(mine, [agg[m] for m in M], atol=1e-12)
        fold_vals = []
        for k, e in fz["folds"].items():
            if not e["evaluation_cases"]:
                continue
            idx = [j for j, n in enumerate(names) if n.split("/")[0] in e["evaluation_cases"]]
            fold_vals.append([aggregate([(fr[c][j][1], fr[c][j][2]) for j in idx])[m] for m in M])
        pooled[c] = {"frame_level_pool": {m: 100 * float(agg[m]) for m in M}, "n_frames": len(names),
                     "equal_weight_mean_of_fold_values": {m: 100 * float(v) for m, v in zip(M, np.mean(fold_vals, axis=0))}}
    out["pooling"] = pooled

    # 1. per evaluation case
    rows = []
    case_cfg: dict = {}
    for k, e in fz["folds"].items():
        d = dump["folds"].get(int(k))
        if not e["evaluation_cases"] or d is None:
            continue
        ev, cs = d["evaluation_mask"], d["cases"]
        for case in e["evaluation_cases"]:
            ix = [j for j, n in enumerate(names) if n.startswith(case + "/")]
            base = aggregate([(fr["single"][j][1], fr["single"][j][2]) for j in ix])
            n_case = int(((cs == case) & ev).sum())
            for g in GATES:
                gt = aggregate([(fr[g][j][1], fr[g][j][2]) for j in ix])
                m = (cs == case) & ev
                rows.append({"fold": f"cv5_f{k}", "classifier_seed": 42, "evaluation_case": case, "n_evaluation_instruments": n_case,
                             "calibration_cases": "+".join(e["calibration_cases"]), "n_calibration_instruments": e["n_calibration"], "gate": g,
                             "threshold": e["scores"][g]["cutoff_repr"], "calibration_fraction": round(e["scores"][g]["calibration_fraction"], 4),
                             "evaluation_fraction": round(float(d["sel"][g][m].sum() / m.sum()), 4),
                             "baseline_mIoU": round(100 * base["mIoU"], 3), "gated_mIoU": round(100 * gt["mIoU"], 3), "gain_mIoU": round(100 * (gt["mIoU"] - base["mIoU"]), 3)})
    out["per_case_rows"] = rows

    # 2. S1 vs max belief
    ov = {}
    for k, d in dump["folds"].items():
        ev = d["evaluation_mask"]
        a, b = d["sel"]["S1"] & ev, d["sel"]["maxbelief"] & ev
        diff_lab = int(((d["pred"]["S1"] != d["pred"]["maxbelief"]) & ev).sum())
        ov[str(k)] = {"selected_S1": int(a.sum()), "selected_maxbelief": int(b.sum()), "selected_by_both": int((a & b).sum()), "only_S1": int((a & ~b).sum()), "only_maxbelief": int((~a & b).sum()),
                      "union": int((a | b).sum()), "jaccard": float((a & b).sum() / (a | b).sum()), "final_labels_differ": diff_lab,
                      "final_labels_differ_among_one_gate_only": int(((a ^ b) & (d["pred"]["S1"] != d["pred"]["maxbelief"])).sum())}
    tot = {key: sum(v[key] for v in ov.values()) for key in ("selected_S1", "selected_maxbelief", "selected_by_both", "only_S1", "only_maxbelief", "union", "final_labels_differ")}
    out["gate_overlap"] = {"per_fold": ov, "total": tot}
    out["gate_full_precision"] = {c: {m: 100 * float(v) for m, v in zip(M, metrics(arr[c], np.ones(len(names))))} for c in fr}
    out["gate_max_abs_metric_difference"] = float(np.abs(metrics(arr["S1"], np.ones(len(names))) - metrics(arr["maxbelief"], np.ones(len(names)))).max() * 100)
    out["gate_frames_with_different_class_ious"] = int(sum(1 for x, y in zip(fr["S1"], fr["maxbelief"]) if x[1] != y[1]))

    # 4. paired case bootstrap: resample the five evaluated cases with replacement, same resample for baseline and gated
    case_ix = {c: np.where(cases == c)[0] for c in case_list}
    correct = {c: {} for c in fr}
    for k, d in dump["folds"].items():
        ev, cs = d["evaluation_mask"], d["cases"]
        for case in set(cs[ev]):
            m = (cs == case) & ev
            for c in fr:
                correct[c][case] = (int((d["pred"][c][m] == d["y"][m]).sum()), int(m.sum()))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(case_list), size=(args.b, len(case_list)))
    boot = {g: {m: [] for m in M + ["instance_accuracy"]} for g in GATES}
    for dr in draws:
        counts = np.bincount(dr, minlength=len(case_list))
        w = np.zeros(len(names))
        for ci_, c in enumerate(case_list):
            w[case_ix[c]] = counts[ci_]
        mb = metrics(arr["single"], w)
        for g in GATES:
            mg = metrics(arr[g], w)
            for i, m in enumerate(M):
                boot[g][m].append(100 * (mg[i] - mb[i]))
            na = sum(counts[i] * correct["single"][c][0] for i, c in enumerate(case_list)); ng = sum(counts[i] * correct[g][c][0] for i, c in enumerate(case_list))
            nn = sum(counts[i] * correct["single"][c][1] for i, c in enumerate(case_list))
            boot[g]["instance_accuracy"].append(100 * (ng - na) / nn)
    est = {}
    for g in GATES:
        pt = metrics(arr[g], np.ones(len(names))) - metrics(arr["single"], np.ones(len(names)))
        est[g] = {m: {"point": 100 * float(pt[i]), "ci95": [float(x) for x in ci(np.array(boot[g][m]))], "share_of_resamples_positive": float((np.array(boot[g][m]) > 0).mean())} for i, m in enumerate(M)}
        tn = sum(correct[g][c][0] for c in case_list) - sum(correct["single"][c][0] for c in case_list)
        N = sum(correct["single"][c][1] for c in case_list)
        est[g]["instance_accuracy"] = {"point": 100 * tn / N, "ci95": [float(x) for x in ci(np.array(boot[g]["instance_accuracy"]))],
                                       "share_of_resamples_positive": float((np.array(boot[g]["instance_accuracy"]) > 0).mean())}
    out["paired_case_bootstrap"] = {"unit": "whole evaluation case (5 cases), resampled with replacement, baseline and gated paired on the same resample", "B": args.b,
                                    "distinct_resamples_possible": 126, "gain_gated_minus_single_pass_in_points": est,
                                    "interpretation": "uncertainty across the five evaluated cases conditional on the fitted models; it does not cover retraining, seed variance, or estimation of the thresholds"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gate_report.json").write_text(json.dumps(out, indent=1))
    with open(args.out_dir / "gate_per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: out[k] for k in ("pooling", "gate_overlap", "gate_full_precision", "gate_max_abs_metric_difference", "gate_frames_with_different_class_ious", "paired_case_bootstrap")}, indent=1))


if __name__ == "__main__":
    main()
