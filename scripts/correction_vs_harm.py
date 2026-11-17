"""Does the uncertainty score separate helpful from harmful refinement?

Reads refinement_transitions.csv (scripts/refinement_transitions.py: one row per instrument and seed with the refine-all outcome). Two analyses, both on the
official test instruments:

1. Among instruments whose correctness changes under refine-all (corrections vs introduced errors), the AUROC of the score for telling the two apart, next to the
   AUROC of the same score for detecting single-pass errors. Case-bootstrap interval over the five test cases, seeds averaged.
2. The share of all corrections, C(S)/C(all), and of all introduced errors, D(S)/D(all), captured by refining the top-B instruments, against budget B, with
   random selection (B/N) and the correction-first oracle for reference.

    python scripts/correction_vs_harm.py --csv <refinement_transitions.csv> --out-json <json> --out-fig <pdf>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

SEEDS = ("42", "43", "44")
SCORES = {"S1": ("s1", 1.0), "maximum belief": ("max_belief", -1.0)}  # sign makes larger mean more uncertain


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    by_seed = {s: [r for r in rows if r["seed"] == s] for s in SEEDS}
    cases = sorted({r["case"] for r in rows})
    rng = np.random.default_rng(0)
    res: dict = {"auroc_change": {}, "auroc_error_detection": {}}

    for name, (col, sign) in SCORES.items():
        per_case = {}
        for s in SEEDS:
            for c in cases:
                sub = [r for r in by_seed[s] if r["case"] == c]
                per_case[(s, c)] = {
                    "c": np.array([sign * float(r[col]) for r in sub if r["transition"] == "wrong_to_correct"]),
                    "d": np.array([sign * float(r[col]) for r in sub if r["transition"] == "correct_to_wrong"]),
                    "w": np.array([sign * float(r[col]) for r in sub if r["gt"] != r["single_pred"]]),
                    "r": np.array([sign * float(r[col]) for r in sub if r["gt"] == r["single_pred"]]),
                }

        def pooled(draw, key_pos, key_neg):
            vals = []
            for s in SEEDS:
                pos = np.concatenate([per_case[(s, cases[i])][key_pos] for i in draw])
                neg = np.concatenate([per_case[(s, cases[i])][key_neg] for i in draw])
                vals.append(auroc(pos, neg))
            return float(np.mean(vals))  # nan when a draw has no corrections or no introduced errors; such draws are skipped by nanpercentile

        point = list(range(len(cases)))
        for tag, kp, kn in (("auroc_change", "c", "d"), ("auroc_error_detection", "w", "r")):
            boots = [pooled(rng.integers(0, len(cases), len(cases)), kp, kn) for _ in range(args.boot)]
            res[tag][name] = {"point": pooled(point, kp, kn), "ci95_case": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                              "per_seed": [auroc(np.concatenate([per_case[(s, c)][kp] for c in cases]), np.concatenate([per_case[(s, c)][kn] for c in cases])) for s in SEEDS]}
    res["n_changing_per_seed"] = {s: {"corrections": sum(r["transition"] == "wrong_to_correct" for r in by_seed[s]),
                                      "introduced": sum(r["transition"] == "correct_to_wrong" for r in by_seed[s])} for s in SEEDS}

    # capture curves
    shares = np.linspace(0.0, 1.0, 101)
    curves = {"S1": ([], []), "maximum belief": ([], [])}
    for s in SEEDS:
        sub = by_seed[s]
        n = len(sub)
        corr = np.array([r["transition"] == "wrong_to_correct" for r in sub], float)
        dmg = np.array([r["transition"] == "correct_to_wrong" for r in sub], float)
        for name, (col, sign) in SCORES.items():
            score = sign * np.array([float(r[col]) for r in sub])
            order = np.argsort(-score, kind="stable")
            cc, dd = np.cumsum(corr[order]) / corr.sum(), np.cumsum(dmg[order]) / dmg.sum()
            ks = np.round(shares * n).astype(int)
            curves[name][0].append(np.concatenate([[0], cc])[ks])
            curves[name][1].append(np.concatenate([[0], dd])[ks])
    n_all, c_all, d_all = len(by_seed["42"]), np.mean([sum(r["transition"] == "wrong_to_correct" for r in by_seed[s]) for s in SEEDS]), np.mean([sum(r["transition"] == "correct_to_wrong" for r in by_seed[s]) for s in SEEDS])
    b = shares * n_all
    oracle_c = np.minimum(1.0, b / c_all)
    oracle_d = np.maximum(0.0, (b - (n_all - d_all)) / d_all)
    k941 = int(round(941 / 2861 * 100))
    res["at_32.9pct"] = {name: {"C_share": float(np.mean(c, axis=0)[33]), "D_share": float(np.mean(d, axis=0)[33])} for name, (c, d) in curves.items()}
    Path(args.out_json).write_text(json.dumps(res, indent=1))

    INK, ACCENT, RED, GREY = "#1f2933", "#2b6cb0", "#c0392b", "#9aa5b1"
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    x = 100 * shares
    ax.plot(x, x, color=GREY, ls=":", lw=1.0, label="random (both)")
    ax.plot(x, 100 * oracle_c, color=INK, ls="--", lw=0.9, label="oracle, corrections")
    ax.plot(x, 100 * np.mean(curves["S1"][0], axis=0), color=ACCENT, lw=1.6, label="$S_1$, corrections")
    ax.plot(x, 100 * np.mean(curves["S1"][1], axis=0), color=RED, lw=1.6, label="$S_1$, introduced errors")
    ax.axvline(32.9, color=GREY, lw=0.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel("instruments refined (%)", fontsize=7)
    ax.set_ylabel("share of refine-all total (%)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(color="#e1e5ea", lw=0.5)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out_fig)
    for tag in ("auroc_change", "auroc_error_detection"):
        print(tag, {k: (round(v["point"], 3), [round(x, 3) for x in v["ci95_case"]]) for k, v in res[tag].items()})
    print(res["at_32.9pct"], res["n_changing_per_seed"])


if __name__ == "__main__":
    main()
