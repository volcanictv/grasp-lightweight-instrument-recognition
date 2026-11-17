"""Multiplicity-controlled suite for the TAPIS comparison: where, on difficulty scales fixed in advance, is our final pipeline reliably ahead of (or behind) TAPIS.

Why a suite and not a scan: every slice tried is one more chance for a lucky gap, so all tests share one family and the intervals are simultaneous (bootstrap max-t over the whole
family, 10,000 frame-cluster resamples). A row is called reliable only when its simultaneous interval excludes zero. The family includes every stratum already looked at
(glare and the similar-class pairs too), whether or not it is shown elsewhere.

Tests
  flags            the eleven challenge flags, their unions, the four confusable pairs, the clear classes, clean, compound, hard union
  hardest tails    hardest quartile and decile of five continuous difficulty scales (class-relative visible area, edge share, blood share and low contrast and low saturation,
                   the last three ranked within each case), plus crowding (3+ instruments) and cover by another instrument at 10% and 25%
  dose-response    for each of the five scales, gap(final - TAPIS) in the hardest quartile minus the gap in the easiest quartile
Also written: per-quartile accuracy for every scale (the dose-response curves) and per-case gaps for every reliable row.

    python scripts/challenge_strata_suite.py --per-instrument <csv> --feat-test <csv> --out-dir docs/reports/challenge_strata
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PAIRS = {"pair: Bipolar / Prograsp": (0, 1), "pair: Prograsp / Suction": (1, 4), "pair: Prograsp / Grasper": (1, 6), "pair: Suction / Grasper": (4, 6)}
FLAGS = ["occ_instrument", "occ_fragmented", "small_visible", "blood", "specular", "lowcontrast_haze", "corner", "edge_dominant", "occlusion_any", "degraded_any", "boundary_any", "compound", "clean"]
B = 10000


def load(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: (np.array([r[k] for r in rows]) if k in ("file", "case") else np.array([float(r[k]) for r in rows])) for k in rows[0]}


def within_case_rank(case: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x))
    for c in np.unique(case):
        m = case == c
        out[m] = (rankdata(x[m]) - 0.5) / m.sum()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-instrument", type=Path, required=True)
    ap.add_argument("--feat-test", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=B)
    args = ap.parse_args()

    P, F = load(args.per_instrument), load(args.feat_test)
    n = len(P["gt"])
    assert np.array_equal(P["instrument"], F["instrument"]) and np.array_equal(P["gt"], F["label"])
    y = P["gt"].astype(int)
    case = P["case"]
    ok_tap = (P["tapis"].astype(int) == y)
    ok_fin = [(P[f"final_s{s}"].astype(int) == y) for s in (42, 43, 44)]
    ok_sin = [(P[f"single_s{s}"].astype(int) == y) for s in (42, 43, 44)]
    OK = np.stack([ok_tap] + ok_fin + ok_sin, axis=1).astype(np.float32)  # n x 7

    area_rank = np.zeros(n)
    for c in range(7):
        m = y == c
        area_rank[m] = (rankdata(F["area_frac"][m]) - 0.5) / m.sum()
    scales = {"visible area (class-relative, small = hard)": 1 - area_rank,
              "edge share of mask": F["edge_frac"],
              "blood share around instrument (within case)": within_case_rank(case, F["blood_frac"]),
              "low contrast around instrument (within case)": 1 - within_case_rank(case, F["ring_contrast"]),
              "low saturation around instrument (within case)": 1 - within_case_rank(case, F["ring_sat"])}

    tests: dict[str, np.ndarray] = {"All instruments": np.ones(n, bool)}
    flag = {k: P[k] > 0 for k in FLAGS if k in P}
    for k in FLAGS:
        tests[f"flag: {k}"] = flag[k]
    hard = flag["occlusion_any"] | flag["blood"] | flag["lowcontrast_haze"] | flag["boundary_any"]
    tests["flag: hard union (no glare, no pairs)"] = hard
    for nm, (a, b) in PAIRS.items():
        tests[nm] = (y == a) | (y == b)
    tests["clear classes (Needle driver, Scissors, Clip applier)"] = np.isin(y, [2, 3, 5])
    tests["crowded frame (3+ instruments)"] = F["n_in_frame"] >= 3
    tests["covered by another instrument >= 25%"] = F["other_in_box"] >= 0.25
    quart = {}
    for nm, s in scales.items():
        q = np.quantile(s, [0.25, 0.75, 0.9])
        tests[f"hardest quartile: {nm}"] = s >= q[1]
        tests[f"hardest decile: {nm}"] = s >= q[2]
        quart[nm] = (s <= q[0], s >= q[1])

    fidx = np.array([hash((case[i], P["file"][i])) for i in range(n)])
    _, fidx = np.unique(fidx, return_inverse=True)
    nf = fidx.max() + 1
    rng = np.random.default_rng(0)
    W = rng.multinomial(nf, np.full(nf, 1 / nf), size=args.boot).astype(np.float32)[:, fidx]

    def boot_acc(m: np.ndarray) -> np.ndarray:
        den = np.maximum(W @ m.astype(np.float32), 1e-9)
        return (W @ (OK * m[:, None].astype(np.float32))) / den[:, None]  # B x 7

    def point(m: np.ndarray) -> np.ndarray:
        return (OK * m[:, None]).sum(0) / m.sum()

    def combine(a: np.ndarray) -> np.ndarray:  # columns: tapis, fin x3, sin x3 -> tapis, fin mean, sin mean (last axis)
        return np.stack([a[..., 0], a[..., 1:4].mean(-1), a[..., 4:7].mean(-1)], axis=-1)

    names, est, bootd, meta = [], [], [], []
    for nm, m in tests.items():
        a_b, a_p = combine(boot_acc(m)), combine(point(m))
        d_b, d_p = a_b[:, 1] - a_b[:, 0], a_p[1] - a_p[0]
        names.append(nm); est.append(d_p); bootd.append(d_b)
        meta.append({"type": "stratum", "n": int(m.sum()), "n_frames": int(len(set(fidx[m]))), "n_cases": int(len(set(case[m]))), "tapis": float(a_p[0]), "ours_final": float(a_p[1]), "ours_single": float(a_p[2]),
                     "tapis_ci": np.quantile(a_b[:, 0], [0.025, 0.975]).tolist(), "final_ci": np.quantile(a_b[:, 1], [0.025, 0.975]).tolist()})
    for nm, (easy, hardq) in quart.items():
        gb = []
        gp = []
        for m in (easy, hardq):
            a_b, a_p = combine(boot_acc(m)), combine(point(m))
            gb.append(a_b[:, 1] - a_b[:, 0]); gp.append(a_p[1] - a_p[0])
        names.append(f"dose-response: gap hardest quartile minus easiest quartile, {nm}")
        est.append(gp[1] - gp[0]); bootd.append(gb[1] - gb[0])
        meta.append({"type": "dose-response", "gap_easiest_quartile": float(gp[0]), "gap_hardest_quartile": float(gp[1])})

    est, D = np.array(est), np.stack(bootd, axis=1)
    se = D.std(0, ddof=1)
    z = np.max(np.abs(D - est) / np.maximum(se, 1e-9), axis=1)
    c = float(np.quantile(z, 0.95))
    p_raw = np.array([min(1.0, 2 * min(((D[:, t] <= 0).sum() + 1) / (args.boot + 1), ((D[:, t] >= 0).sum() + 1) / (args.boot + 1))) for t in range(D.shape[1])])
    order = np.argsort(p_raw)
    holm = np.empty_like(p_raw)
    run = 0.0
    for r, t in enumerate(order):
        run = max(run, min(1.0, (len(p_raw) - r) * p_raw[t]))
        holm[t] = run

    # per-case gaps
    cases = sorted(set(case))
    out = []
    for t, nm in enumerate(names):
        row = {"test": nm, "gap": float(est[t]), "ci95": np.quantile(D[:, t], [0.025, 0.975]).tolist(), "simultaneous_ci95": [float(est[t] - c * se[t]), float(est[t] + c * se[t])],
               "p_raw": float(p_raw[t]), "p_holm": float(holm[t]), **meta[t]}
        if meta[t]["type"] == "stratum":
            m = tests[nm]
            pc = {}
            for cs in cases:
                mm = m & (case == cs)
                if mm.sum() >= 15:
                    a = combine(point(mm))
                    pc[cs] = {"n": int(mm.sum()), "gap": float(a[1] - a[0])}
            row["per_case"] = pc
        out.append(row)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # dose-response curves: accuracy by quartile of each scale
    curves = {}
    for nm, s in scales.items():
        edges = np.quantile(s, [0, 0.25, 0.5, 0.75, 1.0])
        qs = []
        for k in range(4):
            m = (s >= edges[k]) & ((s <= edges[k + 1]) if k == 3 else (s < edges[k + 1]))
            a_b, a_p = combine(boot_acc(m)), combine(point(m))
            qs.append({"quartile": k + 1, "n": int(m.sum()), "tapis": float(a_p[0]), "tapis_ci": np.quantile(a_b[:, 0], [0.025, 0.975]).tolist(), "final": float(a_p[1]),
                       "final_ci": np.quantile(a_b[:, 1], [0.025, 0.975]).tolist(), "single": float(a_p[2]), "single_ci": np.quantile(a_b[:, 2], [0.025, 0.975]).tolist()})
        curves[nm] = qs
    (out_dir / "suite_results.json").write_text(json.dumps({"family_size": len(names), "simultaneous_multiplier": c, "bootstrap_resamples": args.boot, "tests": out, "dose_response_curves": curves}, indent=1))

    lines = [f"Family of {len(names)} tests, simultaneous 95% multiplier {c:.2f} (a plain 95% interval would use 1.96).", "",
             "| Test | n | TAPIS | Ours final | Gap (pts) | Simultaneous 95% CI | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for r in sorted(out, key=lambda r: -r["gap"]):
        if r["type"] == "stratum":
            lines.append(f"| {r['test']} | {r['n']} | {100*r['tapis']:.1f} | {100*r['ours_final']:.1f} | {100*r['gap']:+.1f} | [{100*r['simultaneous_ci95'][0]:+.1f}, {100*r['simultaneous_ci95'][1]:+.1f}] | {r['p_holm']:.3f} |")
        else:
            lines.append(f"| {r['test']} | | gap easiest {100*r['gap_easiest_quartile']:+.1f} | gap hardest {100*r['gap_hardest_quartile']:+.1f} | {100*r['gap']:+.1f} | [{100*r['simultaneous_ci95'][0]:+.1f}, {100*r['simultaneous_ci95'][1]:+.1f}] | {r['p_holm']:.3f} |")
    (out_dir / "suite_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
