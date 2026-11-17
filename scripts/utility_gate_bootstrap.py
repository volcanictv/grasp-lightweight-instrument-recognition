"""Paired case-bootstrap of the learned gate against the S_1 gate: corrections captured, introduced errors, and net gain at fixed budgets (arrays from scripts/utility_gate.py --order-dir)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SHARES = [0.05, 0.10, 0.20, 0.329]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--model", default="logistic")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = [np.load(args.dir / f"{args.model}_s{s}.npz") for s in (42, 43, 44)]
    cases = sorted(set(runs[0]["case"].tolist()))
    idx = {c: np.where(runs[0]["case"] == c)[0] for c in cases}
    res = {}
    for sh in SHARES:
        n = len(runs[0]["delta"])
        top = int(round(sh * n))
        # per seed, per case: corrections and introduced errors selected by each gate, and the refine-all totals
        stat = {tag: np.zeros((3, len(cases), 2)) for tag in ("S1", "learned")}
        tot = np.zeros((3, len(cases), 2))
        for s, r in enumerate(runs):
            d = r["delta"]
            for tag, order in (("S1", r["order_s1"]), ("learned", r["order"])):
                sel = np.zeros(n, bool)
                sel[order[:top]] = True
                for ci, c in enumerate(cases):
                    m = idx[c]
                    stat[tag][s, ci] = [((d[m] == 1) & sel[m]).sum(), ((d[m] == -1) & sel[m]).sum()]
            for ci, c in enumerate(cases):
                m = idx[c]
                tot[s, ci] = [(d[m] == 1).sum(), (d[m] == -1).sum()]

        def summary(w):
            out = {}
            for tag in ("S1", "learned"):
                c = (stat[tag][:, :, 0] * w).sum(1).mean(); dm = (stat[tag][:, :, 1] * w).sum(1).mean()
                out[tag] = {"net": c - dm, "C_share": c / (tot[:, :, 0] * w).sum(1).mean(), "D_share": dm / (tot[:, :, 1] * w).sum(1).mean()}
            return out

        point = summary(np.ones(len(cases)))
        boots = []
        for _ in range(args.boot):
            w = np.bincount(rng.integers(0, len(cases), len(cases)), minlength=len(cases)).astype(float)
            b = summary(w)
            boots.append([b["learned"]["net"] - b["S1"]["net"], b["learned"]["C_share"] - b["S1"]["C_share"], b["learned"]["D_share"] - b["S1"]["D_share"]])
        boots = np.array(boots)
        res[str(sh)] = {"point": point, "net_diff": float(point["learned"]["net"] - point["S1"]["net"]),
                        "net_diff_ci95_case": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
                        "C_share_diff_ci95_case": [float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5))],
                        "D_share_diff_ci95_case": [float(np.percentile(boots[:, 2], 2.5)), float(np.percentile(boots[:, 2], 97.5))]}
        print(sh, {t: {k: round(v, 3) for k, v in x.items()} for t, x in point.items()}, "net diff", round(res[str(sh)]["net_diff"], 1), [round(v, 1) for v in res[str(sh)]["net_diff_ci95_case"]])
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
