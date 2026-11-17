"""Exploratory: does a fusion-time acceptance rule cut the introduced errors without losing the corrections?

Reads refinement_transitions.csv (scripts/refinement_transitions.py). For the instruments a policy refines, the keyframe label is overturned only when the label
changes and the fused confidence exceeds the keyframe confidence by a margin. Margins are fixed in advance and none is selected: the test cases are the only
data with refine-all outcomes, so this is a post hoc look at classification counts, not a tuned method.

    python scripts/acceptance_rule_margin.py --csv <refinement_transitions.csv> --policy sel_s1_top833 --out <json>
"""
from __future__ import annotations

import argparse
import csv
import json

MARGINS = [None, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--policy", default="sel_s1_top833", help="selection column; in the top-941 run of the transitions script this column holds the 941 highest-S1 instances")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    res = {}
    for m in MARGINS:
        per_seed = []
        for seed in sorted({r["seed"] for r in rows}):
            corr = intro = 0
            for r in rows:
                if r["seed"] != seed or r["has_valid_track"] != "True" or r[args.policy] != "True":
                    continue
                gt, sp, rp = r["gt"], r["single_pred"], r["refine_all_pred"]
                overturn = rp != sp and (m is None or float(r["refine_all_conf"]) - float(r["single_conf"]) > m)
                final = rp if overturn else sp
                corr += sp != gt and final == gt
                intro += sp == gt and final != gt
            per_seed.append((corr, intro))
        c, i = (sum(x[k] for x in per_seed) / len(per_seed) for k in (0, 1))
        res["always overturn" if m is None else f"margin {m}"] = {"corrections": c, "introduced": i, "net": c - i}
    json.dump(res, open(args.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:16s} corrections {v['corrections']:6.1f} introduced {v['introduced']:5.1f} net {v['net']:6.1f}")


if __name__ == "__main__":
    main()
