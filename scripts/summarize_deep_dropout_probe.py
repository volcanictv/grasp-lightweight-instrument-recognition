"""One comparison table over the deep-dropout probe configs
(probe_deep_dropout.py -> complementarity_analysis.py per config), against the
head-only baseline in complementarity_analysis.json.

Columns answer two separate questions:
  strength     : vote_dis / mutual_info AUROC (does deeper dropout make the
                 disagreement signal a better error detector by itself?)
  complementarity : dAUROC of confidence+signal over confidence alone
                 (leave-one-case-out), the conditional AUROC among
                 near-equal-confidence instances, and the same conditional
                 AUROC for a pure confidence variant (conf_mc_unc) as the
                 control that separates real information from leftover
                 confidence variation.

Usage:
    python scripts/summarize_deep_dropout_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "docs" / "reports"
CONFIGS = [f"{place}_p{p}" for place in ("late", "midlate") for p in ("10", "20", "30")]


def row(name: str, r: dict) -> list:
    cond = r["E7"]["conditional_auroc"]
    return [
        name,
        f"{r['E0']['vote_dis']['auroc']:.3f}",
        f"{r['E0']['mutual_info']['auroc']:.3f}",
        f"{r['E2']['C conf+vote_dis']['delta_auroc_vs_A']['mean']:+.4f}",
        f"{r['E2']['D conf+mutual_info']['delta_auroc_vs_A']['mean']:+.4f}",
        f"{r['E2']['F conf+vote_dis+det_vote_dis+mi']['delta_auroc_vs_A']['mean']:+.4f}",
        f"{cond['vote_dis|10strata']['auroc']:.3f}/{cond['vote_dis|20strata']['auroc']:.3f}",
        f"{cond['mutual_info|10strata']['auroc']:.3f}/{cond['mutual_info|20strata']['auroc']:.3f}",
        f"{cond['conf_mc_unc|10strata']['auroc']:.3f}/{cond['conf_mc_unc|20strata']['auroc']:.3f}",
        f"{r['E7']['gbm']['four_features']:.3f}",
    ]


def main() -> None:
    header = ["config", "vote_dis AUROC", "MI AUROC", "dAUROC +vote_dis", "dAUROC +MI", "dAUROC +all", "cond vote_dis 10/20",
              "cond MI 10/20", "cond conf ctrl 10/20", "GBM +all"]
    rows = [row("head-only (baseline)", json.loads((REPORTS / "complementarity_analysis.json").read_text()))]
    for c in CONFIGS:
        path = REPORTS / "deep_dropout" / f"{c}.json"
        if path.exists():
            rows.append(row(c, json.loads(path.read_text())))
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for r in [header] + rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))
    print("\ndAUROC is vs confidence alone, leave-one-case-out; positive = complementary. "
          "Read 'cond' against the confidence control column, not against 0.5.")


if __name__ == "__main__":
    main()
