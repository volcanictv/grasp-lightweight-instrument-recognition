"""Merge the per-dataset analysis, final-eval and paired-comparison jsons of the
EndoVis-trained evidential experiment into one results.json
(docs/reports/evidential_endovis/).

Usage:
    python scripts/endovis_evidential_collect.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs" / "reports" / "evidential_endovis"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> None:
    out = {"design": "docs/DECISIONS.md 2026-09-29, evidential ensemble trained on EndoVis with the same gate",
           "mc_reference": {
               "2018": {"base": [0.8817703497369236, 0.8629795619460985], "tracked_final": [0.9278861033735686, 0.9187404567418348],
                        "n_tracked": 878, "source": "docs/reports/endovis_transfer/stage1_2018/final_pipeline.json"},
               "2017": {"base": [0.9170597484276729, 0.8855360165896442], "n_flagged": 506, "tracking": "not possible, images missing"}}}
    for ds in ("2018", "2017"):
        d = ROOT / f"stage1_{ds}"
        out[ds] = {"train_result": load(d / "result.json"), "analysis": load(d / "analysis.json")}
    d = ROOT / "stage1_2018"
    out["2018"]["final_pipeline"] = load(d / "final.json")
    out["2018"]["paired_vs_mc"] = load(d / "paired.json")
    (ROOT / "results.json").write_text(json.dumps(out, indent=1))
    print("wrote", ROOT / "results.json")


if __name__ == "__main__":
    main()
