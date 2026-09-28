"""Domain-transfer con for the evidential-gated pipeline: the official-split
evidential ensemble run zero-shot (no retraining, no tracking, single
deterministic pass) on EndoVis 2018/2017, next to the shipped CE pipeline's
existing zero-shot numbers (docs/DECISIONS.md 2026-09-28).

Usage:
    python scripts/evidential_endovis_con.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score

from surgical_ai.evaluation.evidential import alpha_from_logits

WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}
CE_ZERO_SHOT = {"2018": {"accuracy": 0.528, "macro_f1": 0.414}, "2017": {"accuracy": 0.551, "macro_f1": 0.570}}


def score(path: Path, n_classes: int = 7) -> dict:
    d = np.load(path)
    y = d["y"]
    alpha = sum(v * alpha_from_logits(d[f"det_{k}"]) for k, v in WEIGHTS.items())
    pred = alpha.argmax(axis=1)
    present = sorted(set(y.tolist()))
    return {
        "n": int(len(y)), "accuracy": float((pred == y).mean()),
        "macro_f1_present": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "classes_present": len(present),
    }


def main() -> None:
    extract_dir = REPO_ROOT / "experiments_edl" / "extract"
    out = {}
    for year, fname in (("2018", "E_endovis2018_official_s42.npz"), ("2017", "E_endovis2017_official_s42.npz")):
        p = extract_dir / fname
        if not p.exists():
            print(f"missing {p}, skipping {year}")
            continue
        r = score(p)
        r["ce_pipeline_zero_shot"] = CE_ZERO_SHOT[year]
        out[year] = r
        print(f"EndoVis {year}: evidential zero-shot accuracy={r['accuracy']:.4f} "
              f"macro_f1({r['classes_present']} classes present)={r['macro_f1_present']:.4f}  "
              f"vs CE pipeline {CE_ZERO_SHOT[year]['accuracy']:.3f}/{CE_ZERO_SHOT[year]['macro_f1']:.3f}")
    out_path = REPO_ROOT / "docs" / "reports" / "evidential_pipeline_switch" / "endovis_con.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
