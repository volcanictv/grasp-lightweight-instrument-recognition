"""Per-class numbers and confusion matrix of the three-member evidential ensemble alone on the official
test (no tracking), for the PI report. Run where the extracted logits live (titanxp).

Usage:
    python scripts/export_base_official.py --out docs/reports/causal_realtime/base_official.json
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
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import evidential_three_member_yolo as t

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", type=Path, default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_official_s42.npz")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    y, pred, s1 = t.base_scores(args.extract)
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(7), zero_division=0)
    result = {
        "ensemble": "three-member evidential (resnet50_320, baseline, letterbox_crop; weights 0.40/0.30/0.30), seed 42, one pass per member",
        "n": int(len(y)), "accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=range(7))),
        "per_class": {c: {"n": int(n[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, c in enumerate(CLASSES)},
        "confusion_counts": confusion_matrix(y, pred, labels=range(7)).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(result["accuracy"], result["macro_f1"])


if __name__ == "__main__":
    main()
