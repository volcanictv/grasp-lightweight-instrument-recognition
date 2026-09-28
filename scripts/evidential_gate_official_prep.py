"""Applies the fold1-calibrated evidential gate threshold, unchanged, to the
official test set, and writes the newly-flagged instance indices for SAM2
tracking (docs/DECISIONS.md 2026-09-28).

Usage:
    python scripts/evidential_gate_official_prep.py --out-dir experiments/evidential_gate_switch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path,
                    default=REPO_ROOT / "docs" / "reports" / "evidential_pipeline_switch" / "fold1_calibration.json")
    ap.add_argument("--extract", type=Path,
                    default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_official_s42.npz")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-shards", type=int, default=2)
    args = ap.parse_args()

    cal = json.loads(args.calibration.read_text())
    tau = cal["chosen_threshold"]

    d = np.load(args.extract)
    y = d["y"]
    alpha = sum(v * alpha_from_logits(d[f"det_{k}"]) for k, v in WEIGHTS.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    base_pred = mu.argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    base_acc = float((base_pred == y).mean())
    base_f1 = float(f1_score(y, base_pred, average="macro", labels=list(range(7))))
    flag = s1 >= tau
    print(f"official test evidential base: n={len(y)} accuracy={base_acc:.4f} macro_f1={base_f1:.4f} "
          f"errors={int((base_pred != y).sum())}")
    print(f"applying fold1-chosen threshold {tau:.6f} unchanged: flags {int(flag.sum())} of {len(y)} "
          f"({100 * flag.mean():.1f}%), ~{flag.sum() * 24 / 3600:.1f} GPU-hours at 24s/instance")

    idx = np.where(flag)[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(args.num_shards):
        (args.out_dir / f"official_track_shard{shard}.json").write_text(
            json.dumps({"errors": [{"index": int(i)} for i in idx[shard::args.num_shards]]}))
    (args.out_dir / "official_base.json").write_text(json.dumps({
        "n": int(len(y)), "base_accuracy": base_acc, "base_macro_f1": base_f1,
        "tau": tau, "tracked": int(flag.sum()), "share": float(flag.mean()),
    }, indent=1))
    print(f"wrote shards to {args.out_dir}")


if __name__ == "__main__":
    main()
