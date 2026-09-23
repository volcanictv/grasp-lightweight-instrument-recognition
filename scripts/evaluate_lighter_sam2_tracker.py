"""Final-pipeline accuracy on the official test set using a lighter SAM2 variant
in place of sam2.1_hiera_large, everything else unchanged (docs/DECISIONS.md
2026-09-22). Reads the frame logits produced by
evaluate_temporal_track_ensemble.py --sam2-checkpoint <tiny|small> and combines
them the same way the shipped pipeline does (V2, uncertainty-weighted vote).

Usage:
    python scripts/evaluate_lighter_sam2_tracker.py --tracker tiny \
        --frames docs/reports/lighter_tracker/frames_tiny.npz \
        --out docs/reports/lighter_tracker/final_pipeline_tiny.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import yaml
from sklearn.metrics import f1_score

from build_tracking_sample_b import plurality, vote_share
from evaluate_softmax_free_combine import RULES, track_predictions

CLASS_NAMES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
               "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    ap.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    ap.add_argument("--threshold", type=float, default=0.09)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members, w320 = cfg["members"], cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y = cache["y_true"]
    n_classes = len(CLASS_NAMES)
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    base = plurality(v_mc, v_det)
    u = np.round(1 - v_mc.max(axis=1), 9)
    flag = u >= args.threshold

    data = np.load(args.frames)
    tracked = {}
    for key in data.files:
        if key.startswith("det_"):
            i = int(key[4:])
            tracked[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    missing = [i for i in np.where(flag)[0] if i not in tracked]
    pred = base.copy()
    for i in np.where(flag)[0]:
        if i in tracked:
            pred[i] = tracked[i][RULES[1]]

    out = {
        "tracker": args.tracker, "n": int(len(y)), "tracked": int(flag.sum()), "missing_tracking_data": len(missing),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=list(range(n_classes)), average="macro")),
        "fixed": int(((base != y) & (pred == y) & flag).sum()),
        "broken": int(((base == y) & (pred != y) & flag).sum()),
        "errors_left": int((pred != y).sum()),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
