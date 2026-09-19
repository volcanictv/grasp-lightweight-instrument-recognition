"""Defines the two tracking gates for the end-to-end comparison and writes the
instance lists SAM2 tracking has to run on (docs/DECISIONS.md 2026-09-19).

Ensemble: the deep-dropout retrained one (region_ensemble_deepdropout.yaml),
gates computed from the cached MC-dropout logits of the annotated center frame
-- the same crop the confidence gate has always seen.

  confidence gate : ensemble max-softmax < 0.80 (the incumbent, unchanged)
  MC Dropout gate : vote disagreement >= t, where t is fixed by a rule that
                    uses no labels and no tracking outcome: the threshold whose
                    flagged count is closest to the confidence gate's (ties go
                    to the higher threshold). Equal review/tracking budget is
                    the fairest comparison of the two signals.

Tracking an instance does not depend on which gate selected it, so both gates
are scored from one run over the union of what they flag. The union is split
into two interleaved shard files (one per GPU) in the format
evaluate_temporal_track_ensemble.py takes via --error-cases-json.

Usage:
    python scripts/compute_tracking_gates.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml

from complementarity_analysis import build_signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--conf-threshold", type=float, default=0.80)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_mcgate")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()

    cache = np.load(args.logits_cache)
    y_true = cache["y_true"]
    det = [cache[f"det_{m['label']}"] for m in members]
    mc = [cache[f"mc_{m['label']}"] for m in members]
    p_det = sum(w * torch.softmax(torch.from_numpy(d), dim=1).numpy() for w, d in zip(weights, det))
    y_pred = p_det.argmax(axis=1)
    sig = build_signals(det, mc, weights, y_pred)

    conf_flag = p_det.max(axis=1) < args.conf_threshold
    n_conf = int(conf_flag.sum())
    dis = sig["vote_dis"]
    levels = np.unique(dis[dis > 0])
    counts = np.array([(dis >= t).sum() for t in levels])
    best = np.argmin(np.abs(counts - n_conf) - 1e-9 * levels)  # ties -> higher threshold
    t = float(levels[best])
    mc_flag = dis >= t

    union = np.where(conf_flag | mc_flag)[0]
    result = {
        "ensemble_config": args.ensemble_config.name, "n_total": int(len(y_true)),
        "confidence_gate": {"threshold": args.conf_threshold, "n_flagged": n_conf},
        "mc_dropout_gate": {"threshold": t, "n_flagged": int(mc_flag.sum()),
                            "nearest_alternatives": [
                                {"threshold": float(levels[i]), "n_flagged": int(counts[i])}
                                for i in range(max(0, best - 2), min(len(levels), best + 3))]},
        "overlap": int((conf_flag & mc_flag).sum()), "union": int(len(union)),
        "conf_flagged_errors": int((conf_flag & (y_pred != y_true)).sum()),
        "mc_flagged_errors": int((mc_flag & (y_pred != y_true)).sum()),
        "n_errors": int((y_pred != y_true).sum()),
        "conf_flagged_indices": np.where(conf_flag)[0].tolist(),
        "mc_flagged_indices": np.where(mc_flag)[0].tolist(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gates.json").write_text(json.dumps(result, indent=1))
    for shard in (0, 1):
        idx = [int(i) for i in union[shard::2]]
        (args.out_dir / f"track_indices_shard{shard}.json").write_text(json.dumps({"errors": [{"index": i} for i in idx]}))
    summary = {k: v for k, v in result.items() if not k.endswith("_indices")}
    print(json.dumps(summary, indent=1))
    print(f"union split into shards of {len(union[0::2])} and {len(union[1::2])}")


if __name__ == "__main__":
    main()
