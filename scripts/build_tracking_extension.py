"""Instances that still need SAM2 tracking to test lower gate thresholds with
real outcomes, not estimates (docs/DECISIONS.md 2026-09-20).

The softmax-free run already tracked every instance with disagreement u >= 0.20,
all gate-passed errors, and a random sample of gate-passed correct ones. To
evaluate any threshold t in [floor, 0.20) end-to-end, every instance with
u >= floor must have tracking data; this lists the ones that do not yet.

The floor (0.03) was fixed before any extension result exists, by a cost rule
and not by looking at accuracy: it is the lowest threshold that still roughly
doubles the tracking budget (39% of instances flagged vs 18%) and covers 93% of
the errors. Disagreement moves in steps of 0.01, so every threshold from 0.03 to
0.20 can then be scored with real outcomes.

Usage:
    python scripts/build_tracking_extension.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from build_tracking_sample_b import plurality, vote_share

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--prior-sets", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_softmax_free" / "sets.json")
    parser.add_argument("--floor", type=float, default=0.03)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_gate_sweep")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y_true = cache["y_true"]
    n_classes = cache["det_" + members[0]["label"]].shape[1]
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    wrong = plurality(v_mc, v_det) != y_true
    u = np.round(1 - v_mc.max(axis=1), 9)

    already = set(json.loads(args.prior_sets.read_text())["union"])
    need = np.array([i for i in np.where(u >= args.floor)[0] if i not in already])
    grid = [0.20, 0.15, 0.10, 0.05, 0.03]
    sets = {
        "floor": args.floor, "grid": grid, "n_total": int(len(y_true)), "extension": need.tolist(),
        "flagged_by_threshold": {str(t): int((u >= t).sum()) for t in grid},
        "errors_covered_by_threshold": {str(t): int(((u >= t) & wrong).sum()) for t in grid},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sets.json").write_text(json.dumps(sets))
    for shard in (0, 1):
        (args.out_dir / f"track_indices_shard{shard}.json").write_text(
            json.dumps({"errors": [{"index": int(i)} for i in need[shard::2]]}))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in sets.items()}, indent=1))
    print(f"to track: {len(need)} ({len(need[0::2])} / {len(need[1::2])} per shard), already tracked with u >= floor: "
          f"{int(((u >= args.floor)).sum()) - len(need)}")


if __name__ == "__main__":
    main()
