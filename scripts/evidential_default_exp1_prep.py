"""Experiment 1 prep (docs/DECISIONS.md 2026-09-29): rank the official test set by the evidential
epistemic score S1, take the top --budget, keep the 539 already tracked on 2026-09-28, and write
the tracking shards for the rest.

Usage:
    python scripts/evidential_default_exp1_prep.py --out-dir experiments/evidential_default --budget 833
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", type=Path, default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_official_s42.npz")
    ap.add_argument("--gate-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_gate_switch")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=1.7e-05, help="fold1-chosen threshold of 2026-09-28")
    ap.add_argument("--budget", type=int, default=833)
    ap.add_argument("--num-shards", type=int, default=2)
    args = ap.parse_args()

    d = np.load(args.extract)
    alpha = sum(v * alpha_from_logits(d[f"det_{k}"]) for k, v in WEIGHTS.items())
    s1 = variance_scores(alpha)["epistemic"]
    order = np.argsort(-s1, kind="stable")
    top = order[: args.budget]

    tau = args.tau
    flagged539 = set(np.where(s1 >= tau)[0].tolist())
    inside = flagged539 <= set(top.tolist())
    print(f"tau flagged {len(flagged539)}; all inside top {args.budget} by S1: {inside}")
    print(f"S1 at rank {args.budget}: {s1[order[args.budget - 1]]:.3e}, tau {tau:.3e}")

    already = set()
    for shard in (0, 1):
        fd = np.load(args.gate_dir / f"official_frames_shard{shard}.npz")
        already |= {int(k[4:]) for k in fd.files if k.startswith("det_")}
    print(f"already tracked on 2026-09-28: {len(already)} (equal to the 539: {already == flagged539})")

    missing = sorted(int(i) for i in top if int(i) not in already)
    print(f"to track now: {len(missing)}; est {len(missing) * 21 / 3600 / args.num_shards:.2f} h on {args.num_shards} GPUs")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(args.num_shards):
        (args.out_dir / f"exp1_track_shard{shard}.json").write_text(
            json.dumps({"errors": [{"index": i} for i in missing[shard::args.num_shards]]}))
    (args.out_dir / "exp1_ranking.json").write_text(json.dumps({
        "budget": args.budget, "top_indices": [int(i) for i in top], "already": sorted(already),
        "flagged_inside_top": inside, "missing": missing}))


if __name__ == "__main__":
    main()
