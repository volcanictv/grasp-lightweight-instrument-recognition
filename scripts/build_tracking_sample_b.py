"""Instance sets for the softmax-free tracking experiment (docs/DECISIONS.md
2026-09-19, option B). No softmax probability is used anywhere in this
pipeline; everything is a vote over stochastic forward passes.

  prediction : plurality of the pooled MC-dropout votes (4 members x 20 passes,
               member weights as in the ensemble config); a tie goes to the
               deterministic (dropout-off) votes, then to the lower class index
  uncertainty: u = 1 - top vote share (0 = unanimous)
  gate       : u >= 0.20 -> send to SAM2 tracking (the same 0.20 fixed on
               2026-09-19; not retuned)

Sets to track: every gate-flagged instance, plus a sample of gate-passed
instances (all of the passed ones that are wrong, and a random sample of the
passed ones that are right, seeded) to measure whether tracking can rescue
confident errors and how often it breaks confident-correct ones. Shards are
interleaved by sorted index, one file per GPU, in the format
evaluate_temporal_track_ensemble.py takes via --error-cases-json.

Usage:
    python scripts/build_tracking_sample_b.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def vote_share(logits: np.ndarray, n_classes: int) -> np.ndarray:
    if logits.ndim == 2:
        logits = logits[None]
    return (logits.argmax(axis=-1)[..., None] == np.arange(n_classes)).mean(axis=0)


def plurality(v_primary: np.ndarray, v_tiebreak: np.ndarray) -> np.ndarray:
    """argmax of v_primary; ties go to v_tiebreak, then to the lower class index."""
    key = np.round(v_primary, 9) * 1e3 + np.round(v_tiebreak, 9)
    return np.array([max(range(key.shape[1]), key=lambda k, i=i: (key[i, k], -k)) for i in range(len(key))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--n-passed-correct", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "reports" / "tracking_softmax_free")
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
    pred = plurality(v_mc, v_det)
    u = np.round(1 - v_mc.max(axis=1), 9)
    wrong = pred != y_true

    flagged = u >= args.threshold
    passed_wrong = np.where(~flagged & wrong)[0]
    passed_right = np.where(~flagged & ~wrong)[0]
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(passed_right, size=args.n_passed_correct, replace=False))
    union = np.sort(np.concatenate([np.where(flagged)[0], passed_wrong, sample]))

    sets = {
        "n_total": int(len(y_true)), "threshold": args.threshold, "seed": args.seed,
        "ensemble_errors": int(wrong.sum()),
        "flagged": np.where(flagged)[0].tolist(), "flagged_errors": int((flagged & wrong).sum()),
        "passed_wrong": passed_wrong.tolist(), "passed_right_total": int(len(passed_right)),
        "passed_right_sample": sample.tolist(), "union": union.tolist(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sets.json").write_text(json.dumps(sets))
    for shard in (0, 1):
        (args.out_dir / f"track_indices_shard{shard}.json").write_text(
            json.dumps({"errors": [{"index": int(i)} for i in union[shard::2]]}))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in sets.items()}, indent=1))
    print(f"vote-plurality accuracy {(1 - wrong.mean()):.4f}; instances per shard {len(union[0::2])} / {len(union[1::2])}")


if __name__ == "__main__":
    main()
