"""How many MC-Dropout passes per member does the vote gate actually need? Subsamples the
20 cached stochastic passes per member (k = 1, 2, 3, 5, 10, 20, so 4k total votes) and
measures, against the same fixed target (the errors of the full 80-vote ensemble):
  - AUROC of the vote-disagreement score;
  - overlap with the full-80 gate's flagged set when the same NUMBER of instances is flagged
    (the 9% gate flags 833 of 2,861), so only the ranking quality changes, not the threshold
    semantics (with fewer votes the disagreement score takes coarser values);
  - the share of the ensemble's errors inside that flagged set.
Averaged over random subsets of the passes; ties at the flagging cutoff are split in
proportion (no arbitrary tie-breaking).

Usage:
    python scripts/mc_pass_count_ablation.py --cache <mc_logits_cache_deepdropout.npz> --out results.json
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
from sklearn.metrics import roc_auc_score

from build_tracking_sample_b import plurality, vote_share


def flagged_weights(score: np.ndarray, k_flag: int) -> np.ndarray:
    """Per-instance probability of being flagged when the top k_flag by score are flagged, ties split evenly."""
    thr = np.sort(score)[::-1][k_flag - 1]
    above, tied = score > thr, score == thr
    w = above.astype(float)
    w[tied] = (k_flag - above.sum()) / tied.sum()
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    ap.add_argument("--gate", type=float, default=0.09)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members, w320 = cfg["members"], cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    c = np.load(args.cache)
    y = c["y_true"]
    n_classes = c["det_" + members[0]["label"]].shape[1]
    mc = [c["mc_" + m["label"]] for m in members]
    v_det = sum(w * vote_share(c["det_" + m["label"]], n_classes) for w, m in zip(weights, members))

    def score_and_pred(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = sum(w * vote_share(x[idx], n_classes) for w, x in zip(weights, mc))
        return np.round(1 - v.max(axis=1), 9), plurality(v, v_det)

    u_full, pred_full = score_and_pred(np.arange(20))
    err = pred_full != y
    full_flag = u_full >= args.gate
    k_flag = int(full_flag.sum())
    print(f"full 80-vote ensemble: accuracy {(~err).mean():.4f}, {int(err.sum())} errors, "
          f"gate {args.gate:.2f} flags {k_flag} of {len(y)} ({k_flag / len(y):.1%})")
    rng = np.random.default_rng(0)
    rows = {}
    print(f"{'passes/member':>13}{'votes':>7}{'AUROC':>8}{'overlap w/ full gate':>22}{'errors in flagged set':>23}")
    for k in (1, 2, 3, 5, 10, 20):
        reps = 1 if k == 20 else args.repeats
        auroc, overlap, caught = [], [], []
        for _ in range(reps):
            idx = np.arange(20) if k == 20 else rng.choice(20, size=k, replace=False)
            u, _ = score_and_pred(idx)
            auroc.append(roc_auc_score(err, u))
            w = flagged_weights(u, k_flag)
            overlap.append(float((w * full_flag).sum() / k_flag))
            caught.append(float((w * err).sum() / err.sum()))
        rows[str(k)] = {"votes": 4 * k, "auroc": float(np.mean(auroc)), "auroc_sd": float(np.std(auroc)),
                        "overlap_with_full_gate": float(np.mean(overlap)), "errors_in_flagged_set": float(np.mean(caught))}
        print(f"{k:>13}{4 * k:>7}{np.mean(auroc):>8.3f}{np.mean(overlap):>22.3f}{np.mean(caught):>23.3f}")
    if args.out:
        args.out.write_text(json.dumps({"n": int(len(y)), "errors_full": int(err.sum()), "gate": args.gate,
                                        "flagged_full": k_flag, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
