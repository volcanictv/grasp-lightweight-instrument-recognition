"""Vote-disagreement uncertainty for the Task B ensemble, and its use as the
triage gate (docs/DECISIONS.md 2026-09-18).

Every (member, MC-dropout pass) casts a vote for a class; a member's votes
carry that member's ensemble weight, split evenly across its passes. With V
the resulting weighted vote distribution over classes and y the ensemble's
final prediction:

  primary   mc_vote_disagreement   1 - V[y]   share of votes against the prediction
  variants  mc_vote_variation_ratio 1 - max(V)  prediction-agnostic
            mc_vote_entropy         H(V)
            det_vote_disagreement   same as primary but one deterministic
                                    (dropout-off) vote per member, no MC
            pooled_vote_disagreement equal mix of the MC and deterministic vote
                                    distributions

No parameters are fitted; the primary definition is fixed here. It was
chosen after seeing 2026-09-17/18 test-set AUROCs for the same quantity, so
the official test number is not an untouched estimate -- flagged, not hidden.

Also reports gate operating points (fraction flagged, share of real errors
caught, precision) for fixed thresholds on the disagreement score, next to
the incumbent max-softmax confidence < 0.80 gate.

Reads the logits cached by analyze_mc_logit_signals.py; CPU only.

Usage:
    python scripts/disagreement_method.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]


def vote_distribution(logits: np.ndarray, weight: float, n_classes: int) -> np.ndarray:
    """(S, N, C) or (N, C) logits -> (N, C) weighted share of argmax votes."""
    if logits.ndim == 2:
        logits = logits[None]
    votes = (logits.argmax(axis=2)[..., None] == np.arange(n_classes)).mean(axis=0)
    return weight * votes


def operating_point(is_wrong: np.ndarray, flagged: np.ndarray) -> dict:
    n_flag = int(flagged.sum())
    caught = int((flagged & (is_wrong == 1)).sum())
    return {
        "flagged_frac": n_flag / len(is_wrong),
        "errors_caught": caught / int(is_wrong.sum()),
        "precision": caught / n_flag if n_flag else float("nan"),
    }


def bootstrap_ci(is_wrong: np.ndarray, score: np.ndarray, rng: np.random.Generator, n_boot: int = 1000):
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(is_wrong), len(is_wrong))
        if is_wrong[idx].min() != is_wrong[idx].max():
            aucs.append(roc_auc_score(is_wrong[idx], score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_mcdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "disagreement_method.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w_320 = cfg["weight_resnet50_320"]
    weights = [w_320 if m["label"] == "resnet50_320" else (1 - w_320) / (len(members) - 1) for m in members]

    cache = np.load(args.logits_cache)
    y_true = cache["y_true"]
    det = [cache[f"det_{m['label']}"] for m in members]
    mc = [cache[f"mc_{m['label']}"] for m in members]
    n_classes = det[0].shape[1]

    probs = sum(w * torch.softmax(torch.from_numpy(d), dim=1).numpy() for w, d in zip(weights, det))
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    is_wrong = (y_pred != y_true).astype(int)
    rows = np.arange(len(y_true))

    v_mc = sum(vote_distribution(x, w, n_classes) for w, x in zip(weights, mc))
    v_det = sum(vote_distribution(d, w, n_classes) for w, d in zip(weights, det))
    v_pool = 0.5 * (v_mc + v_det)
    entropy = -(v_mc * np.log(v_mc + 1e-12)).sum(axis=1)

    scores = {
        "mc_vote_disagreement": 1 - v_mc[rows, y_pred],
        "mc_vote_variation_ratio": 1 - v_mc.max(axis=1),
        "mc_vote_entropy": entropy,
        "det_vote_disagreement": 1 - v_det[rows, y_pred],
        "pooled_vote_disagreement": 1 - v_pool[rows, y_pred],
    }

    rng = np.random.default_rng(42)
    results = {"n_total": int(len(y_true)), "n_errors": int(is_wrong.sum()), "primary": "mc_vote_disagreement",
               "auroc": {}, "auroc_ci95": {}, "gate": {}}
    print(f"n={len(y_true)}, errors={is_wrong.sum()} (accuracy {1 - is_wrong.mean():.4f})\n")
    print("AUROC vs incorrect=1:")
    for name, s in scores.items():
        auc = float(roc_auc_score(is_wrong, s))
        lo, hi = bootstrap_ci(is_wrong, s, rng)
        results["auroc"][name], results["auroc_ci95"][name] = auc, [lo, hi]
        print(f"  {name:<28} {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
    ref = float(roc_auc_score(is_wrong, -confidence))
    results["auroc"]["max_softmax_confidence (reference)"] = ref
    print(f"  {'max_softmax_confidence (ref)':<28} {ref:.4f}")

    print(f"\nGate operating points (score = {results['primary']}, flag if score >= t):")
    print(f"  {'gate':<28} {'flagged':>8} {'errors caught':>14} {'precision':>10}")
    gates = {f"disagreement >= {t:.2f}": scores["mc_vote_disagreement"] >= t for t in THRESHOLDS}
    gates["confidence < 0.80 (incumbent)"] = confidence < 0.80
    for label, flagged in gates.items():
        op = operating_point(is_wrong, flagged)
        results["gate"][label] = op
        print(f"  {label:<28} {op['flagged_frac']:>8.1%} {op['errors_caught']:>14.1%} {op['precision']:>10.1%}")
    n_levels = len(np.unique(scores["mc_vote_disagreement"]))
    results["primary_distinct_score_levels"] = int(n_levels)
    print(f"\nprimary score takes {n_levels} distinct values (coarse: many ties, so thresholds are steppy)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
