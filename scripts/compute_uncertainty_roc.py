"""ROC curve points for the report's uncertainty figure: how well each signal
separates wrong from correct predictions on the official test set, retrained
ensemble (docs/DECISIONS.md 2026-09-20).

  MC vote disagreement   : share of stochastic votes against the winning class,
                           scored against the errors of the vote-plurality prediction
  between-member votes   : the same with one dropout-off vote per member
  softmax confidence     : 1 - top averaged probability, scored against the errors
                           of its own (averaged-softmax) prediction; the baseline

Writes docs/reports/final_pipeline/roc_curves.json.

Usage:
    python scripts/compute_uncertainty_roc.py
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
from sklearn.metrics import roc_auc_score, roc_curve

from build_tracking_sample_b import plurality, vote_share


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "final_pipeline" / "roc_curves.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w320 = cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y = cache["y_true"]
    n_classes = cache["det_" + members[0]["label"]].shape[1]
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    wrong_vote = plurality(v_mc, v_det) != y
    p_soft = sum(w * torch.softmax(torch.from_numpy(cache[f"det_{m['label']}"]), dim=1).numpy() for w, m in zip(weights, members))
    wrong_soft = p_soft.argmax(axis=1) != y

    signals = {
        "MC Dropout vote disagreement": (np.round(1 - v_mc.max(axis=1), 9), wrong_vote),
        "Between-member disagreement (dropout off)": (np.round(1 - v_det.max(axis=1), 9), wrong_vote),
        "Softmax confidence (baseline)": (1 - p_soft.max(axis=1), wrong_soft),
    }
    out = {"n_total": int(len(y)), "curves": {}}
    for name, (score, wrong) in signals.items():
        fpr, tpr, _ = roc_curve(wrong.astype(int), score)
        out["curves"][name] = {"auroc": float(roc_auc_score(wrong.astype(int), score)), "n_errors": int(wrong.sum()),
                               "fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}
        print(f"{name}: AUROC {out['curves'][name]['auroc']:.4f}, {len(fpr)} points, {int(wrong.sum())} errors")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
