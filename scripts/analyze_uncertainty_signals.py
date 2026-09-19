"""Compares candidate uncertainty signals for the Task B ensemble as
predictors of real misclassification, since reviewers reasonably object to
using raw softmax confidence as "uncertainty" -- a softmax output is
trained to be peaked, not calibrated, and says nothing about epistemic
(model) uncertainty specifically.

Tested here, all against the same real-error-detection task (does the
signal separate the 188 actual official-test misclassifications from the
2,673 correct predictions), scored by AUROC -- the standard metric for
"how good is this uncertainty score at flagging wrong predictions",
independent of any particular threshold choice:

1. Max-softmax confidence (the incumbent, already used for the tracking
   triage gate, docs/DECISIONS.md 2026-09-16).
2. Predictive entropy of the combined ensemble softmax.
3. Ensemble vote disagreement: fraction of the 4 independently-trained
   members whose own argmax differs from the final ensemble prediction --
   free, already-available epistemic-uncertainty proxy (Deep Ensembles,
   Lakshminarayanan et al. 2017), needs no retraining.
4. Ensemble variance: variance across the 4 members' assigned probability
   for the ensemble's chosen class -- a continuous version of (3).
5. MC Dropout variance: variance, across stochastic forward passes with
   Dropout left in train mode (Gal & Ghahramani 2016), of each
   MC-dropout-capable member's assigned probability for the final
   ensemble's chosen class -- the continuous, probability-space MC
   Dropout signal.
6. MC top-class disagreement: fraction of those same stochastic forward
   passes whose own argmax differs from the final ensemble prediction --
   the discrete, argmax-space analogue of (5), same relationship (3) has
   to (4) but for MC samples instead of independently-trained members.
   Reuses the same forward passes as (5), no extra compute.

With partial ensemble coverage (docs/DECISIONS.md 2026-09-16: ResNet-50
had zero Dropout layers, so MC Dropout was silently deterministic for
60% of the ensemble's decision weight) or full coverage (2026-09-17:
both ResNet-50 members retrained with dropout added), whichever
ensemble config is passed determines which members have real MC Dropout
signal to contribute; a member with none is excluded and the coverage
gap reported plainly.

Usage:
    python scripts/analyze_uncertainty_signals.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def enable_mc_dropout(model: nn.Module) -> None:
    """Puts only Dropout submodules into train mode (stochastic), leaving
    everything else (BatchNorm, etc.) in eval mode -- the standard MC
    Dropout inference recipe (Gal & Ghahramani 2016). Dropout2d is not a
    subclass of nn.Dropout, so it is listed explicitly."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "uncertainty_signals.json")
    parser.add_argument("--mc-samples", type=int, default=20)
    args = parser.parse_args()

    ensemble_config_path = args.ensemble_config
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split = "test"
    mc_samples = args.mc_samples
    out_path = args.out

    ensemble_config = yaml.safe_load(ensemble_config_path.read_text())
    members_cfg = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]
    w_rest = (1 - weight_320) / (len(members_cfg) - 1)

    class_names = None
    y_true = None
    per_member_probs = []  # list of (N, C) arrays, one per member
    per_member_dropout_count = []

    for m in members_cfg:
        ds = GraspRegionDataset(
            data_root, split, transform=build_transforms(m["image_size"], train=False), letterbox=m["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])

        model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()
        n_dropout = sum(1 for mod in model.modules() if isinstance(mod, nn.Dropout))
        per_member_dropout_count.append(n_dropout)
        print(f"{m['label']}: {n_dropout} dropout layer(s)")

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        per_member_probs.append(np.concatenate(probs))

    per_member_probs = np.stack(per_member_probs)  # (M, N, C)
    weights = np.array([weight_320 if m["label"] == "resnet50_320" else w_rest for m in members_cfg])
    combined = (weights[:, None, None] * per_member_probs).sum(axis=0)  # (N, C)
    y_pred = combined.argmax(axis=1)
    is_wrong = (y_pred != y_true).astype(int)
    print(f"\nbaseline: accuracy={1 - is_wrong.mean():.4f}, errors={is_wrong.sum()}/{len(y_true)}")

    signals = {}

    signals["max_softmax_confidence"] = combined.max(axis=1)  # higher = more confident -> use negative for AUROC

    eps = 1e-12
    signals["predictive_entropy"] = -(combined * np.log(combined + eps)).sum(axis=1)  # higher = more uncertain

    per_member_argmax = per_member_probs.argmax(axis=2)  # (M, N)
    signals["ensemble_vote_disagreement"] = (per_member_argmax != y_pred[None, :]).mean(axis=0)  # higher = more disagreement

    pred_class_probs = np.take_along_axis(per_member_probs, y_pred[None, :, None], axis=2)[:, :, 0]  # (M, N)
    signals["ensemble_variance"] = pred_class_probs.var(axis=0)  # higher = more disagreement on the chosen class

    # MC Dropout: only members with a real Dropout layer contribute variance;
    # a member with none is silently deterministic and excluded from the weighted sum.
    mc_variance = np.zeros(len(y_true))
    mc_disagreement = np.zeros(len(y_true))
    mc_members = [m for m, n in zip(members_cfg, per_member_dropout_count) if n > 0]
    mc_coverage = sum(weights[members_cfg.index(m)] for m in mc_members)
    if mc_members:
        print(f"\nrunning MC Dropout ({mc_samples} samples) on: {[m['label'] for m in mc_members]} "
              f"(covers {mc_coverage*100:.0f}% of ensemble weight)")
        mc_weight_total = sum(weights[i] for i, m in enumerate(members_cfg) if m in mc_members)
        for m in mc_members:
            ds = GraspRegionDataset(
                data_root, split, transform=build_transforms(m["image_size"], train=False), letterbox=m["letterbox"],
            )
            model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
            model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
            enable_mc_dropout(model)

            loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            mc_samples_probs = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    probs = []
                    for images, _labels in loader:
                        logits = model(images.to(device))
                        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                    mc_samples_probs.append(np.concatenate(probs))
            mc_samples_probs = np.stack(mc_samples_probs)  # (S, N, C)
            member_pred_class_probs = np.take_along_axis(mc_samples_probs, y_pred[None, :, None], axis=2)[:, :, 0]
            member_weight = weights[members_cfg.index(m)]
            mc_variance += (member_weight / mc_weight_total) * member_pred_class_probs.var(axis=0)

            pred_t = mc_samples_probs.argmax(axis=2)  # (S, N) -- argmax per stochastic pass
            member_disagreement = (pred_t != y_pred[None, :]).mean(axis=0)  # fraction of passes disagreeing with final prediction
            mc_disagreement += (member_weight / mc_weight_total) * member_disagreement
    coverage_suffix = "" if mc_coverage > 0.99 else f"_{int(round(mc_coverage*100))}pct_coverage"
    signals[f"mc_dropout_variance{coverage_suffix}"] = mc_variance
    signals[f"mc_top_class_disagreement{coverage_suffix}"] = mc_disagreement

    results = {"n_total": int(len(y_true)), "n_errors": int(is_wrong.sum()), "baseline_accuracy": float(1 - is_wrong.mean()),
               "mc_dropout_coverage": float(mc_coverage),
               "dropout_layer_counts": {m["label"]: n for m, n in zip(members_cfg, per_member_dropout_count)},
               "auroc": {}, "is_wrong": is_wrong.tolist(), "signal_scores": {}}
    print("\nAUROC for detecting real misclassifications (0.5 = no better than chance, 1.0 = perfect):")
    for name, values in signals.items():
        # for confidence, LOW confidence should predict error -> flip sign so higher score = more likely wrong, for every signal
        score = -values if name == "max_softmax_confidence" else values
        auroc = float(roc_auc_score(is_wrong, score))
        results["auroc"][name] = auroc
        results["signal_scores"][name] = score.tolist()  # already sign-flipped so higher = more-likely-wrong for every signal, for direct ROC plotting
        print(f"  {name:<38} AUROC={auroc:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
