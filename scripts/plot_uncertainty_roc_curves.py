"""Plots ROC curves for the uncertainty signals compared in
scripts/analyze_uncertainty_signals.py (docs/DECISIONS.md 2026-09-16) --
each curve traces true-positive rate (real misclassifications caught)
against false-positive rate (correct predictions incorrectly flagged) as
the flagging threshold sweeps, for detecting whether a given prediction is
actually wrong.

Usage:
    python scripts/plot_uncertainty_roc_curves.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "max_softmax_confidence": "Max-softmax confidence",
    "predictive_entropy": "Predictive entropy",
    "ensemble_variance": "Ensemble variance",
    "ensemble_vote_disagreement": "Ensemble vote disagreement",
    "mc_dropout_variance_mobilenet_only": "MC Dropout variance (MobileNet only, 40% coverage)",
    "mc_dropout_variance": "MC Dropout variance (full ensemble)",
}
COLORS = {
    "max_softmax_confidence": "#1f3a5f",
    "predictive_entropy": "#5b7ea8",
    "ensemble_variance": "#2f7d52",
    "ensemble_vote_disagreement": "#c77b2e",
    "mc_dropout_variance_mobilenet_only": "#a13d4c",
    "mc_dropout_variance": "#a13d4c",
}


def _label_for(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    return name.replace("_", " ").capitalize()  # covers any dynamically-named partial-coverage MC Dropout key


def _color_for(name: str, index: int) -> str:
    fallback_palette = ["#1f3a5f", "#5b7ea8", "#2f7d52", "#c77b2e", "#a13d4c"]
    return COLORS.get(name, fallback_palette[index % len(fallback_palette)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "docs" / "reports" / "uncertainty_signals.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "figures" / "uncertainty_signals_roc.png")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    is_wrong = np.array(data["is_wrong"])

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, (name, scores) in enumerate(data["signal_scores"].items()):
        fpr, tpr, _ = roc_curve(is_wrong, np.array(scores))
        auroc = data["auroc"][name]
        ax.plot(fpr, tpr, label=f"{_label_for(name)} (AUROC={auroc:.3f})", color=_color_for(name, i), linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance (AUROC=0.500)")
    ax.set_xlabel("False positive rate (correct predictions flagged)")
    ax.set_ylabel("True positive rate (real errors caught)")
    ax.set_title(f"Detecting real misclassifications (n={data['n_total']}, {data['n_errors']} errors)\nofficial GraSP test set, Task B ensemble")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
