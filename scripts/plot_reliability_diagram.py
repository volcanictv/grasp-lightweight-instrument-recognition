"""Reliability diagram (accuracy vs. confidence per bin) for the
uncalibrated vs. temperature-scaled ensemble, from
scripts/analyze_calibration.py's output. Perfect calibration sits on the
diagonal; bars below the diagonal mean overconfidence in that bin (the
usual failure mode for deep nets), bars above mean underconfidence.

Usage:
    python scripts/plot_reliability_diagram.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def plot_panel(ax, bins: list[dict], title: str, ece: float, accuracy: float) -> None:
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["accuracy"] if b["accuracy"] is not None else 0.0 for b in bins]
    width = 1.0 / len(bins)

    ax.bar(centers, accs, width=width * 0.9, color="#5b7ea8", edgecolor="#1f3a5f", label="Accuracy in bin")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={ece:.4f}, accuracy={accuracy:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)


def main() -> None:
    data = json.loads((REPO_ROOT / "docs" / "reports" / "calibration_analysis.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    plot_panel(axes[0], data["uncalibrated"]["bins"], "Uncalibrated (T=1)", data["uncalibrated"]["ece"], data["uncalibrated"]["accuracy"])
    plot_panel(axes[1], data["temperature_scaled"]["bins"], "Temperature-scaled", data["temperature_scaled"]["ece"], data["temperature_scaled"]["accuracy"])
    fig.suptitle(f"Reliability diagram, eval set n={data['n_eval']} (cases {', '.join(data['eval_cases'])})")
    fig.tight_layout()

    out_path = REPO_ROOT / "docs" / "reports" / "figures" / "reliability_diagram.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
