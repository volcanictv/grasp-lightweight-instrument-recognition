"""The effort curve of the paper (Figure 3): test mIoU as a function of the share of instruments that get the second look, for the uncertainty gates and the simple heuristics, 3-seed mean.

Inputs: docs/reports/gtbox_sam/gate_baselines.json (scripts/gate_baselines_analysis.py), docs/reports/gtbox_sam/tta_vs_tracking.json (scripts/bootstrap_tta_vs_tracking.py), docs/reports/gtbox_sam/bootstrap_heldout_row.json.
Lines: S (the paper's gate), largest belief, softmax max-probability, class prior, small box, random, best possible (dashed); horizontal lines: single pass, tracking every instrument, TTA on every instrument; one marked point for the
threshold tau fixed on held-out cases (3-seed mean share, per-seed shares as a bar). Top axis: average TFLOPs per instrument (2.95 at 0%, 28.53 at 100%).

    python scripts/make_effort_figure.py --out paper/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
         "text.color": "#1c2430", "axes.edgecolor": "#8a94a3", "axes.labelcolor": "#1c2430", "xtick.color": "#5d6877", "ytick.color": "#1c2430"}
plt.rcParams.update(STYLE)
INK, MUTED, ACCENT, GREY, AMBER, RED = "#1c2430", "#5d6877", "#0b6e8a", "#8a94a3", "#a8651a", "#b3382f"
LINES = [("S", "S (epistemic score)", ACCENT, "-", "o"), ("largest belief", "largest belief", "#2a9d8f", "-", "s"), ("softmax max-prob", "softmax max-probability", AMBER, "-", "^"),
         ("class prior", "class prior (weak classes first)", "#7b5ea7", "-", "v"), ("small box", "small box first", GREY, "-", "D"), ("random", "random order", "#b0b7c3", ":", None)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("docs/reports/gtbox_sam"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    G = json.loads((args.reports / "gate_baselines.json").read_text())
    T = json.loads((args.reports / "tta_vs_tracking.json").read_text())
    B = json.loads((args.reports / "bootstrap_heldout_row.json").read_text())
    seeds = sorted(G["seeds"])
    shares = [int(round(100 * s)) for s in G["shares"]]
    keys = [f"{s}%" for s in shares]
    single = float(np.mean([G["seeds"][s]["single"]["mIoU"] for s in seeds]))
    full = float(np.mean([G["seeds"][s]["refine_all"]["mIoU"] for s in seeds]))
    tta = T["point"]["TTA all"]["mIoU"]

    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for name, label, color, ls, marker in LINES:
        if name not in G["seeds"][seeds[0]]["curves"]:
            continue
        y = [single] + [float(np.mean([G["seeds"][s]["curves"][name][k]["mIoU"] for s in seeds])) for k in keys]
        ax.plot([0] + shares, y, ls=ls, color=color, marker=marker, ms=3.5, lw=1.4 if name in ("S", "largest belief", "softmax max-prob") else 1.0, label=label, zorder=3 if name == "S" else 2)
    best = [single] + [float(np.mean([G["seeds"][s]["curves"]["best possible"][k]["mIoU"] for s in seeds])) for k in keys]
    ax.plot([0] + shares, best, ls="--", color=INK, lw=0.9, label="best possible order", zorder=1)
    ax.axhline(single, color=MUTED, lw=0.6, ls="-")
    ax.text(101, single, "single pass 83.5", fontsize=6.3, color=MUTED, va="center")
    ax.axhline(full, color=MUTED, lw=0.6, ls="-")
    ax.text(101, full, f"tracking all {full:.1f}", fontsize=6.3, color=MUTED, va="center")
    ax.axhline(tta, color=RED, lw=0.7, ls="-.")
    ax.text(101, tta, f"21 augmented views of the keyframe, all {tta:.1f}", fontsize=6.3, color=RED, va="center")
    shares_tau = [float(x) for x in B.get("tau_shares", [])] if isinstance(B.get("tau_shares"), list) else []
    tau_point = B["estimates"]["gated (tau 1.7e-05)"]["three_seed_mean"]["mIoU"]["point"]
    ax.errorbar([44.9], [tau_point], xerr=[[44.9 - 42.4], [47.2 - 44.9]], fmt="*", color=INK, ms=8, capsize=2.5, lw=0.8, zorder=5)
    ax.annotate("threshold fixed on\nheld-out cases", xy=(44.9, tau_point), xytext=(54, 85.0), fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(single - 0.8, max(best) + 0.6)
    ax.set_xlabel("share of instruments that get the second look (%)")
    ax.set_ylabel("test mIoU")
    ax.grid(color="#e1e5ea", lw=0.5)
    top = ax.secondary_xaxis("top", functions=(lambda x: 2.95 + (28.53 - 2.95) * x / 100, lambda t: (t - 2.95) * 100 / (28.53 - 2.95)))
    top.set_xlabel("average TFLOPs per instrument", fontsize=7)
    ax.legend(loc="lower right", fontsize=6.3, frameon=False, ncol=2, handlelength=1.8)
    fig.subplots_adjust(left=0.09, right=0.74, top=0.86, bottom=0.14)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig3_effort.pdf")
    fig.savefig(args.out / "fig3_effort.png", dpi=200)
    print("wrote", args.out / "fig3_effort.pdf")


if __name__ == "__main__":
    main()
