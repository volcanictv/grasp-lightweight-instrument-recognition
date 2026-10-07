"""The two main figures of the blueprint draft: the effort curve (errors caught against the share of instruments reviewed, and accuracy gain against the share refined) and the per-class
IoU before and after refinement. Reads docs/reports/evidential/calibration_baselines.json, docs/reports/realtime/gate_budget_curve.json and docs/reports/gtbox_sam/final_3seed.json.

Usage (laptop, repo root):
    python scripts/make_effort_figures.py --out paper/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 7.5, "pdf.fonttype": 42,
                     "axes.edgecolor": "#8a96a3", "axes.linewidth": 0.7, "xtick.color": "#4b5865", "ytick.color": "#4b5865"})
INK, GREY, ACCENT, RED = "#1d2733", "#8a96a3", "#0f7b8a", "#b6372e"
COL = {"Evidential S1 (ours)": ACCENT, "Softmax confidence": "#55687c", "MC dropout, vote": "#b3741b", "MC dropout, mean softmax": "#d9a441", "Deep ensemble, entropy (3 seeds)": "#a23a5c"}
NAMES = ["Bipolar", "Prograsp", "Needle driver", "Scissors", "Suction", "Clip applier", "Grasper"]


def effort(out: Path) -> None:
    R = REPO_ROOT / "docs" / "reports"
    cal = json.loads((R / "evidential" / "calibration_baselines.json").read_text())
    gate = json.loads((R / "realtime" / "gate_budget_curve.json").read_text())["offline SAM2-large (SAM2 + SAM3 masks)"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={"wspace": 0.30})

    f = np.array(cal["review_curves"]["fractions"]) * 100
    a.plot([0, 100], [0, 100], color=GREY, lw=0.9, ls=":", label="random order")
    styles = {"Evidential S1 (ours)": dict(lw=2.0), "Softmax confidence": dict(lw=1.3), "MC dropout, vote": dict(lw=1.3), "MC dropout, mean softmax": dict(lw=1.1, ls=(0, (4, 2))),
              "Deep ensemble, entropy (3 seeds)": dict(lw=1.1, ls=(0, (4, 2)))}
    for name, st in styles.items():
        a.plot(f, 100 * np.array(cal["review_curves"]["curves"][name]), color=COL[name], label=name, **st)
    test = 100 * np.array(cal["test_evidential"]["review_curve"])
    a.plot(f, test, color=ACCENT, lw=1.3, ls=(0, (1, 1.5)), label="Evidential S1, test cases")
    i20 = 20
    ev20, te20 = 100 * cal["review_curves"]["curves"]["Evidential S1 (ours)"][i20], test[i20]
    a.axvline(20, color=RED, lw=0.8, ls=":")
    a.plot([20], [ev20], "o", color=ACCENT, ms=4.2)
    a.annotate(f"20% reviewed:\n{ev20:.0f}% of errors (held-out fold)\n{te20:.0f}% (test cases)", (20, ev20), xytext=(29, 57), fontsize=6.7, color=ACCENT,
               arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6))
    a.set_xlim(0, 60)
    a.set_ylim(0, 100)
    a.set_xlabel("instruments reviewed (%), highest uncertainty first", fontsize=7.4)
    a.set_ylabel("errors found (%)")
    a.grid(color="#e1e5ea", lw=0.5)
    a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=5.9, loc="lower right", handlelength=2.2, labelspacing=0.3)
    a.set_title("(a) Review: errors found", fontsize=8.2, loc="left")

    k, n = np.array(gate["k"]), gate["n"]
    acc, base = np.array(gate["accuracy_mean"]), gate["base_accuracy"]
    orc = np.array(gate["oracle_mean"])
    x = 100 * k / n
    b.plot(x, 100 * (acc - base), color=ACCENT, lw=2.0, label="evidential gate")
    b.plot(x, 100 * (orc - base), color=GREY, lw=0.9, ls=(0, (3, 2)), label="best possible gate")
    i29 = int(np.argmin(np.abs(k - 833)))
    best = 100 * (acc.max() - base)
    b.axvline(100 * k[i29] / n, color=RED, lw=0.8, ls=":")
    b.plot([x[i29]], [100 * (acc[i29] - base)], "o", color=ACCENT, ms=4.2)
    b.annotate(f"29% refined: {100 * (acc[i29] - base) / (acc.max() - base):.0f}% of the\nbest gain measured (up to 42%)", (x[i29], 100 * (acc[i29] - base)), xytext=(11, 1.1), fontsize=6.7, color=ACCENT,
               arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6))
    b.text(0.97, 0.05, "softmax, MC-dropout and random gates,\nand refining everything: to be added", transform=b.transAxes, ha="right", va="bottom", fontsize=6.2, color=GREY, style="italic")
    b.set_xlim(0, 43)
    b.set_xlabel("instruments refined (%), highest uncertainty first", fontsize=7.4)
    b.set_ylabel("instrument accuracy gain (points)")
    b.grid(color="#e1e5ea", lw=0.5)
    b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=6.3, loc="center right")
    b.set_title("(b) Refinement: accuracy gained", fontsize=8.2, loc="left")
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(out / "fig2_effort.pdf", bbox_inches="tight")
    plt.close(fig)


def per_class(out: Path) -> None:
    d = json.loads((REPO_ROOT / "docs" / "reports" / "gtbox_sam" / "final_3seed.json").read_text())["configs"]
    s = [100 * d["single pass"]["per_class_iou"][str(i)] for i in range(1, 8)]
    r = [100 * d["gated (top 833)"]["per_class_iou"][str(i)] for i in range(1, 8)]
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    x = np.arange(7)
    ax.bar(x - 0.19, s, 0.38, color="#b9c4cf", label="single pass")
    ax.bar(x + 0.19, r, 0.38, color=ACCENT, label="after refinement")
    for i in range(7):
        ax.text(x[i] + 0.19, r[i] + 1.0, f"+{r[i] - s[i]:.1f}", ha="center", va="bottom", fontsize=6.8, color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, fontsize=7.2)
    ax.set_ylim(40, 100)
    ax.set_ylabel("class IoU")
    ax.grid(axis="y", color="#e1e5ea", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / "fig4_perclass.pdf", bbox_inches="tight")
    plt.close(fig)
    print("per-class single/refined:", [round(v, 1) for v in s], [round(v, 1) for v in r])


def reliability(out: Path) -> None:
    m = json.loads((REPO_ROOT / "docs" / "reports" / "evidential" / "calibration_baselines.json").read_text())["methods"]
    order = [("Softmax ensemble", "#55687c"), ("Softmax + temperature scaling", "#8467b0"), ("MC dropout (80 passes)", "#b3741b"), ("Evidential (flagship, 1 pass)", ACCENT),
             ("Deep ensemble (3 seeds, 12 nets)", "#a23a5c"), ("Evidential, 3 seeds (12 nets)", ACCENT)]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.4), gridspec_kw={"wspace": 0.28, "hspace": 0.42})
    for ax, (name, col) in zip(axes.ravel(), order):
        bins = m[name]["reliability"]
        cf, ac, nn = (np.array([b[i] for b in bins]) for i in range(3))
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=(0, (3, 3)))
        ax.fill_between(cf, ac, cf, color=col, alpha=0.14, lw=0)
        ax.plot(cf, ac, color=col, lw=1.4)
        ax.scatter(cf, ac, s=6 + nn / 25, color=col, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(name.replace(" (flagship, 1 pass)", " (ours)"), fontsize=7.4, loc="left")
        ax.text(0.97, 0.05, f"ECE {m[name]['calibration']['ece']['mean']:.3f}", transform=ax.transAxes, ha="right", fontsize=7, color=INK)
        ax.grid(color="#e1e5ea", lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("confidence", fontsize=7.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy", fontsize=7.2)
    fig.savefig(out / "fig5_reliability.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    effort(args.out)
    per_class(args.out)
    reliability(args.out)
    print("wrote figures to", args.out)
