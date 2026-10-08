"""Figure for the Method section: how the pipeline works, from a frame with a given box to a label, and where the uncertainty sends the effort.
Vector PDF in the style of the other figures (serif, the teal accent). Usage:  python scripts/make_pipeline_figure.py --out paper/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 7,
         "pdf.fonttype": 42, "text.color": "#1c2430"}
plt.rcParams.update(STYLE)
INK, MUTED, ACCENT, GREY, AMBER = "#1c2430", "#5d6877", "#0b6e8a", "#8a94a3", "#a8651a"
TINT = {INK: "#eef1f5", ACCENT: "#e4f2f6", AMBER: "#f8eedd", GREY: "#f3f4f6"}


def box(ax, x, y, w, h, title, body, color=INK, title_size=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=TINT[color], ec=color, lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=MUTED if color == GREY else color, zorder=3)
    ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=6.3, color=INK, linespacing=1.35, zorder=3)


def arrow(ax, p, q, color=MUTED, rad=0.0, lw=0.9, text=None, tpos=None, tcolor=None):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if text:
        ax.text(*tpos, text, fontsize=6.3, color=tcolor or color, ha="center", va="center", fontweight="bold", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    # 1. the input: a frame with the given box
    ax.add_patch(Rectangle((1, 13), 12, 14, fc="#dfe4ea", ec=INK, lw=0.9, zorder=2))
    ax.add_patch(Rectangle((5.2, 16.2), 5.6, 6.6, fc="none", ec="#d6a400", lw=1.4, zorder=3))
    ax.text(7, 11.2, "frame +\ngiven box", ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.25)

    # 2. segmenters, 3. evidential ensemble
    box(ax, 17, 10, 19, 20, "Box-prompted\nsegmenters", "SAM2.1-large and SAM3,\nfine-tuned on GraSP,\nmirror-averaged\nlogits  $\\rightarrow$  mask", ACCENT)
    box(ax, 40, 10, 19, 20, "Evidential\nensemble", "4 small networks,\none forward pass\n$\\rightarrow$ class $\\hat{y}$, belief $\\mu$,\nuncertainty $S$", ACCENT)

    # 4. the gate
    cx, cy = 67.5, 20
    ax.add_patch(Polygon([[cx - 5.5, cy], [cx, cy + 7], [cx + 5.5, cy], [cx, cy - 7]], closed=True, fc="white", ec=INK, lw=0.9, zorder=2))
    ax.text(cx, cy + 0.6, "$S\\geq\\tau$", ha="center", va="center", fontsize=7.2, color=INK, zorder=3)
    ax.text(cx, cy - 3.0, "or top-$K$", ha="center", va="center", fontsize=5.3, color=MUTED, zorder=3)

    # 5. the three outcomes
    box(ax, 77, 29, 22.5, 10.2, "Review", "a person corrects the label:\n20% checked, 90% of errors found", AMBER, 7.2)
    box(ax, 77, 12.6, 22.5, 14.8, "Refine", "SAM2-large carries the mask\n$\\pm$10 frames; each frame is\nclassified; confidence-weighted\nfusion  $\\rightarrow$  fused label", ACCENT, 7.2)
    box(ax, 77, 0.6, 22.5, 8.6, "Keep", "single-pass label", GREY, 7.2)

    # arrows
    arrow(ax, (13, 20), (17, 20))
    arrow(ax, (36, 20), (40, 20))
    arrow(ax, (59, 20), (cx - 5.5, 20))
    arrow(ax, (cx, cy + 7), (77, 34.1), AMBER, rad=-0.25, text="high $S$: either use", tpos=(70.3, 31.6), tcolor=AMBER)
    arrow(ax, (cx + 5.5, cy), (77, 20), ACCENT, text="", tpos=(0, 0))
    arrow(ax, (cx, cy - 7), (77, 4.9), GREY, rad=0.25, text="low $S$", tpos=(70.7, 6.6), tcolor=MUTED)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.01)
    fig.savefig(args.out / "fig_pipeline.pdf")
    fig.savefig(args.out / "fig_pipeline.png", dpi=220)
    print("wrote", args.out / "fig_pipeline.pdf")


if __name__ == "__main__":
    main()
