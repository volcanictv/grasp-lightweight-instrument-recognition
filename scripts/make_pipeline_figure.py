"""Pipeline overview figure, drawn in the style of an encoder/decoder module diagram: one wide left-to-right panel, pale rounded boxes with thin outlines,
dashed grey containers around each module with a numbered caption underneath, small stacked feature tokens and real frame thumbnails.
Usage:  python scripts/make_pipeline_figure.py --out paper/figures
Thumbnails come from docs/reports/figures/f03a_sample_frame_1.png and f03b_sample_frame_2.png (GraSP frames); the mask thumbnail is the blue overlay of frame 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "font.size": 6, "pdf.fonttype": 42, "text.color": "#1d2330"})
INK, MUTED = "#1d2330", "#6b7280"
BLUE, BLUE_T = "#2a6fb0", "#e6f0fa"
GOLD, GOLD_T = "#b8892a", "#fdf3dc"
PURP, PURP_T = "#6a4fa3", "#efe9f8"
GREEN, GREEN_T = "#4c8a3a", "#e9f5e3"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "reports" / "figures"


def rbox(ax, x, y, w, h, fc, ec, lw=0.8, r=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def label(ax, x, y, s, size=6, color=INK, weight="normal", ha="center", va="center", z=4, **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, linespacing=1.25, **kw)


def container(ax, x0, y0, x1, y1, caption, align="c"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="#9aa1ad", lw=0.7, ls=(0, (3, 2)), zorder=0))
    label(ax, (x0 + x1) / 2 if align == "c" else x1 - 9.7, y1 + 2.8, caption, size=6, color=INK, ha="center" if align == "c" else "right")


def arrow(ax, p, q, color=INK, lw=0.7, dotted=False, rad=0.0):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.35", color=color, lw=lw, ls=(0, (1.5, 1.5)) if dotted else "-",
                                shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def thumb(ax, img, x, y, w, ec=INK, lw=0.6, z=3):
    h = w * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=z)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=z + 1))
    return h


def tokens(ax, x, y, n, color, tint, w=2.2, h=2.2, gap=0.5):
    for i in range(n):
        ax.add_patch(Rectangle((x, y - i * (h + gap)), w, h, fc=tint, ec=color, lw=0.7, zorder=3))


def parallelogram(ax, x, y, w, h, color, tint, skew=1.2):
    ax.add_patch(Polygon([[x, y], [x + w, y], [x + w + skew, y + h], [x + skew, y + h]], closed=True, fc=tint, ec=color, lw=0.7, zorder=3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    f1 = mpimg.imread(FIG_DIR / "f03a_sample_frame_1.png")[..., :3]
    f2 = mpimg.imread(FIG_DIR / "f03b_sample_frame_2.png")[..., :3]
    blue = (f1[..., 2] > f1[..., 0] + 0.12) & (f1[..., 2] > 0.45)
    mask_img = np.repeat(blue[..., None].astype(float), 3, axis=2)
    out_img = f1.copy()
    out_img[blue] = 0.55 * out_img[blue] + 0.45 * np.array([0.30, 0.69, 0.31])  # recoloured mask in the output

    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 51)
    ax.axis("off")
    ax.invert_yaxis()  # image coordinates: y grows downward, thumbnails keep their orientation
    ax.set_ylim(51, 0)

    # input frame with the given box
    h = thumb(ax, f1, 1, 21, 15)
    ax.add_patch(Rectangle((1 + 15 * 0.66, 21 + h * 0.34), 15 * 0.34, h * 0.40, fc="none", ec="#e5b800", lw=1.1, zorder=6))
    label(ax, 8.5, 21 + h + 3.4, "frame $t$ and\ngiven box", size=5.6)

    # (1) box-prompted segmenters
    container(ax, 19, 13, 48, 38, "(1) Box-prompted segmenters")
    for yy, name in ((16, "SAM2.1-large"), (27, "SAM3")):
        rbox(ax, 21, yy, 13.5, 8, BLUE_T, BLUE)
        label(ax, 27.75, yy + 4, name, size=5.8)
    label(ax, 27.75, 14.6, "fine-tuned on GraSP", size=4.6, color=MUTED)
    arrow(ax, (16, 28), (21, 20.0))
    arrow(ax, (16, 28), (21, 31.0))
    arrow(ax, (34.5, 20), (37, 25.5))
    arrow(ax, (34.5, 31), (37, 28.5))
    hm = thumb(ax, mask_img, 37, 22, 9.5)
    label(ax, 41.75, 22 + hm + 2.2, "mask", size=5.4, color=MUTED)

    # (2) evidential ensemble
    container(ax, 50, 13, 81, 38, "(2) Evidential ensemble")
    rbox(ax, 52, 16, 15, 20, GOLD_T, GOLD)
    label(ax, 59.5, 18.6, "4 networks", size=5.8)
    for i in range(4):
        rbox(ax, 54, 21 + i * 3.5, 11, 2.7, "white", GOLD, lw=0.5, r=0.5)
    arrow(ax, (46.5, 27), (52, 27))
    arrow(ax, (67, 27), (70, 27))
    for i, c in enumerate((GOLD, GOLD, GOLD)):
        parallelogram(ax, 70.5, 18.5 + i * 3.6, 4.5, 2.6, GOLD, GOLD_T)
    label(ax, 74, 31.5, r"belief $\mu$", size=5.4)
    label(ax, 74, 34.3, r"class $\hat{y}$", size=5.4)
    rbox(ax, 76.6, 20.5, 4, 8, "white", GOLD, lw=0.7, r=0.6)
    label(ax, 78.6, 24.5, "$S$", size=7.5, weight="bold")

    # (3) uncertainty gate
    container(ax, 83, 13, 100, 38, "(3) Uncertainty\ngate", align="r")
    cx, cy = 91.5, 25.5
    ax.add_patch(Polygon([[cx - 6, cy], [cx, cy - 7], [cx + 6, cy], [cx, cy + 7]], closed=True, fc=GREEN_T, ec=GREEN, lw=0.8, zorder=3))
    label(ax, cx, cy - 0.8, r"$S\geq\tau$", size=6.2)
    label(ax, cx, cy + 2.6, "or top-$K$", size=4.4, color=MUTED)
    arrow(ax, (80.6, 25), (85.5, 25.5))
    label(ax, 96.6, 17.0, "uncertain", size=4.8, color=GREEN, weight="bold")
    label(ax, 93.0, 35.4, "confident", size=4.8, color=GREEN, weight="bold", ha="left")

    # (4) temporal refinement
    container(ax, 102, 13, 128, 38, "(4) Temporal refinement")
    for k, (dx, dy) in enumerate(((0, 0), (1.6, -1.6), (3.2, -3.2))):
        thumb(ax, f2, 104 + dx, 27 + dy, 8.5, z=3 + k)
    label(ax, 108.3, 33.4, r"frames $t\pm10$", size=5.2)
    rbox(ax, 104.5, 14.5, 13.5, 6, PURP_T, PURP)
    label(ax, 111.25, 17.5, "SAM2-large tracker", size=5.2)
    arrow(ax, (109.5, 23.6), (109.5, 20.5), color=PURP)
    rbox(ax, 118.5, 21, 8.5, 15, "white", INK, lw=0.7, r=1.2)
    label(ax, 122.75, 23.2, "Fusion", size=5.4)
    for i in range(3):
        yy = 26.0 + i * 2.8
        ax.add_patch(Circle((120.3, yy), 0.7, fc="white", ec=INK, lw=0.5, zorder=4))
        label(ax, 120.3, yy, "$\\times$", size=3.8)
        parallelogram(ax, 122.0, yy - 1.2, 2.6, 2.2, PURP, PURP_T, skew=0.8)
    label(ax, 122.75, 34.4, "conf.-\nweighted", size=3.8, color=MUTED)
    arrow(ax, (118.0, 17.5), (122.75, 21.0), color=PURP)
    arrow(ax, (97.5, 22.0), (104.5, 17.5), color=GREEN)

    # output
    thumb(ax, out_img, 130.2, 23.5, 9.5)
    label(ax, 135, 23.5 + 9.5 * 0.625 + 3.4, "instrument\nlabel", size=5.4)
    arrow(ax, (127.0, 28.5), (130.2, 28.5))

    # neighbour frames are fetched for the gated instruments only
    ax.plot([8.5, 8.5, 103.2, 103.2], [21, 6.5, 6.5, 31.0], color=MUTED, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
    ax.annotate("", xy=(104.0, 31.0), xytext=(103.2, 31.0), arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.3", color=MUTED, lw=0.6), zorder=1)
    label(ax, 58, 4.6, "neighbouring frames are read only for the gated instruments", size=4.8, color=MUTED)

    # confident instruments keep the one-pass label
    ax.plot([91.5, 91.5, 134.95], [32.5, 47.2, 47.2], color=GREEN, lw=0.8, zorder=1)
    arrow(ax, (134.95, 47.2), (134.95, 36.0), color=GREEN)
    label(ax, 112, 45.6, "keep the one-pass label", size=4.8, color=GREEN)

    fig.subplots_adjust(0.003, 0.003, 0.997, 0.997)
    fig.savefig(args.out / "fig_pipeline.pdf")
    fig.savefig(args.out / "fig_pipeline.png", dpi=200)
    print("wrote", args.out / "fig_pipeline.pdf")


if __name__ == "__main__":
    main()
