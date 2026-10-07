"""Figures of the MIDL paper. `fig1` needs the test frames and the final run's tracks (titanxp, CPU); `plots` draws the result plots from the saved JSON files
(any machine). Both write vector PDFs into --out.

Usage:
    titanxp:  python scripts/make_paper_figures.py fig1 --index 810 --out experiments/figure_examples/paper
    laptop :  python scripts/make_paper_figures.py plots --out paper/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "cm",
         "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "text.color": "#1c2430","axes.edgecolor": "#8a94a3",
         "axes.labelcolor": "#1c2430", "xtick.color": "#5d6877", "ytick.color": "#1c2430"}
plt.rcParams.update(STYLE)
INK, MUTED, ACCENT, GREY, RED, GREEN = "#1c2430", "#5d6877", "#0b6e8a", "#8a94a3", "#b3382f", "#1d7a46"
CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
SHORT = ["Bipolar", "Prograsp", "Needle driver", "Scissors", "Suction", "Clip applier", "Grasper"]


def fig1(index: int, out: Path) -> None:
    import figure_example as fe
    from surgical_ai.data.region_dataset import GraspRegionDataset
    import torch

    torch.set_num_threads(2)
    members = fe.load_members(fe.REPO_ROOT / "configs" / "arms" / "ens4_N_official_s44.yaml")
    ds = GraspRegionDataset(fe.DATA_ROOT, "test", letterbox=True)
    ex = fe.example(index, members, ds, fe.load_tracks())
    frames = ex["frames"]
    want = [-20, -17, -13, -10, -7, -3, 0] if min(f["offset"] for f in frames) < -10 and max(f["offset"] for f in frames) <= 0 else [-10, -7, -3, 0, 3, 7, 10]
    by_off = {f["offset"]: f for f in frames}
    show = [by_off[o] for o in want if o in by_off]
    true = ex["true_index"]
    x, y, w, h = ex["box"]
    key = by_off[0]

    fig = plt.figure(figsize=(6.6, 4.1))
    top = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.3, 1.05], left=0.01, right=0.99, top=0.94, bottom=0.45, wspace=0.0)
    gs = fig.add_gridspec(1, len(show), wspace=0.08, left=0.01, right=0.99, top=0.33, bottom=0.03)
    # (a) full keyframe with the box
    axa = fig.add_subplot(top[0, 0])
    axa.imshow(key["image"])
    axa.add_patch(Rectangle((x, y), w, h, fill=False, ec="#ffd54a", lw=1.6))
    axa.contour(key["mask"].astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.1)
    axa.axis("off")
    axa.set_title("(a) Keyframe: given box, predicted mask", fontsize=8, loc="left")
    # (b) beliefs before and after fusion
    axb = fig.add_subplot(top[0, 2])
    pos = np.arange(7)[::-1]
    axb.barh(pos + 0.19, ex["single"], height=0.34, color="#c9ced6", label="keyframe alone")
    axb.barh(pos - 0.19, ex["fused"], height=0.34, color=ACCENT, label="fused over the frames")
    axb.set_yticks(pos)
    labels = axb.set_yticklabels(SHORT, fontsize=7.5)
    labels[true].set_fontweight("bold")
    labels[true].set_color(GREEN)
    axb.set_xlim(0, 1)
    axb.set_xlabel("class belief $\\mu$ (true class in green)", fontsize=7.5)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.02, 0.97))
    axb.set_title("(b) Class belief", fontsize=8, loc="left")
    axb.text(0.98, 0.42, f"keyframe alone: {SHORT[int(np.argmax(ex['single']))]} ({ex['single'].max():.2f}),\nuncertainty $S$ = {ex['s1']:.1e}\nfused: {SHORT[int(np.argmax(ex['fused']))]} ({ex['fused'].max():.2f})",
             transform=axb.transAxes, ha="right", va="center", fontsize=7, color=INK, linespacing=1.3)
    # (c) the frames
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * 1.5
    for j, f in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        H, W = f["image"].shape[:2]
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        ax.imshow(f["image"][y0:y1, x0:x1])
        ax.contour(f["mask"][y0:y1, x0:x1].astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        k = int(np.argmax(f["mu"]))
        col = GREEN if k == true else RED
        for s in ax.spines.values():
            s.set_edgecolor("#ffd54a" if f["offset"] == 0 else "#d5d9df")
            s.set_linewidth(2.2 if f["offset"] == 0 else 0.6)
        tag = "keyframe" if f["offset"] == 0 else f"$t$ = {f['offset']:+d} s".replace("-", "−")
        ax.set_xlabel(f"{tag}\n{SHORT[k]} {f['mu'][k]:.2f}", fontsize=7, color=col, labelpad=2)
        if j == 0:
            ax.set_title("(c) Past frames with the propagated mask, each classified alone", fontsize=8, loc="left")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig1_example.pdf")
    fig.savefig(out / "fig1_example.png", dpi=200)
    print("wrote", out / "fig1_example.pdf", ex["file"], ex["true"], "S1", ex["s1"])


def plots(out: Path) -> None:
    R = REPO_ROOT / "docs" / "reports"
    out.mkdir(parents=True, exist_ok=True)

    # results against TAPIS: final_3seed.json means, bootstrap_cis.json case intervals, published TAPIS numbers
    rows = [("mIoU", 87.37, (86.85, 88.21), 86.61), ("IoU", 86.22, (85.56, 87.09), 83.38), ("mcIoU", 78.33, (76.33, 79.81), 77.42)]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for i, (name, v, ci, t) in enumerate(rows):
        yy = len(rows) - 1 - i
        ax.plot(ci, [yy, yy], color=ACCENT, lw=1.6, solid_capstyle="butt")
        for c in ci:
            ax.plot([c, c], [yy - 0.1, yy + 0.1], color=ACCENT, lw=1.4)
        ax.plot(v, yy, "o", color=ACCENT, ms=5.5, zorder=3)
        ax.plot(t, yy, "s", color=GREY, ms=5, zorder=3)
        ax.text(v, yy + 0.24, f"{v:.2f}", ha="center", fontsize=7, color=ACCENT, fontweight="bold")
        ax.text(t, yy - 0.34, f"{t:.2f}", ha="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(74, 90)
    ax.set_ylim(-0.7, 2.6)
    ax.grid(axis="x", color="#e1e5ea", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.plot([], [], "o", color=ACCENT, label="ours (95% interval over test cases)")
    ax.plot([], [], "s", color=GREY, label="TAPIS")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "whisker.pdf")
    plt.close(fig)

    # per-class IoU of the final pipeline (final_3seed.json, gated top 833)
    data = [("Scissors", 94.2), ("Needle driver", 87.0), ("Bipolar", 84.0), ("Suction", 78.1), ("Clip applier", 76.8), ("Prograsp", 67.5), ("Grasper", 60.7)]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for i, (n, v) in enumerate(data):
        yy = len(data) - 1 - i
        ax.barh(yy, v, color=RED if v < 70 else ACCENT, height=0.62)
        ax.text(v + 1, yy, f"{v:.1f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in data][::-1])
    ax.set_xlim(0, 106)
    ax.set_xlabel("IoU")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "classes.pdf")
    plt.close(fig)

    # ROC curves on the held-out fold (roc_fold1.json)
    roc = json.loads((R / "evidential" / "roc_fold1.json").read_text())
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--")
    ax.plot(roc["fpr"], roc["tpr_mean"]["MC dropout"], color=GREY, lw=1.6, label=f"MC dropout (AUROC {roc['auroc_mean']['MC dropout']:.3f})")
    ax.plot(roc["fpr"], roc["tpr_mean"]["Evidential"], color=ACCENT, lw=1.6, label=f"Evidential (AUROC {roc['auroc_mean']['Evidential']:.3f})")
    ax.set_xlabel("correct predictions flagged")
    ax.set_ylabel("errors caught")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e1e5ea", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=6.8)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "roc.pdf")
    plt.close(fig)

    # gate sweep on the held-out fold (fold1_calibration.json)
    cal = json.loads((R / "evidential_pipeline_switch" / "fold1_calibration.json").read_text())
    pts = sorted((100 * r["share"], r["accuracy"]) for r in cal["rows"].values())
    chosen = (100 * cal["chosen_row"]["share"], cal["chosen_row"]["accuracy"])
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    ax.axhline(cal["base_accuracy"], color=GREY, ls="--", lw=0.9)
    ax.text(32.5, cal["base_accuracy"] + 0.0007, f"no tracking {cal['base_accuracy']:.4f}", ha="right", fontsize=6.8, color=MUTED)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ACCENT, lw=1.5, marker="o", ms=3.2)
    ax.plot(*chosen, "o", ms=8.5, mfc="none", mec=ACCENT, mew=1.3)
    ax.annotate(f"chosen: {chosen[0]:.0f}%, {chosen[1]:.4f}", chosen, xytext=(chosen[0] + 1.5, chosen[1] - 0.0125), fontsize=7, color=ACCENT,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))
    ax.set_xlabel("instruments sent to tracking (%)")
    ax.set_ylabel("held-out accuracy")
    ax.set_xlim(0, 33)
    ax.grid(color="#e1e5ea", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "gate.pdf")
    plt.close(fig)
    fig2(out)
    print("wrote plots to", out)


def fig2(out: Path) -> None:
    """(a) accuracy against the share of instruments tracked, evidence-ordered; (b) mIoU against end-to-end latency (estimated from stage times)."""
    R = REPO_ROOT / "docs" / "reports"
    curves = json.loads((R / "realtime" / "gate_budget_curve.json").read_text())
    import build_pi_report_docx as lat  # latency numbers computed from the measurement files
    plt.rcParams.update(STYLE)  # the import above sets its own plot style

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.32})
    spec = {"causal EdgeTAM (SAM2-tiny masks)": (ACCENT, "causal (past frames)"), "offline SAM2-large (SAM2 + SAM3 masks)": ("#6b5b95", "non-causal")}
    for name, (col, lab) in spec.items():
        c = curves[name]
        x = np.array(c["k"]) / c["n"] * 100
        a.plot(x, c["accuracy_mean"], color=col, lw=1.6, label=lab)
        a.plot(x, c["oracle_mean"], color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.6)
        if name.startswith("causal"):
            i = c["k"].index(540)
            gain_540 = c["accuracy_mean"][i] - c["base_accuracy"]
            gain_all = c["accuracy_mean"][-1] - c["base_accuracy"]
            a.plot(x[i], c["accuracy_mean"][i], "o", color=col, ms=4.5)
            a.annotate(f"19% tracked:\n{100 * gain_540 / gain_all:.0f}% of the gain", (x[i], c["accuracy_mean"][i]), xytext=(x[i] + 3, c["accuracy_mean"][i] + 0.010),
                       fontsize=6.8, color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
    a.plot([], [], color=GREY, lw=0.9, ls=(0, (3, 2)), label="best possible gate")
    a.set_xlabel("instruments sent to tracking (%), highest evidential score first", fontsize=7.2)
    a.set_ylabel("instrument accuracy")
    a.set_xlim(0, 43)
    a.grid(color="#e1e5ea", lw=0.5)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.legend(frameon=False, fontsize=6.5, loc="lower right")
    a.set_title("(a) Evidence concentrates the gain", fontsize=8, loc="left")

    pts = [("ours, causal, EdgeTAM", lat.RT_AVG, 84.22, "o", ACCENT, True), ("ours, causal, YOLO26s", lat.YOLO_AVG, 83.23, "o", ACCENT, True),
           ("ours, causal, no tracking", lat.RT_BEST, 82.13, "o", ACCENT, True), ("ours, non-causal", lat.OFF_AVG_PI, 87.37, "s", ACCENT, False),
           ("ours, causal, SAM2-large", lat.RT_L_AVG, 85.00, "o", ACCENT, True), ("ours, causal,\nSAM2 + SAM3 masks", lat.OFF_AVG_PI, 86.50, "o", ACCENT, True),
           ("TAPIS", lat.tapis_ms, 86.61, "s", GREY, False)]
    for lab, xv, yv, mk, col, filled in pts:
        b.plot(xv, yv, mk, color=col, mfc=col if filled else "white", mew=1.4, ms=6)
    off = {"ours, causal, EdgeTAM": (6, 4, "left"), "ours, causal, YOLO26s": (-6, 2, "right"), "ours, causal, no tracking": (7, -3, "left"),
           "ours, non-causal": (-7, 3, "right"), "TAPIS": (-7, -10, "right"), "ours, causal, SAM2-large": (6, -3, "left"),
           "ours, causal,\nSAM2 + SAM3 masks": (8, -14, "left")}
    for lab, xv, yv, *_ in pts:
        dx, dy, ha = off[lab]
        b.annotate(lab, (xv, yv), xytext=(dx, dy), textcoords="offset points", fontsize=6.6, ha=ha, color=INK)
    b.axvline(1000, color=RED, lw=0.8, ls=":")
    b.text(1080, 81.35, "1 frame/s", fontsize=6.5, color=RED)
    b.set_xscale("log")
    b.set_xlim(200, 60000)
    b.set_ylim(81, 88.3)
    b.set_xlabel("end-to-end latency per instrument (ms, estimated)", fontsize=7.2)
    b.set_ylabel("mIoU")
    b.grid(color="#e1e5ea", lw=0.5, which="both")
    b.set_axisbelow(True)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.set_title("(b) Accuracy against latency", fontsize=8, loc="left")
    fig.savefig(out / "fig2_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fig1")
    a.add_argument("--index", type=int, default=810)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("plots")
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fig1(args.index, args.out) if args.cmd == "fig1" else plots(args.out)
