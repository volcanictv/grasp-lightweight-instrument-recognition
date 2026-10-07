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
    """(a) instrument accuracy against the share of instruments sent to refinement, evidence-ordered, final pipeline; (b) the same accuracy against the average time per instrument on
    one A100: frame stage + share * measured cost per refined instrument (docs/reports/latency_a100/paper_table.json, made by scripts/a100_latency_table.py)."""
    R = REPO_ROOT / "docs" / "reports"
    c = json.loads((R / "realtime" / "gate_budget_curve.json").read_text())["offline SAM2-large (SAM2 + SAM3 masks)"]
    row = json.loads((R / "latency_a100" / "paper_table.json").read_text())["Non-causal, SAM2 + SAM3 masks, SAM2-large tracking"]
    frame_s, cost_s = row["frame_stage_per_instrument_ms"] / 1000, row["cost_per_tracked_ms"] / 1000
    plt.rcParams.update(STYLE)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30})
    col = ACCENT
    k, n, acc, base = np.array(c["k"]), c["n"], np.array(c["accuracy_mean"]), c["base_accuracy"]
    share = k / n
    i540, i31 = c["k"].index(540), min(range(len(c["k"])), key=lambda j: abs(c["k"][j] - 0.31 * n))
    i833 = min(range(len(c["k"])), key=lambda j: abs(c["k"][j] - 833))
    a.plot(100 * share, acc, color=col, lw=1.6, label="evidential gate")
    a.plot(100 * share, c["oracle_mean"], color=GREY, lw=0.9, ls=(0, (3, 2)), label="best possible gate")
    a.plot(100 * share[i540], acc[i540], "o", color=col, ms=4.5)
    gain540, gain31 = acc[i540] - base, acc[i31] - base
    a.annotate(f"19%: {100 * gain540 / gain31:.0f}% of the\ngain at 31%", (100 * share[i540], acc[i540]), xytext=(100 * share[i540] + 5, acc[i540] - 0.026), fontsize=6.8, color=col,
               arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
    a.set_xlabel("instruments sent to refinement (%), highest evidential score first", fontsize=7.2)
    a.set_ylabel("instrument accuracy")
    a.set_xlim(0, 43)
    a.grid(color="#e1e5ea", lw=0.5)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.legend(frameon=False, fontsize=6.5, loc="lower right")
    a.set_title("(a) Evidence concentrates the gain", fontsize=8, loc="left")

    t = frame_s + share * cost_s
    b.plot(t, acc, color=col, lw=1.6)
    b.plot(t[i833], acc[i833], "o", color=col, ms=4.8)
    best_gain = acc.max() - base
    b.annotate(f"29% refined: {t[i833]:.1f} s,\n{100 * (acc[i833] - base) / best_gain:.0f}% of the gain", (t[i833], acc[i833]), xytext=(t[i833] + 0.7, acc[i833] - 0.030), fontsize=6.8, color=col,
               arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
    all_t = frame_s + cost_s
    b.axvline(all_t, color=RED, lw=0.9, ls=":")
    b.text(all_t - 0.15, base + 0.004, f"every instrument\nrefined: {all_t:.1f} s\n(estimated)", fontsize=6.6, color=RED, ha="right", va="bottom")
    b.set_xlim(frame_s - 0.1, all_t + 0.4)
    b.set_xlabel("average time per instrument on one A100 (s)", fontsize=7.2)
    b.set_ylabel("instrument accuracy")
    b.grid(color="#e1e5ea", lw=0.5)
    b.set_axisbelow(True)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.set_title("(b) Accuracy against time", fontsize=8, loc="left")
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
