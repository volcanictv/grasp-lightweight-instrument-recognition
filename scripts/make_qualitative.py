"""Qualitative figure of the paper: for chosen test frames, the ground-truth masks, the causal real-time pipeline (SAM2-tiny masks, EdgeTAM look-back, 539 most
uncertain instruments tracked) and the non-causal pipeline (SAM2 + SAM3 masks, SAM2-large over +-10 frames, 833 tracked), all for the headline seed 44. Masks are coloured by
class, instruments sent to tracking are marked with an asterisk, wrong classes have a red outline. Reads saved results and the test frames only (titanxp, CPU).

Usage (titanxp, surgical environment, repo root):
    python scripts/make_qualitative.py candidates --out experiments/figure_examples/qual_candidates
    python scripts/make_qualitative.py fig --frames CASE050/08645.jpg ... --out experiments/figure_examples/paper
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_codec

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from gtbox_sam_final_eval import fuse, load_tracked
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores

DATA_ROOT = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))
RUNS = {"causal": ("rt_tiny", "tracked_b", 539), "noncausal": ("final", "tracked", 833)}
SEED = 44
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]
SHORT = ["Bipolar", "Prograsp", "Needle driver", "Scissors", "Suction", "Clip applier", "Grasper"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 7, "pdf.fonttype": 42})


def run_labels(name: str) -> dict:
    run, tracks, budget = RUNS[name]
    d = REPO_ROOT / "experiments" / "gtbox_sam" / run
    z = np.load(d / f"logits_s{SEED}.npz")
    y, done = z["y"], z["done"]
    alpha = alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"])
    base = (alpha / alpha.sum(axis=1, keepdims=True)).argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    gated = np.zeros(len(y), bool)
    gated[np.argsort(-s1, kind="stable")[:budget]] = True
    tracked = load_tracked(d / f"{tracks}_s{SEED}")
    pred = base.copy()
    sent = np.zeros(len(y), bool)
    for i in np.where(gated & done)[0]:
        if int(i) in tracked:
            pred[i] = fuse(tracked[int(i)], CONFIGS["four"])[0]
            sent[i] = True
    masks = pickle.loads((d / "masks.pkl").read_bytes())
    return {"y": y, "base": base, "pred": pred, "sent": sent, "masks": masks}


def summaries(ds, labels: dict) -> list[dict]:
    by_frame = defaultdict(list)
    for i, inst in enumerate(ds.instances):
        by_frame[inst[0]].append(i)
    rows = []
    for f, idx in by_frame.items():
        r = {"file": f, "n": len(idx)}
        for k, L in labels.items():
            r[k + "_wrong"] = int(sum(L["pred"][i] != L["y"][i] for i in idx))
            r[k + "_fixed"] = int(sum(L["base"][i] != L["y"][i] and L["pred"][i] == L["y"][i] for i in idx))
            r[k + "_broken"] = int(sum(L["base"][i] == L["y"][i] and L["pred"][i] != L["y"][i] for i in idx))
        r["classes"] = sorted({SHORT[int(labels["causal"]["y"][i])] for i in idx})
        rows.append(r)
    return rows


def panel(ax, image, items, title=None):
    ax.imshow(image)
    for m, cls, wrong, sent, label, fixed in items:
        rgba = np.zeros((*m.shape, 4))
        rgba[m] = (*matplotlib.colors.to_rgb(COLORS[cls]), 0.5)
        ax.imshow(rgba)
        ax.contour(m.astype(float), levels=[0.5], colors="#d1242f" if wrong else ("#3fb950" if fixed else "white"), linewidths=1.4 if (wrong or fixed) else 0.7)
        ys, xs = np.where(m)
        if len(xs):
            txt = label + (" *" if sent else "")
            ax.text(float(np.median(xs)), float(ys.min()) - 6, txt, fontsize=5.4, color="white", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#d1242f" if wrong else "#222222", ec="none", alpha=0.85))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7.5, loc="left", pad=2)


def render(files, ds, labels, out: Path, name: str) -> None:
    by_frame = defaultdict(list)
    for i, inst in enumerate(ds.instances):
        by_frame[inst[0]].append(i)
    frames_root = DATA_ROOT / "frames-001" / "frames"
    h = 1.40 * len(files) + 0.5
    fig, axes = plt.subplots(len(files), 3, figsize=(6.6, h), gridspec_kw={"wspace": 0.02, "hspace": 0.04, "left": 0.003, "right": 0.997, "top": 1 - 0.22 / h,
                                                                           "bottom": 0.5 / h})
    axes = np.atleast_2d(axes)
    for r, f in enumerate(files):
        idx = by_frame[f]
        image = np.array(Image.open(frames_root / f).convert("RGB"))
        gt = [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(labels["causal"]["y"][i]), False, False, SHORT[int(labels["causal"]["y"][i])], False) for i in idx]
        panel(axes[r, 0], image, gt, "Ground truth" if r == 0 else None)
        for c, k in enumerate(("causal", "noncausal")):
            L = labels[k]
            items = [(mask_codec.decode(L["masks"][i]).astype(bool), int(L["pred"][i]), bool(L["pred"][i] != L["y"][i]), bool(L["sent"][i]), SHORT[int(L["pred"][i])],
                      bool(L["base"][i] != L["y"][i] and L["pred"][i] == L["y"][i])) for i in idx if i in L["masks"]]
            panel(axes[r, c + 1], image, items, ("Ours, causal (real time)" if k == "causal" else "Ours, non-causal") if r == 0 else None)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[i], alpha=0.7) for i in range(7)]
    fig.legend(handles, SHORT, loc="lower center", ncol=7, frameon=False, fontsize=6.5, handlelength=1.0, columnspacing=1.1, bbox_to_anchor=(0.5, 0.2 / h))
    fig.text(0.5, 0.02 / h, "* sent to tracking     green outline: fixed by tracking     red outline: wrong class", ha="center", va="bottom", fontsize=6.3, color="#444")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("candidates")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("fig")
    b.add_argument("--frames", nargs="+", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--name", default="fig_qualitative")
    args = ap.parse_args()
    ds = GraspRegionDataset(DATA_ROOT, "test", letterbox=True)
    labels = {k: run_labels(k) for k in RUNS}
    if args.cmd == "candidates":
        rows = summaries(ds, labels)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "frames.json").write_text(json.dumps(rows, indent=1))
        fixed = [r for r in rows if r["causal_fixed"] >= 1 and r["causal_wrong"] == 0 and 2 <= r["n"] <= 4]
        clean = [r for r in rows if r["causal_wrong"] == 0 and r["causal_fixed"] == 0 and 3 <= r["n"] <= 4]
        fail = [r for r in rows if r["causal_wrong"] >= 1 and 2 <= r["n"] <= 3 and r["noncausal_wrong"] == 0]
        print(len(rows), "frames;", len(fixed), "fixed and all right;", len(clean), "all right untracked;", len(fail), "causal wrong, non-causal right")
        for tag, pool in (("fixed", fixed), ("clean", clean), ("fail", fail)):
            for r in pool[:6]:
                print(tag, r["file"], r["n"], r["classes"], "fixed", r["causal_fixed"], "wrong", r["causal_wrong"])
        render([r["file"] for r in fixed[:3]], ds, labels, args.out, "sheet_fixed")
        render([r["file"] for r in clean[:3]], ds, labels, args.out, "sheet_clean")
        render([r["file"] for r in fail[:3]], ds, labels, args.out, "sheet_fail")
    else:
        render(args.frames, ds, labels, args.out, args.name)
        print("wrote", args.out / f"{args.name}.pdf")


if __name__ == "__main__":
    main()
