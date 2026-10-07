"""Qualitative figure of the paper: for chosen test frames, the ground-truth masks, the labels of the single pass (no refinement) and the labels after evidence-gated refinement
(SAM2 + SAM3 masks, SAM2-large over +-10 frames, the 833 most uncertain instruments refined), on the same masks, for the headline seed 44. Masks are coloured by class, instruments
sent to refinement are marked with an asterisk, wrong classes have a red outline and classes fixed by refinement a green one. Reads saved results and the test frames only (titanxp, CPU).

Usage (titanxp, surgical environment, repo root):
    python scripts/make_qualitative.py candidates --out experiments/figure_examples/qual_candidates_v2
    python scripts/make_qualitative.py fig --frames CASE050/08645.jpg ... --out experiments/figure_examples/paper_v2
(The earlier version of this figure contrasted a causal real-time pipeline with the pipeline that reads future frames; that version is kept in paper_local, not in the paper.)
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
RUN, TRACKS, BUDGET = "final", "tracked", 833
SEED = 44
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]
SHORT = ["Bipolar", "Prograsp", "Needle driver", "Scissors", "Suction", "Clip applier", "Grasper"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 7, "pdf.fonttype": 42})


def run_labels() -> dict:
    d = REPO_ROOT / "experiments" / "gtbox_sam" / RUN
    z = np.load(d / f"logits_s{SEED}.npz")
    y, done = z["y"], z["done"]
    alpha = alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"])
    base = (alpha / alpha.sum(axis=1, keepdims=True)).argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    gated = np.zeros(len(y), bool)
    gated[np.argsort(-s1, kind="stable")[:BUDGET]] = True
    tracked = load_tracked(d / f"{TRACKS}_s{SEED}")
    pred = base.copy()
    sent = np.zeros(len(y), bool)
    for i in np.where(gated & done)[0]:
        if int(i) in tracked:
            pred[i] = fuse(tracked[int(i)], CONFIGS["four"])[0]
            sent[i] = True
    masks = pickle.loads((d / "masks.pkl").read_bytes())
    refined = {"y": y, "base": base, "pred": pred, "sent": sent, "masks": masks}
    single = {**refined, "pred": base.copy(), "sent": np.zeros(len(y), bool)}
    return {"single": single, "refined": refined}


def summaries(ds, labels: dict) -> list[dict]:
    by_frame = defaultdict(list)
    for i, inst in enumerate(ds.instances):
        by_frame[inst[0]].append(i)
    S, R = labels["single"], labels["refined"]
    rows = []
    for f, idx in by_frame.items():
        rows.append({"file": f, "n": len(idx), "single_wrong": int(sum(S["pred"][i] != S["y"][i] for i in idx)), "refined_wrong": int(sum(R["pred"][i] != R["y"][i] for i in idx)),
                     "sent": int(sum(R["sent"][i] for i in idx)), "fixed": int(sum(R["base"][i] != R["y"][i] and R["pred"][i] == R["y"][i] for i in idx)),
                     "broken": int(sum(R["base"][i] == R["y"][i] and R["pred"][i] != R["y"][i] for i in idx)), "classes": sorted({SHORT[int(R["y"][i])] for i in idx})})
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
    y = labels["refined"]["y"]
    for r, f in enumerate(files):
        idx = by_frame[f]
        image = np.array(Image.open(frames_root / f).convert("RGB"))
        gt = [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y[i]), False, False, SHORT[int(y[i])], False) for i in idx]
        panel(axes[r, 0], image, gt, "Ground truth" if r == 0 else None)
        for c, k in enumerate(("single", "refined")):
            L = labels[k]
            items = [(mask_codec.decode(L["masks"][i]).astype(bool), int(L["pred"][i]), bool(L["pred"][i] != L["y"][i]), bool(L["sent"][i]), SHORT[int(L["pred"][i])],
                      bool(L["base"][i] != L["y"][i] and L["pred"][i] == L["y"][i])) for i in idx if i in L["masks"]]
            panel(axes[r, c + 1], image, items, ("Single pass" if k == "single" else "Evidence-gated refinement") if r == 0 else None)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[i], alpha=0.7) for i in range(7)]
    fig.legend(handles, SHORT, loc="lower center", ncol=7, frameon=False, fontsize=6.5, handlelength=1.0, columnspacing=1.1, bbox_to_anchor=(0.5, 0.2 / h))
    fig.text(0.5, 0.02 / h, "* sent to refinement     green outline: fixed by refinement     red outline: wrong class", ha="center", va="bottom", fontsize=6.3, color="#444")
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
    labels = run_labels()
    rows = summaries(ds, labels)
    n_all = len(rows)
    print(f"{n_all} frames: all right after refinement {sum(r['refined_wrong'] == 0 for r in rows)} ({100 * sum(r['refined_wrong'] == 0 for r in rows) / n_all:.1f}%), "
          f"all right in the single pass {sum(r['single_wrong'] == 0 for r in rows)} ({100 * sum(r['single_wrong'] == 0 for r in rows) / n_all:.1f}%)")
    if args.cmd == "candidates":
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "frames.json").write_text(json.dumps(rows, indent=1))
        fixed = [r for r in rows if r["fixed"] >= 1 and r["refined_wrong"] == 0 and 2 <= r["n"] <= 4]
        clean = [r for r in rows if r["single_wrong"] == 0 and r["sent"] == 0 and 3 <= r["n"] <= 4]
        fail = [r for r in rows if r["refined_wrong"] >= 1 and 2 <= r["n"] <= 3]
        print(len(fixed), "fixed and all right;", len(clean), "all right, none refined;", len(fail), "still wrong after refinement")
        for tag, pool in (("fixed", fixed), ("clean", clean), ("fail", fail)):
            for r in pool[:8]:
                print(tag, r["file"], r["n"], r["classes"], "fixed", r["fixed"], "sent", r["sent"], "wrong", r["refined_wrong"])
        render([r["file"] for r in fixed[:3]], ds, labels, args.out, "sheet_fixed")
        render([r["file"] for r in clean[:3]], ds, labels, args.out, "sheet_clean")
        render([r["file"] for r in fail[:3]], ds, labels, args.out, "sheet_fail")
    else:
        render(args.frames, ds, labels, args.out, args.name)
        print("wrote", args.out / f"{args.name}.pdf")


if __name__ == "__main__":
    main()
