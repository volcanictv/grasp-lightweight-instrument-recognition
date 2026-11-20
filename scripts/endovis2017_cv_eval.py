"""EndoVis 2017 four-fold cross-validation (annotation masks given): the CNN ensemble and the evidential DINOv2, single pass and with the uncertainty-gated revisit, scored per fold and averaged over folds.

Per fold (cache from scripts/endovis2017_cv_cache.py, outputs under <root>/fold<k>/): single-pass logits of the CNN ensemble (single_ens.npz) and of the evidential DINOv2 (dino_val.npy), the SAM2 tracks of the union of the
instruments both gate (tracked_ens.npz, dino_tracks.npz). Each method gates the top share (default 941/2861) of the fold's instruments by its own evidential uncertainty and fuses the track frames with confidence
weights, as in the paper. Readouts: plain argmax and the frame decode with soft duplicate penalties from the fold's training frames. Metrics: mIoU, IoU, mcIoU (TAPIS scorer), accuracy, macro-F1; per-class IoU.

    python scripts/endovis2017_cv_eval.py --cache ~/endovis2017_cv --root ~/endovis2017_runs --out endovis2017_cv.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec
from sklearn.metrics import f1_score

from endovis_ablation_eval import fuse as ens_fuse
from endovis_ablation_eval import subset_alpha
from evidential_seeds_e2e_eval import ORDER
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

KEYS = ("mIoU", "IoU", "mcIoU")
FRAC = 941 / 2861
NAMES = ["BF", "PF", "LND", "VS", "GR", "MCS", "Other"]


def run_fold(cache: Path, root: Path, k: int, frac: float) -> dict:
    tab = pickle.loads((cache / f"fold{k}" / "instances.pkl").read_bytes())
    val, train = tab["val"], tab["train"]
    has, dup = np.zeros(7), np.zeros(7)
    trf = collections.defaultdict(list)
    for d in train:
        trf[(d["seq"], d["frame"])].append(d["cls"])
    for v in trf.values():
        for c, n in collections.Counter(v).items():
            has[c] += 1
            dup[c] += n > 1
    pen = np.log((dup + 1) / (has + 2))
    fr: dict = collections.defaultdict(list)
    for idx, r in enumerate(val):
        fr[(r["seq"], r["frame"])].append(idx)
    frames = list(fr.values())
    y = np.array([r["cls"] for r in val])
    mask_of = {i: mask_codec.decode(r["rle"]).astype(bool) for i, r in enumerate(val)}
    gts = {key: paint(tuple(val[ix[0]]["rle"]["size"]), [(mask_of[i], int(y[i]) + 1, 0.0) for i in ix]) for key, ix in fr.items()}

    def decode(P: np.ndarray) -> np.ndarray:
        lp = np.log(P + 1e-9)
        out = lp.argmax(1)
        for ix in frames:
            if len(ix) < 2:
                continue
            best, bs = None, -1e18
            for combo in itertools.product(range(7), repeat=len(ix)):
                s = sum(lp[i, c] for i, c in zip(ix, combo))
                for c, n in collections.Counter(combo).items():
                    if n > 1:
                        s += (n - 1) * pen[c]
                if s > bs:
                    bs, best = s, combo
            for i, c in zip(ix, best):
                out[i] = c
        return out

    def score(prob: np.ndarray, readout: str) -> dict:
        pred = prob.argmax(1) if readout == "argmax" else decode(prob)
        res = [frame_class_ious(paint(tuple(val[ix[0]]["rle"]["size"]), [(mask_of[i], int(pred[i]) + 1, 1.0) for i in ix]), gts[key]) for key, ix in fr.items()]
        a = aggregate(res)
        rec = {k2: 100 * float(a[k2]) for k2 in KEYS}
        rec["acc"] = 100 * float((pred == y).mean())
        rec["F1"] = 100 * float(f1_score(y, pred, average="macro", labels=range(7)))
        pc = a.get("per_class_iou", {})
        rec["per_class_iou"] = {NAMES[c]: (100 * float(pc.get(str(c + 1), pc.get(c + 1, float("nan"))))) for c in range(7)}
        return rec

    n_budget = int(round(frac * len(y)))
    d = root / f"fold{k}"
    # CNN ensemble
    z = np.load(d / "single_ens.npz")
    det = np.stack([z[f"det_{m}"] for m in ORDER], axis=1)
    a4 = subset_alpha(det, tuple(ORDER))
    emu = a4 / a4.sum(1, keepdims=True)
    et = np.load(d / "tracked_ens.npz")
    etr = {int(key[4:]): et[key] for key in et.files if key.startswith("det_")}
    eprob = emu.copy()
    for i in np.argsort(-variance_scores(a4)["epistemic"], kind="stable")[:n_budget]:
        eprob[i] = ens_fuse(etr[int(i)], tuple(ORDER))
    # evidential DINOv2
    lg = np.load(d / "dino_val.npy")
    ad = alpha_from_logits(lg)
    dmu = ad / ad.sum(1, keepdims=True)
    dt = np.load(d / "dino_tracks.npz")
    dtr = {int(key[4:]): dt[key] for key in dt.files if key.startswith("det_")}
    dprob = dmu.copy()
    for i in np.argsort(-variance_scores(ad)["epistemic"], kind="stable")[:n_budget]:
        a = alpha_from_logits(dtr[int(i)])
        m = a / a.sum(1, keepdims=True)
        c = (m.max(1)[:, None] * m).sum(0)
        dprob[i] = c / c.sum()
    variants = {"CNN ensemble, single pass": emu, "CNN ensemble + revisit": eprob, "Evidential DINOv2, single pass": dmu, "Evidential DINOv2 + revisit": dprob}
    out = {"n_val": len(y), "val_per_class": np.bincount(y, minlength=7).tolist(), "variants": {}}
    for name, pr in variants.items():
        out["variants"][name] = {ro: score(pr, ro) for ro in ("argmax", "decode")}
        print(f"fold {k} {name:34s} " + " | ".join(f"{ro}: " + " ".join(f"{out['variants'][name][ro][m]:.2f}" for m in (*KEYS, "acc")) for ro in ("argmax", "decode")), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--frac", type=float, default=FRAC)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    folds = {k: run_fold(args.cache, args.root, k, args.frac) for k in args.folds}
    names = list(next(iter(folds.values()))["variants"])
    report = {"folds": folds, "mean_over_folds": {}}
    print(f"\nmean over folds {args.folds}                 readout   mIoU    IoU  mcIoU    acc     F1")
    for name in names:
        for ro in ("argmax", "decode"):
            r = {m: float(np.mean([folds[k]["variants"][name][ro][m] for k in folds])) for m in (*KEYS, "acc", "F1")}
            report["mean_over_folds"][f"{name}|{ro}"] = r
            print(f"{name:38s} {ro:7s} " + " ".join(f"{r[m]:6.2f}" for m in (*KEYS, "acc", "F1")))
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
