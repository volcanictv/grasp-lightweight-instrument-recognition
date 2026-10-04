"""Offline study of mask post-processing and ensembling on the logit windows of scripts/dump_fold_logits.py (CPU only).

A fixed pipeline of optional steps, applied to the (SAM2, SAM3) mask logits of each instrument:
    combine (weight w on SAM2, 1 - w on SAM3) -> gaussian smoothing of the logits -> threshold at a bias -> clip to the given box ->
    fill holes -> keep the largest component -> polygon simplification -> one-pixel dilation or erosion
The settings are chosen by one pass of coordinate ascent over the stages, in that order, on mean per-instrument mask IoU. To say how much of the
gain is real, the cases of the fold are split in two halves: settings tuned on one half are scored on the other, and the two held-out scores are averaged.

Usage (titanxp or laptop): python scripts/postproc_study.py --parts experiments/postproc/fold1_part0.pkl experiments/postproc/fold1_part1.pkl --out docs/reports/gtbox_sam/postproc_fold1.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    import cv2
except ImportError:  # polygon simplification is skipped without OpenCV
    cv2 = None

STAGES = [("w", [1.0, 0.7, 0.5, 0.3, 0.0]), ("bias", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]), ("clip", [False, True]), ("fill", [False, True]),
          ("largest", [False, True]), ("sigma", [0.0, 0.7, 1.5]), ("poly", [0.0, 1.5, 3.0]), ("morph", [0, 1, -1])]
DEFAULT = {"w": 1.0, "bias": 0.0, "clip": False, "fill": False, "largest": False, "sigma": 0.0, "poly": 0.0, "morph": 0}


def box_mask(row: dict) -> np.ndarray:
    x0, y0, x1, y1 = row["win"]
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    b = row["box"]
    return ((ys >= b[1]) & (ys <= b[3]))[:, None] & ((xs >= b[0]) & (xs <= b[2]))[None, :]


def predict(row: dict, cfg: dict) -> np.ndarray:
    l2 = row["l2"].astype(np.float32)
    logit = l2 if row["l3"] is None or cfg["w"] >= 1.0 else cfg["w"] * l2 + (1 - cfg["w"]) * row["l3"].astype(np.float32)
    if cfg["sigma"] > 0:
        logit = ndimage.gaussian_filter(logit, cfg["sigma"])
    m = logit > cfg["bias"]
    if cfg["clip"]:
        m &= row["_box"]
    if cfg["fill"]:
        m = ndimage.binary_fill_holes(m)
    if cfg["largest"] and m.any():
        lab, n = ndimage.label(m)
        if n > 1:
            m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    if cfg["poly"] > 0 and cv2 is not None and m.any():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = np.zeros(m.shape, np.uint8)
        for c in cs:
            cv2.fillPoly(out, [cv2.approxPolyDP(c, cfg["poly"], True)], 1)
        m = out.astype(bool)
    if cfg["morph"] > 0:
        m = ndimage.binary_dilation(m, iterations=cfg["morph"])
    elif cfg["morph"] < 0:
        m = ndimage.binary_erosion(m, iterations=-cfg["morph"])
    return m


def score(rows: list[dict], cfg: dict) -> float:
    v = []
    for r in rows:
        m = predict(r, cfg)
        u = np.logical_or(m, r["gt"]).sum()
        v.append(np.logical_and(m, r["gt"]).sum() / u if u else 0.0)
    return float(np.mean(v))


def tune(rows: list[dict], has_sam3: bool, log: list | None = None) -> dict:
    cfg = dict(DEFAULT)
    for name, values in STAGES:
        if name == "w" and not has_sam3:
            continue
        if name == "poly" and cv2 is None:
            continue
        best = max(values, key=lambda x: score(rows, {**cfg, name: x}))
        if log is not None:
            log.append({"stage": name, "values": {str(x): round(score(rows, {**cfg, name: x}), 5) for x in values}, "chosen": best})
        cfg[name] = best
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="debug: use the first N instruments")
    args = ap.parse_args()
    rows = [r for p in args.parts for r in pickle.loads(p.read_bytes())]
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        r["_box"] = box_mask(r)
    has_sam3 = all(r["l3"] is not None for r in rows)
    cases = sorted({r["frame"].split("/")[0] for r in rows})
    half = {c: i % 2 for i, c in enumerate(cases)}
    parts = [[r for r in rows if half[r["frame"].split("/")[0]] == h] for h in (0, 1)]
    res: dict = {"instruments": len(rows), "cases": cases, "has_sam3": has_sam3,
                 "sam2_alone": score(rows, DEFAULT), "sam3_alone": score(rows, {**DEFAULT, "w": 0.0}) if has_sam3 else None,
                 "ensemble_plain": score(rows, {**DEFAULT, "w": 0.5}) if has_sam3 else None}
    print({k: v for k, v in res.items() if k.endswith(("alone", "plain"))}, flush=True)
    log: list = []
    full = tune(rows, has_sam3, log)
    res["tuning_steps"] = log
    res["chosen"] = full
    res["tuned_on_all"] = score(rows, full)
    held = []
    for a in (0, 1):
        cfg = tune(parts[a], has_sam3)
        held.append({"tuned_on_half": a, "config": cfg, "held_out_default": score(parts[1 - a], DEFAULT), "held_out_tuned": score(parts[1 - a], cfg)})
    res["half_split"] = held
    res["held_out_gain"] = float(np.mean([h["held_out_tuned"] - h["held_out_default"] for h in held]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("sam2_alone", "sam3_alone", "ensemble_plain", "chosen", "tuned_on_all", "held_out_gain")}, indent=1, default=str))


if __name__ == "__main__":
    main()
