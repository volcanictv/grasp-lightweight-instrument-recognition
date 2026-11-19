"""Temporal window and causality on EndoVis 2018 (annotation masks): the revisit of the top 32.9% most uncertain instruments (evidential ensemble) fuses only the frames of the SAM2 track inside a window.

Symmetric +-k: frames with |offset| <= k. Causal: the keyframe and the k frames before it (-k <= offset <= 0). The offsets of a track's frames are rebuilt from the track file exactly as scripts/endovis_gtbox_logits.py selects them
(the keyframe and every frame whose tracked mask has at least MIN_AREA pixels), so they line up with the saved per-frame logits. Everything else (gate, fusion, readouts, scoring) is as in scripts/endovis_ablation_eval.py.

    python scripts/endovis_window_eval.py --root experiments/endovis_isinet_e17 --instances ~/endovis_isinet/instances.pkl --dup17 dup17.npz --seeds 42 --out window.json
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

import endovis_gtbox_common as C
from endovis_ablation_eval import fuse, subset_alpha
from evidential_seeds_e2e_eval import ORDER
from surgical_ai.evaluation.evidential import variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

KEYS = ("mIoU", "IoU", "mcIoU")
FRAC = 941 / 2861
WINDOWS = (1, 2, 3, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--dup17", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    p = pickle.load(open(args.instances, "rb"))
    val, train = p["val"], p["train"]
    has, dup = np.zeros(7), np.zeros(7)
    trf = collections.defaultdict(list)
    for i in train:
        trf[(i["seq"], i["frame"])].append(i["cls"])
    for v in trf.values():
        for c, n in collections.Counter(v).items():
            has[c] += 1
            dup[c] += n > 1
    d17 = np.load(args.dup17)
    pen = np.log((dup + d17["dup"] + 1) / (has + d17["has"] + 2))
    fr: dict = collections.defaultdict(list)
    for idx, r in enumerate(val):
        fr[(r["seq"], r["frame"])].append(idx)
    frames = list(fr.values())
    y = np.array([r["cls"] for r in val])
    mask_of = {i: mask_codec.decode(r["rle"]).astype(bool) for i, r in enumerate(val)}
    gts = {key: paint(tuple(val[ix[0]]["rle"]["size"]), [(mask_of[i], int(y[i]) + 1, 0.0) for i in ix]) for key, ix in fr.items()}

    # offsets of each track's saved frames
    offsets: dict[int, list[int]] = {}
    for path in sorted((args.root / "tracks_all").glob("tracks_shard*.pkl")):
        for idx, fs in pickle.loads(path.read_bytes()).items():
            offsets[idx] = [off for off in sorted(fs) if off == 0 or mask_codec.area(fs[off][1]) >= C.MIN_AREA]

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
        return {"res": res, "acc": 100 * float((pred == y).mean()), "F1": 100 * float(f1_score(y, pred, average="macro", labels=range(7)))}

    def metrics(res: list) -> dict:
        a = aggregate(res)
        return {k: 100 * float(a[k]) for k in KEYS}

    per: dict = {}
    members = tuple(ORDER)
    n_budget = int(round(FRAC * len(y)))
    for seed in args.seeds:
        z = np.load(args.root / f"single_ens_s{seed}.npz")
        det = np.stack([z[f"det_{m}"] for m in members], axis=1)
        t = np.load(args.root / f"tracked_all_s{seed}.npz")
        tracks = {int(k[4:]): t[k] for k in t.files if k.startswith("det_")}
        alpha = subset_alpha(det, members)
        mu = alpha / alpha.sum(1, keepdims=True)
        order = [int(i) for i in np.argsort(-variance_scores(alpha)["epistemic"], kind="stable")[:n_budget]]
        bad = sum(1 for i in order if i in tracks and len(offsets.get(i, [])) != tracks[i].shape[0])
        print(f"seed {seed}: {len(order)} gated, {bad} tracks whose frame count does not match the saved logits (skipped)", flush=True)
        variants = {"none": mu}
        for mode in ("sym", "causal"):
            for k in WINDOWS:
                prob = mu.copy()
                for i in order:
                    if i not in tracks or len(offsets.get(i, [])) != tracks[i].shape[0]:
                        continue
                    keep = [q for q, off in enumerate(offsets[i]) if (abs(off) <= k if mode == "sym" else -k <= off <= 0)]
                    prob[i] = fuse(tracks[i][keep], members)
                variants[f"{mode} {k}"] = prob
        for name, pr in variants.items():
            for ro in ("argmax", "decode"):
                per[(seed, name, ro)] = score(pr, ro)
        print(f"seed {seed}: scored", flush=True)

    names = ["none"] + [f"{m} {k}" for m in ("sym", "causal") for k in WINDOWS]
    report: dict = {"table": {}, "contrasts": {}}
    print("\nwindow            readout   mIoU    IoU  mcIoU    acc     F1")
    for name in names:
        for ro in ("argmax", "decode"):
            rows = [{**metrics(per[(s, name, ro)]["res"]), "acc": per[(s, name, ro)]["acc"], "F1": per[(s, name, ro)]["F1"]} for s in args.seeds]
            r = {k: float(np.mean([x[k] for x in rows])) for k in (*KEYS, "acc", "F1")}
            report["table"][f"{name}|{ro}"] = r
            print(f"{name:16s} {ro:7s} " + " ".join(f"{r[k]:6.2f}" for k in (*KEYS, "acc", "F1")))
    rng = np.random.default_rng(0)
    n = len(frames)
    boots = [rng.integers(0, n, n) for _ in range(args.boots)]

    def delta(a: str, b: str, ro: str, sample) -> dict:
        out = {k: [] for k in KEYS}
        for s in args.seeds:
            ra, rb = per[(s, a, ro)]["res"], per[(s, b, ro)]["res"]
            if sample is not None:
                ra, rb = [ra[i] for i in sample], [rb[i] for i in sample]
            ma, mb = metrics(ra), metrics(rb)
            for k in KEYS:
                out[k].append(ma[k] - mb[k])
        return {k: float(np.mean(v)) for k, v in out.items()}

    contrasts = {"symmetric +-10 vs none": ("sym 10", "none"), "symmetric +-1 vs none": ("sym 1", "none"), "causal 10 vs symmetric 10": ("causal 10", "sym 10"), "symmetric +-10 vs symmetric +-1": ("sym 10", "sym 1")}
    print("\ncontrast (difference; 95% frame-bootstrap interval)")
    for name, (a, b) in contrasts.items():
        for ro in ("argmax", "decode"):
            point = delta(a, b, ro, None)
            draws = [delta(a, b, ro, s) for s in boots]
            report["contrasts"][f"{name}|{ro}"] = {k: {"diff": point[k], "lo": float(np.percentile([d[k] for d in draws], 2.5)), "hi": float(np.percentile([d[k] for d in draws], 97.5))} for k in KEYS}
            print(f"{name:34s} {ro:7s} " + "  ".join(f"{k} {point[k]:+.2f} [{report['contrasts'][f'{name}|{ro}'][k]['lo']:+.2f}, {report['contrasts'][f'{name}|{ro}'][k]['hi']:+.2f}]" for k in KEYS), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
