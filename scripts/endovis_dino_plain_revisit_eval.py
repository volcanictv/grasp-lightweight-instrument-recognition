"""The revisit on top of a plain fine-tuned DINOv2 (cross-entropy, no evidential anything): EndoVis 2018 (annotation masks), per seed.

The gate is the softmax confidence of the plain model itself (1 - max probability); the per-frame softmax distributions along the SAM2 track are fused with confidence weights, as for the paper's models. Compared, with paired
frame bootstrap, against the plain model's single pass, random and revisit-all gates on the same tracks, and the CNN ensemble's own revisit (recomputed from its logits and tracks).

    python scripts/endovis_dino_plain_revisit_eval.py --root experiments/endovis_isinet_e17 --dino-dir experiments/endovis_isinet_e17/dino_plain --instances ~/endovis_isinet/instances.pkl --dup17 dup17.npz --seed 42 --out out.json
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
from scipy.special import softmax
from sklearn.metrics import f1_score

from endovis_ablation_eval import fuse as ens_fuse
from endovis_ablation_eval import subset_alpha
from evidential_seeds_e2e_eval import ORDER
from surgical_ai.evaluation.evidential import variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

KEYS = ("mIoU", "IoU", "mcIoU")
FRAC = 941 / 2861
PERMS = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dino-dir", type=Path, required=True)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--dup17", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac", type=float, default=FRAC)
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
    rng = np.random.default_rng(0)

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

    root, dino, seed = args.root, args.dino_dir, args.seed
    n_budget = int(round(args.frac * len(y)))
    mu = softmax(np.load(dino / f"val_logits_dinov2-base_s{seed}.npy"), axis=1)
    conf = 1.0 - mu.max(1)                                  # the softmax gate: low confidence = uncertain
    t = np.load(dino / f"dino_plain_tracks_s{seed}.npz")
    tracks = {int(k[4:]): t[k] for k in t.files if k.startswith("det_")}
    fused = {}
    for i, det in tracks.items():
        m = softmax(det, axis=1)
        c = (m.max(1)[:, None] * m).sum(0)
        fused[i] = c / c.sum()
    pool = np.array(sorted(i for i in tracks))
    print(f"seed {seed}: {len(tracks)} tracked instruments, budget {n_budget}", flush=True)

    def revisit(order: np.ndarray, n: int) -> np.ndarray:
        prob = mu.copy()
        for i in order[:n]:
            if int(i) in fused:
                prob[i] = fused[int(i)]
        return prob

    variants = {
        "Plain DINOv2, single pass": mu,
        "Plain DINOv2 + revisit, softmax gate": revisit(pool[np.argsort(-conf[pool], kind="stable")], n_budget),
        "Plain DINOv2 + revisit, revisit all": revisit(pool, len(pool)),
    }
    for pi in range(PERMS):
        variants[f"random{pi}"] = revisit(rng.permutation(pool), n_budget)
    z = np.load(root / f"single_ens_s{seed}.npz")
    det = np.stack([z[f"det_{m}"] for m in ORDER], axis=1)
    et = np.load(root / f"tracked_all_s{seed}.npz")
    etr = {int(k[4:]): et[k] for k in et.files if k.startswith("det_")}
    a4 = subset_alpha(det, tuple(ORDER))
    emu = a4 / a4.sum(1, keepdims=True)
    eprob = emu.copy()
    for i in np.argsort(-variance_scores(a4)["epistemic"], kind="stable")[:n_budget]:
        if int(i) in etr:
            eprob[i] = ens_fuse(etr[int(i)], tuple(ORDER))
    variants["CNN ensemble, single pass"] = emu
    variants["CNN ensemble + revisit, evidence gate"] = eprob

    per = {(name, ro): score(pr, ro) for name, pr in variants.items() for ro in ("argmax", "decode")}

    def mean_of(names: list[str], ro: str) -> dict:
        rows = [{**metrics(per[(n, ro)]["res"]), "acc": per[(n, ro)]["acc"], "F1": per[(n, ro)]["F1"]} for n in names]
        return {k: float(np.mean([r[k] for r in rows])) for k in (*KEYS, "acc", "F1")}

    groups = {k: [k] for k in variants if not k.startswith("random")}
    groups["Plain DINOv2 + revisit, random gate"] = [f"random{i}" for i in range(PERMS)]
    report: dict = {"table": {}, "contrasts": {}}
    print("\nsetup                                              readout   mIoU    IoU  mcIoU    acc     F1")
    for name, names in groups.items():
        for ro in ("argmax", "decode"):
            r = mean_of(names, ro)
            report["table"][f"{name}|{ro}"] = r
            print(f"{name:50s} {ro:7s} " + " ".join(f"{r[k]:6.2f}" for k in (*KEYS, "acc", "F1")))
    n = len(frames)
    boots = [rng.integers(0, n, n) for _ in range(args.boots)]

    def delta(a: list[str], b: list[str], ro: str, sample) -> dict:
        out = {k: [] for k in KEYS}
        for va in a:
            for vb in b:
                ra, rb = per[(va, ro)]["res"], per[(vb, ro)]["res"]
                if sample is not None:
                    ra, rb = [ra[i] for i in sample], [rb[i] for i in sample]
                ma, mb = metrics(ra), metrics(rb)
                for k in KEYS:
                    out[k].append(ma[k] - mb[k])
        return {k: float(np.mean(v)) for k, v in out.items()}

    gate = ["Plain DINOv2 + revisit, softmax gate"]
    contrasts = {
        "plain DINOv2 revisit - its single pass": (gate, ["Plain DINOv2, single pass"]),
        "plain DINOv2 revisit - CNN ensemble revisit": (gate, ["CNN ensemble + revisit, evidence gate"]),
        "softmax gate - random gate (plain DINOv2)": (gate, [f"random{i}" for i in range(PERMS)]),
        "softmax gate - revisit all (plain DINOv2)": (gate, ["Plain DINOv2 + revisit, revisit all"]),
        "plain DINOv2 single pass - CNN ensemble single pass": (["Plain DINOv2, single pass"], ["CNN ensemble, single pass"]),
    }
    print("\ncontrast (difference; 95% frame-bootstrap interval)")
    for name, (a, b) in contrasts.items():
        for ro in ("argmax", "decode"):
            point = delta(a, b, ro, None)
            draws = [delta(a, b, ro, s) for s in boots]
            report["contrasts"][f"{name}|{ro}"] = {k: {"diff": point[k], "lo": float(np.percentile([d[k] for d in draws], 2.5)), "hi": float(np.percentile([d[k] for d in draws], 97.5))} for k in KEYS}
            print(f"{name:56s} {ro:7s} " + "  ".join(f"{k} {point[k]:+.2f} [{report['contrasts'][f'{name}|{ro}'][k]['lo']:+.2f}, {report['contrasts'][f'{name}|{ro}'][k]['hi']:+.2f}]" for k in KEYS), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
