"""EndoVis 2018 (ISINet split, annotation masks, members trained on 2018 + 2017): why an ensemble, and why an evidence-based revisit gate.

Ensemble ablation: every member alone, every pair and triple, and all four, each with its own evidential uncertainty and its own tracks (the member axis of the track logits), single pass and with the revisit at
the same budget (the top share of instruments by that configuration's own uncertainty, default 941/2861 as everywhere in the paper).
Gate ablation: the same four-member evidential ensemble, tracks and fusion, only the signal that orders the instruments changes: evidential S, the largest belief and the belief entropy of the same model, the
softmax and MC-dropout signals of the cross-entropy deep-dropout members (scripts/endovis_gtbox_classifiers_ce_e17.py, scripts/endovis_ce_extract.py), random (5 permutations), plus no revisit and revisit all.
Every variant is read out with plain argmax and with the frame decode (soft duplicate penalties from the 2018 + 2017 training frames). Metrics: accuracy, macro-F1, mIoU, IoU, mcIoU (means over seeds);
paired frame-bootstrap intervals for the main contrasts.

    python scripts/endovis_ablation_eval.py --root experiments/endovis_isinet_e17 --instances ~/endovis_isinet/instances.pkl --dup17 dup17.npz --out endovis_ablation.json
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

from evidential_analyze import LABELS, WEIGHTS
from evidential_seeds_e2e_eval import CONFIGS, ORDER
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

KEYS = ("mIoU", "IoU", "mcIoU")
FRAC = 941 / 2861
PERMS = 5
W4 = CONFIGS["four"]


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + 1e-12)).sum(-1)


def subset_alpha(det: np.ndarray, members: tuple[str, ...]) -> np.ndarray:
    """det[..., member axis (ORDER), classes] -> mean Dirichlet parameters of the chosen members (weights of the paper, renormalized)."""
    w = np.array([W4[m] for m in members])
    w = w / w.sum()
    return sum(wi * alpha_from_logits(det[..., ORDER.index(m), :]) for wi, m in zip(w, members))


def fuse(det_frames: np.ndarray, members: tuple[str, ...]) -> np.ndarray:
    alpha = subset_alpha(det_frames, members)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    comb = (mu.max(axis=1)[:, None] * mu).sum(axis=0)
    return comb / comb.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--dup17", type=Path, required=True)
    ap.add_argument("--masks", type=Path, default=None)
    ap.add_argument("--tracked-name", default="tracked_all_s{seed}.npz")
    ap.add_argument("--frac", type=float, default=FRAC)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--skip-ce", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 4], help="member-subset sizes of the ensemble ablation (1 = each member alone, 4 = the full ensemble; 2 and 3 add pairs and triples)")
    ap.add_argument("--dino-dir", type=Path, default=None, help="directory with val_logits_<backbone>_s<seed>.npy of scripts/endovis_dino_train.py (single-frame references)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    p = pickle.load(open(args.instances, "rb"))
    val, train = p["val"], p["train"]
    NAMES = ["BF", "PF", "LND", "MCS", "UP", "SI", "CA"]
    has, dup = np.zeros(7), np.zeros(7)
    trf = collections.defaultdict(list)
    for i in train:
        trf[(i["seq"], i["frame"])].append(i["cls"])
    for v in trf.values():
        for c, n in collections.Counter(v).items():
            has[c] += 1
            dup[c] += n > 1
    d17 = np.load(args.dup17)
    pen = np.log((dup + d17["dup"] + 1) / (has + d17["has"] + 2))   # soft duplicate penalties from the 2018 + 2017 training frames
    fr: dict = collections.defaultdict(list)
    for idx, r in enumerate(val):
        fr[(r["seq"], r["frame"])].append(idx)
    frames = list(fr.values())
    y = np.array([r["cls"] for r in val])
    mask_of = {i: mask_codec.decode(r["rle"]).astype(bool) for i, r in enumerate(val)}   # decoded once; every scoring reuses them
    gts = {}
    for key, ix in fr.items():
        shape = tuple(val[ix[0]]["rle"]["size"])
        gts[key] = paint(shape, [(mask_of[i], int(y[i]) + 1, 0.0) for i in ix])
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

    def score(prob: np.ndarray, readout: str, with_iou: bool = True) -> dict:
        pred = prob.argmax(1) if readout == "argmax" else decode(prob)
        rec = {"acc": 100 * float((pred == y).mean()), "F1": 100 * float(f1_score(y, pred, average="macro", labels=range(7)))}
        if with_iou:
            res = []
            for key, ix in fr.items():
                shape = tuple(val[ix[0]]["rle"]["size"])
                res.append(frame_class_ious(paint(shape, [(mask_of[i], int(pred[i]) + 1, 1.0) for i in ix]), gts[key]))
            rec["res"] = res
        return rec

    def metrics(res: list) -> dict:
        a = aggregate(res)
        return {k: 100 * float(a[k]) for k in KEYS}

    per: dict = {}      # (seed, variant, readout) -> record
    curves: dict = {}   # (seed, signal) -> {share: (acc, F1)} with argmax readout

    members_all = tuple(ORDER)
    configs = {}
    for k in sorted(set(args.sizes)):
        for sub in itertools.combinations(members_all, k):
            configs["+".join(sub)] = sub

    for seed in args.seeds:
        z = np.load(args.root / f"single_ens_s{seed}.npz")
        assert (z["y"] == y).all()
        det = np.stack([z[f"det_{m}"] for m in ORDER], axis=1)                      # (N, members, 7)
        t = np.load(args.root / args.tracked_name.format(seed=seed))
        tracks = {int(k[4:]): t[k] for k in t.files if k.startswith("det_")}
        n_budget = int(round(args.frac * len(y)))
        print(f"seed {seed}: {len(tracks)} tracked instruments, budget {n_budget}", flush=True)

        def revisit(sub: tuple[str, ...], order: np.ndarray, n: int) -> np.ndarray:
            alpha = subset_alpha(det, sub)
            prob = alpha / alpha.sum(axis=1, keepdims=True)
            for i in order[:n]:
                if int(i) in tracks:
                    prob[i] = fuse(tracks[int(i)], sub)
            return prob

        # ---- ensemble ablation
        for name, sub in configs.items():
            alpha = subset_alpha(det, sub)
            s1 = variance_scores(alpha)["epistemic"]
            order = np.argsort(-s1, kind="stable")
            single = alpha / alpha.sum(axis=1, keepdims=True)
            for ro in ("argmax", "decode"):
                per[(seed, f"ens|{name}|single", ro)] = score(single, ro)
                per[(seed, f"ens|{name}|revisit", ro)] = score(revisit(sub, order, n_budget), ro)
            print(f"  ensemble {name}: done", flush=True)

        # ---- single-frame DINO references (trained on the same crops, one run)
        if args.dino_dir is not None:
            for backbone, label in (("dinov2-base", "DINOv2 ViT-B/14"), ("dinov3-base", "DINOv3 ViT-B/16")):
                f = args.dino_dir / f"val_logits_{backbone}_s{seed}.npy"
                if f.exists():
                    for ro in ("argmax", "decode"):
                        per[(seed, f"ref|{label}", ro)] = score(softmax(np.load(f), axis=1), ro)

        # ---- gate ablation (the four-member evidential ensemble's tracks and fusion; only the ordering signal changes)
        alpha4 = subset_alpha(det, members_all)
        mu4 = alpha4 / alpha4.sum(axis=1, keepdims=True)
        signals = {"evidential S": variance_scores(alpha4)["epistemic"], "largest belief (same model)": 1 - mu4.max(1), "belief entropy (same model)": entropy(mu4)}
        ce_path = args.root / f"ce_val_s{seed}.npz"
        if ce_path.exists() and not args.skip_ce:
            ce = np.load(ce_path)
            assert (ce["y"] == y).all()
            p_soft = sum(w * softmax(ce[f"det_{m}"], axis=1) for w, m in zip(WEIGHTS, LABELS))
            mc = [softmax(ce[f"mc_{m}"], axis=2) for m in LABELS]
            votes = np.concatenate([x.argmax(axis=2) for x in mc], axis=0)
            vshare = np.stack([(votes == c).sum(axis=0) for c in range(7)], axis=1).max(axis=1) / votes.shape[0]
            p_mc = sum(w * x.mean(axis=0) for w, x in zip(WEIGHTS, mc))
            signals["softmax max-prob (CE)"] = 1 - p_soft.max(1)
            signals["softmax entropy (CE)"] = entropy(p_soft)
            signals["MC dropout vote disagreement"] = 1 - vshare
            signals["MC dropout mutual information"] = entropy(p_mc) - sum(w * entropy(x).mean(axis=0) for w, x in zip(WEIGHTS, mc))
        usable = np.array([i in tracks for i in range(len(y))])
        pool = np.where(usable)[0]
        n_gate = min(n_budget, len(pool))
        none = alpha4 / alpha4.sum(axis=1, keepdims=True)
        for ro in ("argmax", "decode"):
            per[(seed, "gate|none", ro)] = score(none, ro)
            per[(seed, "gate|all", ro)] = score(revisit(members_all, pool, len(pool)), ro)
        for name, sig in signals.items():
            order = pool[np.argsort(-sig[pool], kind="stable")]
            for ro in ("argmax", "decode"):
                per[(seed, f"gate|{name}", ro)] = score(revisit(members_all, order, n_gate), ro)
            curves[(seed, name)] = {}
            for share in (0.1, 0.2, 0.329, 0.45, 0.6, 1.0):
                pr = revisit(members_all, order, int(round(share * len(pool))))
                curves[(seed, name)][share] = score(pr, "argmax", with_iou=False)
        for pi in range(PERMS):
            order = rng.permutation(pool)
            for ro in ("argmax", "decode"):
                per[(seed, f"gate|random{pi}", ro)] = score(revisit(members_all, order, n_gate), ro)
            curves[(seed, f"random{pi}")] = {share: score(revisit(members_all, order, int(round(share * len(pool)))), "argmax", with_iou=False) for share in (0.1, 0.2, 0.329, 0.45, 0.6, 1.0)}
        print(f"seed {seed}: gate variants done", flush=True)

    def mean_rows(variants: list[str], ro: str) -> dict:
        rows = []
        for s in args.seeds:
            for v in variants:
                if (s, v, ro) in per:
                    r = per[(s, v, ro)]
                    rows.append({**metrics(r["res"]), "acc": r["acc"], "F1": r["F1"]})
        return {k: float(np.mean([r[k] for r in rows])) for k in (*KEYS, "acc", "F1")} if rows else {}

    report: dict = {"ensemble": {}, "gate": {}, "contrasts": {}, "curves": {}}
    print("\n=== ENSEMBLE ABLATION (mean of seeds): rows = members, columns = readout x {single pass, revisit}")
    print(f"{'members':58s} {'readout':7s} " + "  ".join(f"{k:>6}" for k in (*KEYS, 'acc')) + "   |  revisit: " + "  ".join(f"{k:>6}" for k in (*KEYS, 'acc')))
    for name in configs:
        for ro in ("argmax", "decode"):
            a, b = mean_rows([f"ens|{name}|single"], ro), mean_rows([f"ens|{name}|revisit"], ro)
            report["ensemble"][f"{name}|{ro}"] = {"single": a, "revisit": b}
            print(f"{name:58s} {ro:7s} " + "  ".join(f"{a[k]:6.2f}" for k in (*KEYS, 'acc')) + "   |            " + "  ".join(f"{b[k]:6.2f}" for k in (*KEYS, 'acc')))
    print("\n=== SINGLE-FRAME REFERENCES (single pass, same crops and data; one run per seed)")
    for label in ("DINOv2 ViT-B/14", "DINOv3 ViT-B/16"):
        for ro in ("argmax", "decode"):
            r = mean_rows([f"ref|{label}"], ro)
            if r:
                report["ensemble"][f"ref {label}|{ro}"] = {"single": r}
                print(f"{label:58s} {ro:7s} " + "  ".join(f"{r[k]:6.2f}" for k in (*KEYS, 'acc')))
    gate_names = ["gate|none", *[f"gate|{n}" for n in ([s for (sd, s) in curves if sd == args.seeds[0] and not s.startswith('random')])], "gate|random", "gate|all"]
    print("\n=== GATE ABLATION (same tracks and fusion; budget = top share by the signal)")
    print(f"{'gate':36s} {'readout':7s} " + "  ".join(f"{k:>6}" for k in (*KEYS, 'acc', 'F1')))
    for gname in gate_names:
        variants = [f"gate|random{i}" for i in range(PERMS)] if gname == "gate|random" else [gname]
        for ro in ("argmax", "decode"):
            r = mean_rows(variants, ro)
            report["gate"][f"{gname[5:]}|{ro}"] = r
            print(f"{gname[5:]:36s} {ro:7s} " + "  ".join(f"{r[k]:6.2f}" for k in (*KEYS, 'acc', 'F1')))

    # paired frame bootstrap of the main contrasts
    n = len(frames)
    boots = [rng.integers(0, n, n) for _ in range(args.boots)]

    def delta(a_vars: list[str], b_vars: list[str], ro: str, sample) -> dict:
        out = {k: [] for k in KEYS}
        for s in args.seeds:
            for va in a_vars:
                for vb in b_vars:
                    if (s, va, ro) not in per or (s, vb, ro) not in per:
                        continue
                    ra, rb = per[(s, va, ro)]["res"], per[(s, vb, ro)]["res"]
                    if sample is not None:
                        ra, rb = [ra[i] for i in sample], [rb[i] for i in sample]
                    ma, mb = metrics(ra), metrics(rb)
                    for k in KEYS:
                        out[k].append(ma[k] - mb[k])
        return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}

    four = "+".join(members_all)
    # the best single member (by mean mIoU of its revisit, argmax readout) is picked from the data and only used for this contrast
    best_revisit = max(ORDER, key=lambda m: mean_rows([f"ens|{m}|revisit"], "argmax").get("mIoU", -1))
    best_single = max(ORDER, key=lambda m: mean_rows([f"ens|{m}|single"], "argmax").get("mIoU", -1))
    report["best_single_member"] = {"single_pass": best_single, "with_revisit": best_revisit}
    contrasts = {
        "revisit(S) - none": (["gate|evidential S"], ["gate|none"]),
        "ensemble(4) revisit - best single member with revisit": ([f"ens|{four}|revisit"], [f"ens|{best_revisit}|revisit"]),
        "ensemble(4) single pass - best single member (single pass)": ([f"ens|{four}|single"], [f"ens|{best_single}|single"]),
        "ensemble(4) revisit - best single member single pass": ([f"ens|{four}|revisit"], [f"ens|{best_single}|single"]),
        "ensemble(4) single pass - DINOv2": ([f"ens|{four}|single"], ["ref|DINOv2 ViT-B/14"]),
        "ensemble(4) single pass - DINOv3": ([f"ens|{four}|single"], ["ref|DINOv3 ViT-B/16"]),
        "ensemble(4) revisit - DINOv2 single pass": ([f"ens|{four}|revisit"], ["ref|DINOv2 ViT-B/14"]),
        "S - random": (["gate|evidential S"], [f"gate|random{i}" for i in range(PERMS)]),
        "S - softmax max-prob": (["gate|evidential S"], ["gate|softmax max-prob (CE)"]),
        "S - MC dropout vote": (["gate|evidential S"], ["gate|MC dropout vote disagreement"]),
        "S - revisit all": (["gate|evidential S"], ["gate|all"]),
    }
    print(f"\nbest single member (argmax mIoU): single pass {best_single}, with revisit {best_revisit}")
    print("\ncontrast (mean difference over seeds; 95% frame-bootstrap interval)")
    for name, (a, b) in contrasts.items():
        for ro in ("argmax", "decode"):
            point = delta(a, b, ro, None)
            if np.isnan(point["mIoU"]):
                continue
            draws = [delta(a, b, ro, s) for s in boots]
            report["contrasts"][f"{name}|{ro}"] = {k: {"diff": point[k], "lo": float(np.percentile([d[k] for d in draws], 2.5)), "hi": float(np.percentile([d[k] for d in draws], 97.5))} for k in KEYS}
            print(f"{name:52s} {ro:7s} " + "  ".join(f"{k} {point[k]:+.2f} [{report['contrasts'][f'{name}|{ro}'][k]['lo']:+.2f}, {report['contrasts'][f'{name}|{ro}'][k]['hi']:+.2f}]" for k in KEYS), flush=True)
    print("\n=== accuracy / macro-F1 versus share revisited (argmax readout, mean of seeds)")
    names = sorted({s for (_sd, s) in curves if not s.startswith("random")}) + ["random"]
    shares = (0.1, 0.2, 0.329, 0.45, 0.6, 1.0)
    for name in names:
        row = []
        for sh in shares:
            vals = [curves[(s, f"random{i}" if name == "random" else name)][sh] for s in args.seeds for i in (range(PERMS) if name == "random" else [0])]
            row.append((float(np.mean([v["acc"] for v in vals])), float(np.mean([v["F1"] for v in vals]))))
        report["curves"][name] = {str(sh): {"acc": a, "F1": f} for sh, (a, f) in zip(shares, row)}
        print(f"{name:34s} " + "  ".join(f"{int(100 * sh):3d}%: {a:5.2f}/{f:5.2f}" for sh, (a, f) in zip(shares, row)))
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
