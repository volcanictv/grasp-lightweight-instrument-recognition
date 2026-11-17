"""Exploratory: can a learned second stage or a class-prior adjustment raise mcIoU of the gated pipeline? (no training of any vision model)

Protocol, fixed before any test number is read:
  Training rows: the official-training-case instruments of the cross-validation fold caches (models that never saw those cases). Each row has the pipeline's final belief
  vector (the fused belief where the budget-matched gate (top 32.9% by S_1 within the fold) refines the instrument and has a track, the keyframe belief otherwise), the keyframe
  belief, S_1, ensemble disagreement, box size and shape, and frame context (the number of instruments in the frame and the summed final beliefs of the other instruments).
  Two models: (a) prior   a per-class log-bias added to the log final belief, fitted by coordinate ascent on macro-F1 of the argmax;
              (b) stack   multinomial logistic regression (C = 1, standardized features) on all the features above.
  Hyperparameters are fixed here; nothing is tuned on test. A case-grouped cross-validation on the training rows is printed first.
  Test: official-split final models, three seeds, budget-matched gate (top 941 by S_1). Labels from each model replace the pipeline labels; masks are unchanged. The labels are
  written in TAPIS's prediction format so that TAPIS's evaluator scores mIoU, IoU and mcIoU (scripts are run separately).

    python scripts/second_stage_mciou.py --cv-dir ~/cv_gate --final-dir experiments/gtbox_sam/final --out-dir <dir>
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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from adaptive_depth import TEST_CASES, WEIGHTS, load_tracks
from evidential_seeds_e2e_eval import ORDER, alpha_mix
from gtbox_sam_final_eval import alpha_from_logits
from surgical_ai.evaluation.evidential import variance_scores

GATE_FRACTION = 941 / 2861


def fuse_vec(det: np.ndarray) -> np.ndarray:
    al = sum(w * alpha_from_logits(det[:, ORDER.index(k), :]) for k, w in WEIGHTS.items())
    m = al / al.sum(axis=1, keepdims=True)
    c = (m.max(axis=1)[:, None] * m).sum(axis=0)
    return c / c.sum()


def build(z, boxes: np.ndarray, frames: np.ndarray, tracks: dict, gate_fraction: float | int):
    n = len(z["y"])
    alpha = alpha_mix(lambda k: z[f"det_{k}"], WEIGHTS)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    s1 = variance_scores(alpha)["epistemic"]
    k = int(gate_fraction) if gate_fraction > 1 else int(round(gate_fraction * n))
    gate = np.zeros(n, bool)
    gate[np.argsort(-s1, kind="stable")[:k]] = True
    final = mu.copy()
    for i in np.where(gate)[0]:
        if int(i) in tracks:
            final[i] = fuse_vec(tracks[int(i)][0])
    members = np.stack([z[f"det_{kk}"].argmax(1) for kk in ORDER], axis=1)
    disagree = (members != mu.argmax(1)[:, None]).mean(1)
    ent = -(final * np.log(final + 1e-12)).sum(1)
    area, aspect = np.log(boxes[:, 2] * boxes[:, 3] + 1.0), np.log((boxes[:, 2] + 1.0) / (boxes[:, 3] + 1.0))
    by_frame = defaultdict(list)
    for i, f in enumerate(frames):
        by_frame[f].append(i)
    ctx, cnt = np.zeros((n, 7)), np.zeros(n)
    for idx in by_frame.values():
        tot = final[idx].sum(axis=0)
        for i in idx:
            others = tot - final[i]
            ctx[i] = others / max(len(idx) - 1, 1)
            cnt[i] = len(idx)
    feats = np.column_stack([final, mu, np.log(s1 + 1e-12), disagree, ent, area, aspect, cnt, ctx, gate.astype(float)])
    return feats, final, gate


def fit_prior(logp: np.ndarray, y: np.ndarray, sweeps: int = 4) -> np.ndarray:
    b = np.zeros(7)
    grid = np.linspace(-1.5, 1.5, 31)
    best = f1_score(y, (logp + b).argmax(1), average="macro", labels=range(7))
    for _ in range(sweeps):
        for c in range(7):
            for v in grid:
                trial = b.copy()
                trial[c] = v
                sc = f1_score(y, (logp + trial).argmax(1), average="macro", labels=range(7))
                if sc > best + 1e-9:
                    best, b = sc, trial
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-dir", type=Path, required=True)
    ap.add_argument("--final-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(os.environ["GRASP_DATA_ROOT"])
    os.environ.setdefault("GRASP_EXTRA_SPLITS", str(data / "annotations" / "cv5_splits.json"))
    from surgical_ai.data.region_dataset import GraspRegionDataset

    X, Y, G, FIN = [], [], [], []
    for k in range(5):
        fd = args.cv_dir / f"cv5_f{k}" / "final"
        z = np.load(fd / "logits.npz")
        ds = GraspRegionDataset(data, f"cv5_f{k}_test", letterbox=True)
        cases = np.array([f.split("/")[0] for f, _s, _b, _l in ds.instances])
        frames = np.array([f for f, _s, _b, _l in ds.instances])
        boxes = np.array([b for _f, _s, b, _l in ds.instances], float)
        feats, final, _gate = build(z, boxes, frames, load_tracks([fd / "tracked"]), GATE_FRACTION)
        m = ~np.isin(cases, sorted(TEST_CASES)) & z["done"]
        X.append(feats[m]); Y.append(z["y"][m]); G.append(cases[m]); FIN.append(final[m])
    X, Y, G, FIN = np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(FIN)
    print(f"training rows {len(Y)} from {len(set(G))} cases; class counts {np.bincount(Y, minlength=7).tolist()}")
    mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    logp = np.log(FIN + 1e-9)

    def macro(yt, yp):
        return f1_score(yt, yp, average="macro", labels=range(7))

    print("case-grouped cross-validation on the training rows (macro-F1 / accuracy):")
    cases_u = sorted(set(G))
    rep = {"base": [], "prior": [], "stack": []}
    for held in np.array_split(cases_u, 4):
        te = np.isin(G, held)
        tr = ~te
        b = fit_prior(logp[tr], Y[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000).fit((X[tr] - mu_) / sd_, Y[tr])
        for tag, pred in (("base", FIN[te].argmax(1)), ("prior", (logp[te] + b).argmax(1)), ("stack", clf.predict((X[te] - mu_) / sd_))):
            rep[tag].append((macro(Y[te], pred), float((pred == Y[te]).mean())))
    for tag, v in rep.items():
        print(f"  {tag:6s} macro-F1 {np.mean([a for a, _ in v]):.4f}  accuracy {np.mean([c for _, c in v]):.4f}")

    # variant choice by the same case-grouped cross-validation on the training rows only (macro-F1); the test split plays no part in it
    variants = [(C, w) for C in (0.3, 1.0, 3.0) for w in (None, "balanced")]
    cvscore = {}
    for C, w in variants:
        sc = []
        for held in np.array_split(cases_u, 4):
            te = np.isin(G, held)
            m = LogisticRegression(C=C, class_weight=w, max_iter=3000).fit((X[~te] - mu_) / sd_, Y[~te])
            sc.append(macro(Y[te], m.predict((X[te] - mu_) / sd_)))
        cvscore[(C, w)] = float(np.mean(sc))
        print(f"  variant C={C} class_weight={w}: grouped-CV macro-F1 {cvscore[(C, w)]:.4f}")
    best = max(cvscore, key=cvscore.get)
    print("chosen variant", best)
    clf2 = LogisticRegression(C=best[0], class_weight=best[1], max_iter=3000).fit((X - mu_) / sd_, Y)
    bias = fit_prior(logp, Y)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit((X - mu_) / sd_, Y)
    print("fitted class bias", np.round(bias, 2).tolist())

    test_ds = GraspRegionDataset(data, "test", letterbox=True)
    frames_t = np.array([f for f, _s, _b, _l in test_ds.instances])
    boxes_t = np.array([b for _f, _s, b, _l in test_ds.instances], float)
    out = {"bias": bias.tolist(), "seeds": {}}
    for seed in (42, 43, 44):
        z = np.load(args.final_dir / f"logits_s{seed}.npz")
        feats, final, gate = build(z, boxes_t, frames_t, load_tracks([args.final_dir / f"tracked_all_s{seed}"]), 941)
        y = z["y"]
        preds = {"base": final.argmax(1), "prior": (np.log(final + 1e-9) + bias).argmax(1), "stack": clf.predict((feats - mu_) / sd_), "stack_cv": clf2.predict((feats - mu_) / sd_)}
        proba = {"base": final, "prior": None, "stack": clf.predict_proba((feats - mu_) / sd_), "stack_cv": clf2.predict_proba((feats - mu_) / sd_)}
        row = {}
        for tag, p in preds.items():
            row[tag] = {"accuracy": float((p == y).mean()), "macro_f1": float(macro(y, p)), "changed_vs_base": int((p != preds["base"]).sum())}
            print(f"seed {seed} {tag:6s} accuracy {row[tag]['accuracy']:.4f} macro-F1 {row[tag]['macro_f1']:.4f} changed {row[tag]['changed_vs_base']}")
        out["seeds"][str(seed)] = row
        np.savez(args.out_dir / f"labels_s{seed}.npz", **{f"pred_{t}": p for t, p in preds.items()}, **{f"vec_{t}": (v if v is not None else final) for t, v in proba.items()}, y=y)
    (args.out_dir / "second_stage_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
