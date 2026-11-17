"""Second-stage variants for mcIoU (exploratory; extends scripts/second_stage_mciou.py).

Candidates, all fitted on the out-of-fold rows of the official-training cases of the cross-validation fold caches and chosen by case-grouped cross-validation macro-F1 on those rows
(the test split plays no part in the choice):
  lr_rich     multinomial logistic regression on the previous features plus per-track temporal features (agreement of neighbouring frames with the fused and the keyframe label,
              mean and spread of the per-frame belief, mean per-frame belief vector, fused-minus-keyframe belief of the fused label)
  gbm_rich    gradient-boosted trees (depth 3, 150 iterations, learning rate 0.05) on the same features
  lr_spec     lr_rich, then a specialist for the confusable family (Prograsp, Suction, Clip Applier, Grasper): for instances the first stage puts in the family, a 4-class model trained on
              rows whose true class is in the family chooses among those four
Test labels of every candidate are written for scoring with TAPIS's evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from adaptive_depth import TEST_CASES, WEIGHTS, load_tracks
from evidential_seeds_e2e_eval import ORDER, alpha_mix
from gtbox_sam_final_eval import alpha_from_logits
from surgical_ai.evaluation.evidential import variance_scores

FAMILY = [1, 4, 5, 6]


def frame_beliefs(det: np.ndarray) -> np.ndarray:
    al = sum(w * alpha_from_logits(det[:, ORDER.index(k), :]) for k, w in WEIGHTS.items())
    return al / al.sum(axis=1, keepdims=True)


def build(z, boxes, frames, tracks, gate_count):
    n = len(z["y"])
    alpha = alpha_mix(lambda k: z[f"det_{k}"], WEIGHTS)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    s1 = variance_scores(alpha)["epistemic"]
    gate = np.zeros(n, bool)
    gate[np.argsort(-s1, kind="stable")[:gate_count]] = True
    final, temporal = mu.copy(), np.zeros((n, 7 + 8))
    for i in np.where(gate)[0]:
        if int(i) not in tracks:
            continue
        det, center, _off = tracks[int(i)]
        m = frame_beliefs(det)
        comb = (m.max(axis=1)[:, None] * m).sum(axis=0)
        comb = comb / comb.sum()
        final[i] = comb
        lab, key = comb.argmax(), mu[i].argmax()
        others = [j for j in range(len(m)) if j != center]
        am = m.argmax(axis=1)
        agree_f = float(np.mean(am[others] == lab)) if others else 1.0
        agree_k = float(np.mean(am[others] == key)) if others else 1.0
        mm = m.max(axis=1)
        mean_b = m.mean(axis=0)
        temporal[i] = np.concatenate([mean_b, [len(m), agree_f, agree_k, mm.mean(), mm.min(), mm.std(), comb[lab] - mu[i][lab], -(mean_b * np.log(mean_b + 1e-12)).sum()]])
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
            ctx[i] = (tot - final[i]) / max(len(idx) - 1, 1)
            cnt[i] = len(idx)
    feats = np.column_stack([final, mu, np.log(s1 + 1e-12), disagree, ent, area, aspect, cnt, ctx, gate.astype(float), temporal])
    return feats, final, gate


def macro(y, p):
    return f1_score(y, p, average="macro", labels=range(7))


class Candidate:
    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        if self.kind == "gbm_rich":
            self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=0).fit(Z, y)
        else:
            self.m = LogisticRegression(C=1.0, max_iter=4000).fit(Z, y)
        if self.kind == "lr_spec":
            fam = np.isin(y, FAMILY)
            self.spec = LogisticRegression(C=1.0, max_iter=4000).fit(Z[fam], y[fam])
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        p = self.m.predict(Z)
        if self.kind == "lr_spec":
            sel = np.isin(p, FAMILY)
            if sel.any():
                p = p.copy()
                p[sel] = self.spec.predict(Z[sel])
        return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-dir", type=Path, required=True)
    ap.add_argument("--final-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    data = Path(os.environ["GRASP_DATA_ROOT"])
    os.environ.setdefault("GRASP_EXTRA_SPLITS", str(data / "annotations" / "cv5_splits.json"))
    from surgical_ai.data.region_dataset import GraspRegionDataset

    X, Y, G = [], [], []
    for k in range(5):
        fd = args.cv_dir / f"cv5_f{k}" / "final"
        z = np.load(fd / "logits.npz")
        ds = GraspRegionDataset(data, f"cv5_f{k}_test", letterbox=True)
        cases = np.array([f.split("/")[0] for f, _s, _b, _l in ds.instances])
        frames = np.array([f for f, _s, _b, _l in ds.instances])
        boxes = np.array([b for _f, _s, b, _l in ds.instances], float)
        n = len(z["y"])
        feats, _f, _g = build(z, boxes, frames, load_tracks([fd / "tracked"]), int(round(941 / 2861 * n)))
        m = ~np.isin(cases, sorted(TEST_CASES)) & z["done"]
        X.append(feats[m]); Y.append(z["y"][m]); G.append(cases[m])
    X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
    print(f"training rows {len(Y)}, features {X.shape[1]}")
    cases_u = sorted(set(G))
    cv = {}
    for kind in ("lr_rich", "gbm_rich", "lr_spec"):
        sc, acc = [], []
        for held in np.array_split(cases_u, 4):
            te = np.isin(G, held)
            p = Candidate(kind).fit(X[~te], Y[~te]).predict(X[te])
            sc.append(macro(Y[te], p)); acc.append(float((p == Y[te]).mean()))
        cv[kind] = {"macro_f1": float(np.mean(sc)), "accuracy": float(np.mean(acc))}
        print(f"  {kind:9s} grouped-CV macro-F1 {cv[kind]['macro_f1']:.4f} accuracy {cv[kind]['accuracy']:.4f}")
    chosen = max(cv, key=lambda k: cv[k]["macro_f1"])
    print("chosen by grouped CV:", chosen)

    test_ds = GraspRegionDataset(data, "test", letterbox=True)
    frames_t = np.array([f for f, _s, _b, _l in test_ds.instances])
    boxes_t = np.array([b for _f, _s, b, _l in test_ds.instances], float)
    models = {kind: Candidate(kind).fit(X, Y) for kind in ("lr_rich", "gbm_rich", "lr_spec")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"cv": cv, "chosen": chosen, "seeds": {}}
    for seed in (42, 43, 44):
        z = np.load(args.final_dir / f"logits_s{seed}.npz")
        feats, final, gate = build(z, boxes_t, frames_t, load_tracks([args.final_dir / f"tracked_all_s{seed}"]), 941)
        y = z["y"]
        row, arrs = {}, {"y": y}
        for kind, mdl in models.items():
            p = mdl.predict(feats)
            row[kind] = {"accuracy": float((p == y).mean()), "macro_f1": float(macro(y, p))}
            arrs[f"pred_{kind}"] = p
            vec = np.full((len(y), 7), 0.1 / 7)
            vec[np.arange(len(y)), p] = 0.9 + 0.1 / 7
            arrs[f"vec_{kind}"] = vec
            if kind in ("lr_rich", "gbm_rich"):
                arrs[f"proba_{kind}"] = mdl.m.predict_proba((feats - mdl.mu) / mdl.sd)
            print(f"seed {seed} {kind:9s} accuracy {row[kind]['accuracy']:.4f} macro-F1 {row[kind]['macro_f1']:.4f}")
        out["seeds"][str(seed)] = row
        np.savez(args.out_dir / f"labels_v2_s{seed}.npz", **arrs)
    (args.out_dir / "second_stage_v2_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
