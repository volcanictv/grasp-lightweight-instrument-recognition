"""Exploratory: an outcome-supervised refinement gate.

Instead of ranking instruments by uncertainty, rank them by the predicted refinement outcome: P(correction) - P(introduced error), learned from keyframe-only features
(class beliefs, S_1, maximum belief, top-2 margin, entropy, ensemble disagreement, box size and shape).

Training data: the official-training cases of the cross-validation fold caches (models that never saw them): every instrument that has a track, with its outcome under
full-window fusion (+1 correction, -1 introduced error, 0 otherwise). Test: the official-split final models, three seeds, all 2,861 instruments. Two-stage ranking so
that the training and test pools match: the pool is the top half by S_1 (the half the cross-validation runs tracked); inside it instruments are ranked by the learned score,
outside it by S_1. The comparison is the share of refine-all corrections and introduced errors captured at each budget, against ranking by S_1 alone.

    python scripts/utility_gate.py --cv-dir ~/cv_gate --final-dir experiments/gtbox_sam/final --out utility_gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from adaptive_depth import KMAX, TEST_CASES, WEIGHTS, load_tracks
from evidential_seeds_e2e_eval import ORDER, alpha_mix
from gtbox_sam_final_eval import fuse
from surgical_ai.evaluation.evidential import variance_scores

SHARES = [0.05, 0.10, 0.20, 0.329, 0.50]


def keyframe_features(z, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = alpha_mix(lambda kk: z[f"det_{kk}"], WEIGHTS)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    s1 = variance_scores(alpha)["epistemic"]
    srt = np.sort(mu, axis=1)
    ent = -(mu * np.log(mu + 1e-12)).sum(1)
    pred = mu.argmax(1)
    members = np.stack([z[f"det_{k}"].argmax(1) for k in ORDER], axis=1)
    disagree = (members != pred[:, None]).mean(1)
    area, aspect = np.log(boxes[:, 2] * boxes[:, 3] + 1.0), np.log((boxes[:, 2] + 1.0) / (boxes[:, 3] + 1.0))
    feats = np.column_stack([mu, np.log(s1 + 1e-12), srt[:, -1], srt[:, -1] - srt[:, -2], ent, disagree, area, aspect])
    return feats, mu, s1, pred


def outcomes(y, pred, tracks, n):
    delta = np.zeros(n, int)
    valid = np.zeros(n, bool)
    for i, (det, _c, _o) in tracks.items():
        if 0 <= i < n:
            fl = fuse(det, WEIGHTS)[0]
            delta[i] = int(fl == y[i]) - int(pred[i] == y[i])
            valid[i] = True
    return delta, valid


def auroc(score, label):
    r = rankdata(score)
    n1 = int(label.sum())
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(label) - n1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv-dir", type=Path, required=True)
    ap.add_argument("--final-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--order-dir", type=Path, default=None, help="save each seed's full learned ranking (instrument indices, best first) and the outcome arrays")
    args = ap.parse_args()
    data = Path(os.environ["GRASP_DATA_ROOT"])
    os.environ.setdefault("GRASP_EXTRA_SPLITS", str(data / "annotations" / "cv5_splits.json"))
    from surgical_ai.data.region_dataset import GraspRegionDataset

    X, D = [], []
    for k in range(5):
        fd = args.cv_dir / f"cv5_f{k}" / "final"
        z = np.load(fd / "logits.npz")
        ds = GraspRegionDataset(data, f"cv5_f{k}_test", letterbox=True)
        cases = np.array([f.split("/")[0] for f, _s, _b, _l in ds.instances])
        boxes = np.array([b for _f, _s, b, _l in ds.instances], float)
        feats, _mu, _s1, pred = keyframe_features(z, boxes)
        delta, valid = outcomes(z["y"], pred, load_tracks([fd / "tracked"]), len(pred))
        m = valid & ~np.isin(cases, sorted(TEST_CASES))
        X.append(feats[m])
        D.append(delta[m])
    X, D = np.concatenate(X), np.concatenate(D)
    print(f"training instruments {len(X)}: corrections {(D == 1).sum()}, introduced errors {(D == -1).sum()}")
    mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    models = {
        "logistic": lambda: LogisticRegression(C=0.5, max_iter=2000, class_weight=None),
        "boosted trees": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=0),
    }
    test_ds = GraspRegionDataset(data, "test", letterbox=True)
    boxes_t = np.array([b for _f, _s, b, _l in test_ds.instances], float)
    res: dict = {"n_train": int(len(X)), "models": {}}
    for name, make in models.items():
        clf = make().fit((X - mu_) / sd_, D)
        classes = list(clf.classes_)
        per_seed = []
        for seed in (42, 43, 44):
            z = np.load(args.final_dir / f"logits_s{seed}.npz")
            feats, mu, s1, pred = keyframe_features(z, boxes_t)
            delta, valid = outcomes(z["y"], pred, load_tracks([args.final_dir / f"tracked_all_s{seed}"]), len(pred))
            n = len(pred)
            proba = clf.predict_proba((feats - mu_) / sd_)
            util = proba[:, classes.index(1)] - proba[:, classes.index(-1)]
            pool = np.zeros(n, bool)
            pool[np.argsort(-s1, kind="stable")[: n // 2]] = True
            order_s1 = np.argsort(-s1, kind="stable")
            key = np.where(pool, 1e6 + util, s1)  # pool first, ranked by the learned score; the rest by S_1
            order_gate = np.argsort(-key, kind="stable")
            c_all, d_all = (delta == 1).sum(), (delta == -1).sum()
            row = {}
            for tag, order in (("S1", order_s1), ("learned", order_gate)):
                row[tag] = {}
                for sh in SHARES:
                    top = order[: int(round(sh * n))]
                    c, d = int((delta[top] == 1).sum()), int((delta[top] == -1).sum())
                    row[tag][str(sh)] = {"C_share": c / c_all, "D_share": d / d_all, "net": c - d}
            m = pool & valid
            row["auroc_correction_in_pool"] = auroc(util[m], (delta[m] == 1).astype(int))
            row["auroc_S1_correction_in_pool"] = auroc(s1[m], (delta[m] == 1).astype(int))
            if args.order_dir is not None:
                args.order_dir.mkdir(parents=True, exist_ok=True)
                np.savez(args.order_dir / f"{name.replace(' ', '_')}_s{seed}.npz", order=order_gate, order_s1=order_s1, delta=delta, valid=valid,
                         case=np.array([f.split("/")[0] for f, _s, _b, _l in test_ds.instances]))
            per_seed.append(row)
        res["models"][name] = per_seed
        print(name)
        for sh in SHARES:
            for tag in ("S1", "learned"):
                v = [r[tag][str(sh)] for r in per_seed]
                print(f"  budget {sh:5.3f} {tag:8s} corrections captured {100 * np.mean([x['C_share'] for x in v]):5.1f}%  introduced {100 * np.mean([x['D_share'] for x in v]):5.1f}%  net {np.mean([x['net'] for x in v]):6.1f}")
        print("  AUROC for corrections inside the pool: learned", round(float(np.mean([r["auroc_correction_in_pool"] for r in per_seed])), 3), "S1", round(float(np.mean([r["auroc_S1_correction_in_pool"] for r in per_seed])), 3))
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
