"""Box-geometry evidence as one extra term in the frame decode (GraSP official test, ground-truth masks), scored with mIoU / IoU / mcIoU.

A multinomial logistic regression on box geometry only (centre, size, aspect, border contact, number of instruments in the frame and the instrument's rank by position and size) is fitted on the official
training instances. Its evidence log P(c | geometry) - log prior(c) is added with weight lambda to the shipped log-probabilities before the frame decode (prior shift -0.5). The weight was chosen on the two
development folds (lambda 0.25 improved accuracy and macro-F1 in both); the test split is scored once for lambda in {0, 0.25, 0.5}.

    python scripts/geometry_evidence_eval.py --probs-dir /tmp/dv/probs --masks experiments/gtbox_sam/gtmask/masks.pkl --out geometry_evidence.json
where <probs-dir>/test_s<seed>.npz hold the per-instance fused probabilities of the shipped pipeline (scripts/frame_decode_eval.py output: prob, y, done).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from frame_decode_eval import decode
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

W, H = 1280.0, 1024.0


def geometry_features(boxes: list, files: list) -> np.ndarray:
    groups: dict = collections.defaultdict(list)
    for i, f in enumerate(files):
        groups[f].append(i)
    X = np.zeros((len(boxes), 14))
    for ix in groups.values():
        xs = np.array([boxes[i][0] + boxes[i][2] / 2 for i in ix])
        ar = np.array([boxes[i][2] * boxes[i][3] for i in ix])
        for k, i in enumerate(ix):
            x, y, w, h = boxes[i]
            n = len(ix)
            X[i] = [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H, np.log(w * h / (W * H) + 1e-6), np.log((w + 1) / (h + 1)), x <= 3, y <= 3, x + w >= W - 3, y + h >= H - 3, n,
                    (xs < xs[k]).sum() / max(1, n - 1), (ar < ar[k]).sum() / max(1, n - 1), w * h / max(1.0, ar.sum())]
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probs-dir", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    tr = GraspRegionDataset(args.data_root, "train", letterbox=True)
    prior = np.bincount([i[3] for i in tr.instances], minlength=7) / len(tr.instances)
    Xtr = geometry_features([list(i[2]) for i in tr.instances], [i[0] for i in tr.instances])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), [i[3] for i in tr.instances])
    pg = np.clip(clf.predict_proba(sc.transform(geometry_features([list(i[2]) for i in ds.instances], [i[0] for i in ds.instances]))), 1e-4, None)
    evidence = np.log(pg) - np.log(prior)[None, :]
    shift = -args.shift * np.log(prior)
    masks = pickle.loads(args.masks.read_bytes())
    by_frame: dict = collections.defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames = [(f, ix) for f, ix in by_frame.items() if all(i in masks for i in ix)]
    y_all = np.array([int(i[3]) for i in ds.instances])
    gts = {f: paint(tuple(ds.instances[ix[0]][1]["size"]), [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y_all[i]) + 1, 0.0) for i in ix]) for f, ix in frames}
    results: dict = {}
    for seed in args.seeds:
        z = np.load(args.probs_dir / f"test_s{seed}.npz")
        prob, y, done = z["prob"], z["y"], z["done"]
        for lam in args.lambdas:
            pred = decode(prob * np.exp(lam * evidence), by_frame, shift)
            res = [frame_class_ious(paint(tuple(ds.instances[ix[0]][1]["size"]), [(mask_codec.decode(masks[i]).astype(bool), int(pred[i]) + 1, 1.0) for i in ix if done[i]]), gts[f]) for f, ix in frames]
            agg = aggregate(res)
            rec = {k: 100 * float(agg[k]) for k in ("mIoU", "IoU", "mcIoU")}
            rec["acc"] = 100 * float((pred[done] == y[done]).mean())
            rec["F1"] = 100 * float(f1_score(y[done], pred[done], average="macro", labels=range(7)))
            rec["per_class_iou"] = {str(k): 100 * float(v) for k, v in agg.get("per_class_iou", {}).items()}
            results.setdefault(str(lam), {})[str(seed)] = rec
            print(f"seed {seed} lambda {lam}: " + " ".join(f"{k} {rec[k]:.2f}" for k in ("mIoU", "IoU", "mcIoU", "acc", "F1")), flush=True)
    summary = {lam: {k: float(np.mean([r[k] for r in per.values()])) for k in ("mIoU", "IoU", "mcIoU", "acc", "F1")} for lam, per in results.items()}
    print("mean over seeds:")
    for lam, s in summary.items():
        print(f"  lambda {lam}: " + " ".join(f"{k} {v:.2f}" for k, v in s.items()))
    args.out.write_text(json.dumps({"results": results, "mean": summary}, indent=1))


if __name__ == "__main__":
    main()
