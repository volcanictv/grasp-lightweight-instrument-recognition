"""Five-fold cross-validation (ground-truth masks) with the argmax and the frame-decode readouts: instance accuracy and macro-F1 of the single pass and of the revisit (tau gate), per fold and mean over folds.

Per fold: logits and tracks of experiments/gtcv5/cv5_f<k>/final (scripts/campaign_tasks_gtmask.py), the held-out cases cv5_f<k>_test, and the class prior of that fold's training split cv5_f<k>_train for the prior shift.

    python scripts/decode_cv_eval.py --root experiments/gtcv5 --shift -0.5 --out decode_cv5.json
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
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
from frame_decode_eval import decode, fused_distribution
from gtbox_sam_final_eval import load_tracked
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import variance_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tau", type=float, default=1.7e-5)
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    weights = CONFIGS["four"]
    per_fold: dict = {}
    for k in args.folds:
        fo = args.root / f"cv5_f{k}" / "final"
        ds = GraspRegionDataset(args.data_root, f"cv5_f{k}_test", letterbox=True)
        train = np.array([i[3] for i in GraspRegionDataset(args.data_root, f"cv5_f{k}_train", letterbox=True).instances])
        prior = np.bincount(train, minlength=7) / len(train)
        shift = -args.shift * np.log(prior)
        z = np.load(fo / "logits.npz")
        y, done = z["y"], z["done"]
        assert len(y) == len(ds.instances) and (y == np.array([i[3] for i in ds.instances])).all(), f"fold {k}: logits and split are not aligned"
        by_frame: dict[str, list[int]] = defaultdict(list)
        for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
            by_frame[file_name].append(idx)
        alpha = alpha_mix(lambda m: z[f"det_{m}"], weights)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        s1 = variance_scores(alpha)["epistemic"]
        tracked = load_tracked(fo / "tracked")
        flag = (s1 >= args.tau) & done
        prob, n_rev = mu.copy(), 0
        for i in np.where(flag)[0]:
            if int(i) in tracked:
                prob[i] = fused_distribution(tracked[int(i)], weights)
                n_rev += 1
        rec = {"instances": int(done.sum()), "revisited": n_rev, "tracks_missing": int(flag.sum()) - n_rev}
        for tag, p in (("single", mu), ("revisit", prob)):
            for mode, pred in (("argmax", p.argmax(1)), ("decode", decode(p, by_frame, shift))):
                rec[f"{tag}_{mode}"] = {"acc": 100 * float((pred[done] == y[done]).mean()), "F1": 100 * float(f1_score(y[done], pred[done], average="macro", labels=range(7)))}
        per_fold[str(k)] = rec
        print(f"fold {k}: n={rec['instances']} revisited={n_rev} missing={rec['tracks_missing']} | " + " | ".join(f"{t} {v['acc']:.2f}/{v['F1']:.2f}" for t, v in rec.items() if isinstance(v, dict)), flush=True)
    mean = {t: {m: float(np.mean([f[t][m] for f in per_fold.values()])) for m in ("acc", "F1")} for t in next(iter(per_fold.values())) if isinstance(next(iter(per_fold.values()))[t], dict)}
    print("mean over folds", json.dumps({t: {m: round(v, 2) for m, v in d.items()} for t, d in mean.items()}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"shift": args.shift, "tau": args.tau, "per_fold": per_fold, "mean_over_folds": mean}, indent=1))


if __name__ == "__main__":
    main()
