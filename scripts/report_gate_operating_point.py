"""Full metrics of the vote-based pipeline on the official test set at a chosen
gate threshold, from the saved logits cache and SAM2 tracking outputs (no new
compute). Same construction as evaluate_gate_sweep.py: instances with
u >= t take the uncertainty-weighted vote over their tracked frames, the rest keep
the ensemble's vote.

Usage:
    python scripts/report_gate_operating_point.py --threshold 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from build_tracking_sample_b import plurality, vote_share
from evaluate_softmax_free_combine import RULES, track_predictions

CLASS_NAMES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
               "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    ap.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz")
    ap.add_argument("--dirs", type=Path, nargs="+", default=[REPO_ROOT / "docs" / "reports" / "tracking_softmax_free",
                                                              REPO_ROOT / "docs" / "reports" / "tracking_gate_sweep"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members, w320 = cfg["members"], cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(args.logits_cache)
    y = cache["y_true"]
    n_classes = len(CLASS_NAMES)
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    base = plurality(v_mc, v_det)
    u = np.round(1 - v_mc.max(axis=1), 9)

    tracked = {}
    for d in args.dirs:
        for shard in (0, 1):
            path = d / f"frames_shard{shard}.npz"
            if path.exists():
                data = np.load(path)
                for key in data.files:
                    if key.startswith("det_"):
                        i = int(key[4:])
                        tracked[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    flag = u >= args.threshold
    missing = [i for i in np.where(flag)[0] if i not in tracked]
    assert not missing, f"{len(missing)} flagged instances lack tracking data"
    pred = base.copy()
    for i in np.where(flag)[0]:
        pred[i] = tracked[i][RULES[1]]

    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(n_classes)), zero_division=0)
    _, _, f1_base, _ = precision_recall_fscore_support(y, base, labels=list(range(n_classes)), zero_division=0)
    out = {
        "threshold": args.threshold, "n": int(len(y)), "tracked": int(flag.sum()), "share_tracked": float(flag.mean()),
        "accuracy": float((pred == y).mean()), "macro_f1": float(f1.mean()),
        "ensemble_alone_accuracy": float((base == y).mean()), "ensemble_alone_macro_f1": float(f1_base.mean()),
        "fixed": int(((base != y) & (pred == y)).sum()), "broken": int(((base == y) & (pred != y)).sum()),
        "errors_left": int((pred != y).sum()), "errors_passed_by_gate": int(((pred != y) & ~flag).sum()),
        "per_class": {n: {"n": int(sup[i]), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                          "f1_before_tracking": float(f1_base[i])} for i, n in enumerate(CLASS_NAMES)},
        "confusion_counts": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist(),
    }
    print(f"gate {args.threshold:.2f}: {out['tracked']} of {out['n']} tracked ({100 * out['share_tracked']:.1f}%)")
    print(f"ensemble alone: accuracy {out['ensemble_alone_accuracy']:.4f}, macro-F1 {out['ensemble_alone_macro_f1']:.4f}")
    print(f"final pipeline: accuracy {out['accuracy']:.4f}, macro-F1 {out['macro_f1']:.4f}; "
          f"fixed {out['fixed']}, broken {out['broken']}, errors left {out['errors_left']} "
          f"({out['errors_passed_by_gate']} never tracked)")
    print(f"{'class':<28}{'n':>5}{'prec':>8}{'recall':>8}{'F1':>8}{'F1 before':>11}")
    for n, v in out["per_class"].items():
        print(f"{n:<28}{v['n']:>5}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}{v['f1_before_tracking']:>11.3f}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
