"""Scores one causal tracker on the official test with locked settings (docs/reports/causal_tracker_preregistration.md).

Usage:
    python scripts/score_causal_official.py --name edgetam --dir experiments/official_causal/edgetam --k 20 --tau 0.000285 --out results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score

import evidential_three_member_yolo as t
from compare_trackers_causal import load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--extract", type=Path, default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_official_s42.npz")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "evidential" / "ens_E_official_s42_lam0p01a10.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    order = t.member_order(args.config)
    y, base, s1 = t.base_scores(args.extract)
    flag = s1 >= args.tau
    tr = load(args.dir, prefix="official")
    missing = [int(i) for i in np.where(flag)[0] if int(i) not in tr]
    if missing:
        raise SystemExit(f"{len(missing)} gated instances have no tracking (e.g. {missing[:5]})")
    tracked = {}
    for i in np.where(flag)[0]:
        logits, c = tr[int(i)]
        tracked[int(i)] = t.track_predict(logits[max(0, c - args.k):c + 1], order)
    pred = t.apply_gate(base, tracked, flag)
    names = ["Bipolar", "Prograsp", "LargeNeedle", "MonoScissors", "Suction", "ClipApplier", "Grasper"]
    f = f1_score(y, pred, average=None, labels=range(7))
    f_base = f1_score(y, base, average=None, labels=range(7))
    result = {"tracker": args.name, "k": args.k, "tau": args.tau, "n": int(len(y)),
              "base": {"accuracy": float((base == y).mean()), "macro_f1": float(f_base.mean()), "per_class_f1": dict(zip(names, f_base.round(4).tolist()))},
              "tracked": t.row(y, base, pred, flag, threshold=args.tau),
              "per_class_f1": dict(zip(names, f.round(4).tolist())),
              "mean_frames_used": float(np.mean([min(args.k, tr[int(i)][1]) + 1 for i in np.where(flag)[0]]))}
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
