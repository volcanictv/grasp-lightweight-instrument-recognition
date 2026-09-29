"""Final evidential pipeline on EndoVis 2018 after SAM2 tracking of the flagged
instances, for gates (A) fixed GraSP tau and (B) budget-matched to the MC
pipeline (docs/DECISIONS.md 2026-09-29). Non-flagged instances keep the
evidential base prediction; flagged ones take the pre-registered evidential
frame-weighted combine (evidential_gate_final_eval.evidential_track_predict).

Usage:
    python scripts/endovis_evidential_final_eval.py --dir experiments_endovis/evid_stage1_2018 \
        --mc-final docs/reports/endovis_transfer/stage1_2018/final_pipeline.json \
        --out experiments_endovis/evid_stage1_2018/final.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.metrics import f1_score

from evidential_gate_final_eval import WEIGHTS, evidential_track_predict
from surgical_ai.evaluation.evidential import alpha_from_logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--mc-final", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    e = np.load(args.dir / "extract.npz", allow_pickle=True)
    y = e["y"]
    alpha = sum(w * alpha_from_logits(e[f"det_{k}"]) for k, w in WEIGHTS.items())
    base = alpha.argmax(axis=1)
    present = sorted(set(y.tolist()))
    sets = json.loads((args.dir / "flagged_union.json").read_text())

    tracked = {}
    for shard in (0, 1):
        p = args.dir / f"frames_shard{shard}.npz"
        if p.exists():
            fd = np.load(p)
            for key in fd.files:
                tracked[int(key[4:])] = evidential_track_predict(fd[key])
    out = {"n": int(len(y)), "tracked_available": len(tracked),
           "base": {"accuracy": float((base == y).mean()),
                    "macro_f1_present": float(f1_score(y, base, labels=present, average="macro", zero_division=0))}}
    for name, key in (("gate_A_fixed_tau", "gate_A"), ("gate_B_budget_matched", "gate_B")):
        idxs = sets[key]
        missing = [i for i in idxs if i not in tracked]
        pred = base.copy()
        for i in idxs:
            if i in tracked:
                pred[i] = tracked[i]
        flag = np.zeros(len(y), dtype=bool)
        flag[idxs] = True
        out[name] = {
            "n_flagged": len(idxs), "missing_tracking": len(missing),
            "accuracy": float((pred == y).mean()),
            "macro_f1_present": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
            "fixed": int(((base != y) & (pred == y) & flag).sum()),
            "broken": int(((base == y) & (pred != y) & flag).sum()),
            "errors_left": int((pred != y).sum()),
        }
    if args.mc_final and args.mc_final.exists():
        mc = json.loads(args.mc_final.read_text())
        out["mc_pipeline_for_comparison"] = {k: mc.get(k) for k in ("weighted_pred", "baseline", "n_tracked")}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
