"""Paired comparison of the MC pipeline and the evidential pipeline on the same
EndoVis 2018 val instances after tracking (docs/DECISIONS.md 2026-09-29): exact
McNemar test on instances where the two final predictions differ in correctness,
and a per-sequence accuracy table. Run on titanxp next to the experiment files.

Usage:
    python scripts/endovis_evidential_vs_mc_paired.py --out experiments_endovis/evid_stage1_2018/paired.json
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
from scipy.stats import binomtest

from evidential_gate_final_eval import WEIGHTS, evidential_track_predict
from surgical_ai.evaluation.evidential import alpha_from_logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evid-dir", type=Path, default=REPO_ROOT / "experiments_endovis" / "evid_stage1_2018")
    ap.add_argument("--mc-dir", type=Path, default=REPO_ROOT / "experiments_endovis" / "stage1_2018")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    e = np.load(args.evid_dir / "extract.npz", allow_pickle=True)
    y, seq = e["y"], e["seq"]
    alpha = sum(w * alpha_from_logits(e[f"det_{k}"]) for k, w in WEIGHTS.items())
    pred_e = alpha.argmax(axis=1)
    flagged = json.loads((args.evid_dir / "flagged_union.json").read_text())["gate_B"]
    for shard in (0, 1):
        fd = np.load(args.evid_dir / f"frames_shard{shard}.npz")
        for key in fd.files:
            i = int(key[4:])
            if i in set(flagged):
                pred_e[i] = evidential_track_predict(fd[key])

    votes = np.load(args.mc_dir / "votes.npz")
    assert (votes["y"] == y).all()
    pred_c = votes["pred"].copy()
    for shard in (0, 1):
        for r in json.loads((args.mc_dir / f"tracked_shard{shard}.json").read_text()):
            pred_c[r["index"]] = r["weighted_pred"]

    ok_e, ok_c = pred_e == y, pred_c == y
    a, b = int((ok_e & ~ok_c).sum()), int((ok_c & ~ok_e).sum())
    p = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
    out = {"n": int(len(y)), "acc_evidential": float(ok_e.mean()), "acc_mc": float(ok_c.mean()),
           "only_evidential_correct": a, "only_mc_correct": b, "mcnemar_exact_p": float(p),
           "per_sequence": {s: {"n": int((seq == s).sum()), "acc_evidential": float(ok_e[seq == s].mean()),
                                "acc_mc": float(ok_c[seq == s].mean())} for s in sorted(set(seq.tolist()))}}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_sequence"}, indent=1))
    for s, v in out["per_sequence"].items():
        print(f"  {s:<10} n={v['n']:>4}  evidential {v['acc_evidential']:.3f}  mc {v['acc_mc']:.3f}")


if __name__ == "__main__":
    main()
