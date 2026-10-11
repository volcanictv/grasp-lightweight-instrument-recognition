"""Softmax confidence, MC-dropout vote disagreement and the evidential epistemic score S1 as error detectors, on the same instruments and seeds.
AUROC for flagging a wrong prediction and the share of all errors found when the 20% highest-scoring instruments are reviewed, on the held-out fold (fold1) and on the official test cases,
three classifier seeds each. Uses the extracts of scripts/evidential_extract.py (experiments_edl/extract/{C,E}_grasp_{fold1,official}_s<seed>.npz).

Usage (titanxp, surgical environment, repo root):
    python scripts/uncertainty_three_way.py --extract-dir experiments_edl/extract --out docs/reports/evidential/three_way.json
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
from sklearn.metrics import roc_auc_score

from evidential_analyze import arm_c, arm_e, load

NAMES = {"B2_softmax_confidence": "Softmax confidence", "B1_mc_vote_disagreement": "MC dropout (80 passes)", "S1_epistemic": "Evidential S1 (1 pass)"}


def caught(err: np.ndarray, score: np.ndarray, share: float = 0.2) -> float:
    k = int(round(share * len(score)))
    top = np.argsort(-score, kind="stable")[:k]
    return float(err[top].sum() / max(1, err.sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out: dict = {}
    for split in ("fold1", "official"):
        rows = {n: {"auroc": [], "caught20": []} for n in NAMES}
        for seed in (42, 43, 44):
            cf, ef = args.extract_dir / f"C_grasp_{split}_s{seed}.npz", args.extract_dir / f"E_grasp_{split}_s{seed}.npz"
            if not (cf.exists() and ef.exists()):
                print("missing", cf.name, ef.name)
                continue
            c = arm_c(load(cf))
            e = arm_e(load(ef), c["signals"]["B1_mc_vote_disagreement"])
            for key, name in NAMES.items():
                arm = e if key.startswith("S") else c
                err, score = np.asarray(arm["err"]).astype(int), arm["signals"][key]
                rows[key]["auroc"].append(float(roc_auc_score(err, score)))
                rows[key]["caught20"].append(caught(err, score))
        out[split] = {NAMES[k]: {m: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "values": v} for m, v in r.items() if v} for k, r in rows.items()}
        print(split)
        for name, r in out[split].items():
            if r:
                print(f"  {name:26s} AUROC {r['auroc']['mean']:.3f} +- {r['auroc']['sd']:.3f}   errors caught at 20% review {100 * r['caught20']['mean']:.1f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
