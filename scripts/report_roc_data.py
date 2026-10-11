"""Data for the PI report's AUROC figure and latency table: ROC curves of MC-dropout vote disagreement and the evidential epistemic score S1 on held-out fold1
(three seeds, mean curve on a common false-positive grid), and the instruments-per-frame distribution of the official test split.

Usage (titanxp, surgical environment):
    python scripts/report_roc_data.py --extract-dir experiments_edl/extract --out docs/reports/evidential/roc_fold1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from evidential_analyze import arm_c, arm_e, load
from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    grid = np.linspace(0.0, 1.0, 101)
    curves = {"MC dropout": [], "Evidential": []}
    aucs = {"MC dropout": [], "Evidential": []}
    for seed in (42, 43, 44):
        c = arm_c(load(args.extract_dir / f"C_grasp_fold1_s{seed}.npz"))
        e = arm_e(load(args.extract_dir / f"E_grasp_fold1_s{seed}.npz"), c["signals"]["B1_mc_vote_disagreement"])
        for name, arm, key in (("MC dropout", c, "B1_mc_vote_disagreement"), ("Evidential", e, "S1_epistemic")):
            score, err = arm["signals"][key], arm["err"]
            fpr, tpr, _ = roc_curve(err, score)
            curves[name].append(np.interp(grid, fpr, tpr))
            aucs[name].append(float(roc_auc_score(err, score)))
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    per_frame = Counter(inst[0] for inst in ds.instances)
    dist = Counter(per_frame.values())
    out = {"fpr": grid.tolist(), "tpr_mean": {k: np.mean(v, axis=0).tolist() for k, v in curves.items()},
           "auroc_per_seed": aucs, "auroc_mean": {k: float(np.mean(v)) for k, v in aucs.items()},
           "test_frames": len(per_frame), "test_instruments": len(ds.instances),
           "instruments_per_frame_mean": len(ds.instances) / len(per_frame), "instruments_per_frame_max": max(per_frame.values()),
           "instruments_per_frame_distribution": {str(k): v for k, v in sorted(dist.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("fpr", "tpr_mean")}, indent=1))


if __name__ == "__main__":
    main()
