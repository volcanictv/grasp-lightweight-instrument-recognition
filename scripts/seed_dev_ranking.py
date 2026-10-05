"""Ranks the three evidential seeds on the held-out fold1 cases (the members of E_grasp_fold1_s<seed> were trained on the fold2
cases), the legitimate basis for choosing one seed: four-member ensemble alone, accuracy and macro-F1.

Usage (titanxp, surgical environment):
    python scripts/seed_dev_ranking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix

for seed in (42, 43, 44):
    z = np.load(REPO_ROOT / "experiments_edl" / "extract" / f"E_grasp_fold1_s{seed}.npz")
    alpha = alpha_mix(lambda k: z["det_" + k], CONFIGS["four"])
    pred = alpha.argmax(axis=1)
    print(f"seed {seed}: fold1 accuracy {(pred == z['y']).mean():.4f}  macro-F1 {f1_score(z['y'], pred, average='macro', labels=range(7)):.4f}")
