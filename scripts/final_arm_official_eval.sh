#!/bin/bash
# Stage B: the single official-test evaluation of the final arm-N model, with the settings fixed in
# docs/reports/training_experiments_preregistration.md (tracker A pipeline, gate 2.85e-4 on the new ensemble's S1,
# EdgeTAM, 20 causal past frames). Run once, after stage A, on the user's go.
# Usage (titanxp, repo root): scripts/final_arm_official_eval.sh
set -euo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
ENS=configs/arms/ens_N_official_s42.yaml
EXTRACT=experiments_edl/extract/N_grasp_official_s42.npz
D=experiments/official_causal/edgetam_armN
mkdir -p $D docs/reports/causal_realtime

$PY scripts/evidential_extract.py --ensemble-config $ENS --target grasp_test --data-root "$GRASP_DATA_ROOT" --device cuda:0 --out $EXTRACT
$PY - <<E
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np
import evidential_three_member_yolo as t
y, base, s1 = t.base_scores(Path("$EXTRACT"))
idx = np.where(s1 >= 0.000285)[0]
json.dump({"errors": [{"index": int(i)} for i in idx]}, open("$D/idx.json", "w"))
print(f"new ensemble alone: accuracy {(base == y).mean():.4f}; threshold 2.85e-4 gates {len(idx)} of {len(y)} instruments")
E
~/edgetam_venv/bin/python scripts/evaluate_temporal_track_ensemble.py --split test --causal --window 20 --ensemble-config $ENS \
  --sam2-checkpoint ~/EdgeTAM/checkpoints/edgetam.pt --sam2-config configs/edgetam.yaml --error-cases-json $D/idx.json \
  --tmp-dir /tmp/et_official_armN --frame-logits-out $D/official_frames_shard0.npz --device cuda:0 --out $D/official_tracked_shard0.json
$PY scripts/score_causal_official.py --name edgetam_armN --dir $D --k 20 --tau 0.000285 --extract $EXTRACT --config $ENS \
  --out docs/reports/causal_realtime/official_edgetam_armN.json
