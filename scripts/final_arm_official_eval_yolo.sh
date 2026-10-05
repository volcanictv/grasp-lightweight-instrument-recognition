#!/bin/bash
# Tracker B (YOLO26s-seg) counterpart of scripts/final_arm_official_eval.sh for the same final arm-N ensemble, with the
# settings locked in docs/reports/causal_tracker_preregistration.md: 15 causal past frames, gate 5.75e-4 on the new
# ensemble's S1, matching min-iou 0.1 / coast 3 / centre fallback, official-split YOLO weights (last.pt). Uses the single-frame
# extraction already made by the EdgeTAM script, so the test cases are read for tracking only.
# Usage (titanxp, repo root, after the EdgeTAM script's extraction): scripts/final_arm_official_eval_yolo.sh
set -euo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
YPY=$HOME/yolo26_venv/bin/python
ENS=configs/arms/ens_N_official_s42.yaml
EXTRACT=experiments_edl/extract/N_grasp_official_s42.npz
W=experiments/yolo26_seg_official_20260930-131921/weights/last.pt
D=experiments/official_causal/yolo26s_armN
mkdir -p $D

$YPY - <<E
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np
import evidential_three_member_yolo as t
y, base, s1 = t.base_scores(Path("$EXTRACT"))
idx = np.where(s1 >= 0.000575)[0]
json.dump({"errors": [{"index": int(i)} for i in idx]}, open("$D/idx.json", "w"))
print(f"threshold 5.75e-4 gates {len(idx)} of {len(y)} instruments")
E
$YPY scripts/cache_yolo_detections.py --yolo-weights $W --split test --causal --window 15 --error-cases-json $D/idx.json --device cuda:1 --out $D/dets.pkl
$YPY scripts/evaluate_temporal_track_yolo.py --yolo-weights $W --split test --causal --window 15 --min-iou 0.1 --coast 3 --center-fallback \
  --detections-cache $D/dets.pkl --ensemble-config $ENS --error-cases-json $D/idx.json --device cuda:1 \
  --frame-logits-out $D/official_frames_shard0.npz --out $D/official_tracked_shard0.json
$PY scripts/score_causal_official.py --name yolo26s_armN --dir $D --k 15 --tau 0.000575 --extract $EXTRACT --config $ENS \
  --out docs/reports/causal_realtime/official_yolo26s_armN.json
