#!/bin/bash
# Causal (past frames only, 20-frame look-back) YOLO tracking of the top-20% fold1 instances.
# Matching config is the best of the fold1 sweep: min-iou 0.1, coast 3, center fallback.
# Usage (titanxp, repo root): scripts/run_yolo_causal_fold1.sh NAME YOLO_WEIGHTS GPU
set -euo pipefail
NAME=$1; W=$2; GPU=$3
PY=$HOME/yolo26_venv/bin/python
OUT=experiments/yolo_causal/$NAME
IDX=experiments/edgetam_causal/idx_top20.json
export GRASP_DATA_ROOT=${GRASP_DATA_ROOT:-$HOME/Desktop/Classification\ Surgurical\ Tools/GraSP}
mkdir -p $OUT
$PY scripts/cache_yolo_detections.py --yolo-weights "$W" --split fold1 --causal --window 20 \
  --error-cases-json $IDX --device cuda:$GPU --out $OUT/dets.pkl > $OUT/cache.log 2>&1
$PY scripts/evaluate_temporal_track_yolo.py --yolo-weights "$W" --split fold1 --causal --window 20 \
  --min-iou 0.1 --coast 3 --center-fallback --detections-cache $OUT/dets.pkl \
  --ensemble-config configs/evidential/ens_E_fold1_s42_lam0p01a10.yaml --error-cases-json $IDX \
  --device cuda:$GPU --frame-logits-out $OUT/fold1_frames_shard0.npz --out $OUT/fold1_tracked_shard0.json > $OUT/track.log 2>&1
echo done > $OUT/done.flag
