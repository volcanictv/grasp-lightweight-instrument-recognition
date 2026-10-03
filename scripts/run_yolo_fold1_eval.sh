#!/bin/bash
# Fold1 scoring of the YOLO-neighbour arms (baseline, N, Y, NY; seeds 42 43 44) for docs/reports/training_arms/yolo_fold1.json, the file the
# pre-registered choice step of scripts/overnight_yolo_arms.sh reads. Run alongside the fold2 confirmation training (its own fold1
# scoring crashed on a since-fixed f-string error).
# Usage (titanxp, repo root): nohup scripts/run_yolo_fold1_eval.sh > experiments/yolo_fold1_eval.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
TN=experiments/temporal_neighbors
TNY=experiments/temporal_neighbors_yolo
echo "[$(date +%H:%M:%S)] fold1 scoring start"
$PY scripts/evaluate_training_arms.py --fold fold1 --arms baseline N Y NY --seeds 42 43 44 --heldout-neighbour-dir $TN/fold1 --heldout-yolo-neighbour-dir $TNY/fold1 \
  --data-root "$GRASP_DATA_ROOT" --device cuda:1 --out docs/reports/training_arms/yolo_fold1.json
echo "[$(date +%H:%M:%S)] fold1 scoring finished (exit $?)"
