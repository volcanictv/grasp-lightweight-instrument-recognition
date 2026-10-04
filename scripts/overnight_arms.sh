#!/bin/bash
# Unattended run of docs/reports/training_experiments_preregistration.md: waits for the neighbour-crop generators,
# trains arms P and N on fold1 (3 seeds), scores them, then runs the fold2 confirmation (seed 42) and scores that.
# Usage (titanxp, repo root): scripts/overnight_arms.sh
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
LOG=experiments/overnight_arms.log
mkdir -p docs/reports/training_arms
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "waiting for neighbour generation"
until grep -q "^done:" experiments/temporal_neighbors/gen_fold1.log && grep -q "^done:" experiments/temporal_neighbors/gen_fold2.log; do sleep 60; done
say "generation finished: $(tail -1 experiments/temporal_neighbors/gen_fold2.log) | $(tail -1 experiments/temporal_neighbors/gen_fold1.log)"

say "fold1: training arms P and N"
$PY scripts/run_training_arms.py --fold fold1 --arms P N --seeds 42 43 44 --neighbour-dir experiments/temporal_neighbors/fold2 --data-root "$GRASP_DATA_ROOT" >> $LOG 2>&1
say "fold1: evaluating"
$PY scripts/evaluate_training_arms.py --fold fold1 --arms baseline P N --seeds 42 43 44 --heldout-neighbour-dir experiments/temporal_neighbors/fold1 \
  --data-root "$GRASP_DATA_ROOT" --out docs/reports/training_arms/fold1.json >> $LOG 2>&1

say "fold2 confirmation: training arms P and N, seed 42"
$PY scripts/run_training_arms.py --fold fold2 --arms P N --seeds 42 --neighbour-dir experiments/temporal_neighbors/fold1 --data-root "$GRASP_DATA_ROOT" >> $LOG 2>&1
say "fold2 confirmation: evaluating"
$PY scripts/evaluate_training_arms.py --fold fold2 --arms baseline P N --seeds 42 --heldout-neighbour-dir experiments/temporal_neighbors/fold2 \
  --data-root "$GRASP_DATA_ROOT" --out docs/reports/training_arms/fold2.json >> $LOG 2>&1
say "all done"
echo done > experiments/overnight_arms.done
