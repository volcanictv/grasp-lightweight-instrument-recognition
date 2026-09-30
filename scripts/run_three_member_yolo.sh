#!/bin/bash
# One-shot run of the three-member evidential pipeline with the YOLO26 tracker.
# Usage (titanxp, repo root): scripts/run_three_member_yolo.sh FOLD1_YOLO_WEIGHTS OFFICIAL_YOLO_WEIGHTS [GPU]
# FOLD1 weights: YOLO trained with data.split fold1 (fold2 cases). OFFICIAL weights: data.split official.
set -euo pipefail
FOLD1_W=$1; OFFICIAL_W=$2; GPU=${3:-0}
PY=$HOME/yolo26_venv/bin/python
WORK=experiments/three_member_yolo
export GRASP_DATA_ROOT=${GRASP_DATA_ROOT:-$HOME/Desktop/Classification\ Surgurical\ Tools/GraSP}
mkdir -p $WORK

$PY scripts/evidential_three_member_yolo.py prep-fold1 --work-dir $WORK
$PY scripts/evaluate_temporal_track_yolo.py --yolo-weights "$FOLD1_W" --split fold1 \
    --ensemble-config configs/evidential/ens_E_fold1_s42_lam0p01a10.yaml \
    --error-cases-json $WORK/fold1_track_shard0.json --device cuda:$GPU \
    --frame-logits-out $WORK/fold1_frames_shard0.npz --out $WORK/fold1_tracked_shard0.json > $WORK/fold1_track.log 2>&1
$PY scripts/evidential_three_member_yolo.py calibrate --work-dir $WORK

$PY scripts/evidential_three_member_yolo.py prep-official --work-dir $WORK
$PY scripts/evaluate_temporal_track_yolo.py --yolo-weights "$OFFICIAL_W" --split test \
    --ensemble-config configs/evidential/ens_E_official_s42_lam0p01a10.yaml \
    --error-cases-json $WORK/official_track_shard0.json --device cuda:$GPU \
    --frame-logits-out $WORK/official_frames_shard0.npz --out $WORK/official_tracked_shard0.json > $WORK/official_track.log 2>&1
$PY scripts/evidential_three_member_yolo.py score --work-dir $WORK
