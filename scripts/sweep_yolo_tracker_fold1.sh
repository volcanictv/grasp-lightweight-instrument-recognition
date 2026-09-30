#!/bin/bash
# Fold1-only sweep of the YOLO tracker's matching options; official test is never touched.
# Usage (titanxp, repo root): scripts/sweep_yolo_tracker_fold1.sh FOLD1_YOLO_WEIGHTS
set -uo pipefail
W=$1
PY=$HOME/yolo26_venv/bin/python
OUT=experiments/yolo_sweep
export GRASP_DATA_ROOT=${GRASP_DATA_ROOT:-$HOME/Desktop/Classification\ Surgurical\ Tools/GraSP}
mkdir -p $OUT
IDX=experiments/three_member_yolo/fold1_track_shard0.json
[ -f $OUT/fold1_dets.pkl ] || $PY scripts/cache_yolo_detections.py --yolo-weights "$W" --split fold1 \
  --error-cases-json $IDX --device cuda:0 --out $OUT/fold1_dets.pkl > $OUT/cache.log 2>&1
# name|min_iou|coast|center_fallback
CONFIGS=("iou0.1|0.1|0|0" "iou0.2|0.2|0|0" "iou0.3_coast2|0.3|2|0" "iou0.1_coast2|0.1|2|0" "iou0.1_coast3_center|0.1|3|1" "iou0.2_coast2_center|0.2|2|1")

run_one() {
  IFS='|' read -r name iou coast center <<< "$1"; gpu=$2
  d=$OUT/$name; mkdir -p $d
  flag=""; [ "$center" = "1" ] && flag="--center-fallback"
  $PY scripts/evaluate_temporal_track_yolo.py --yolo-weights "$W" --split fold1 --min-iou $iou --coast $coast $flag --detections-cache $OUT/fold1_dets.pkl \
    --ensemble-config configs/evidential/ens_E_fold1_s42_lam0p01a10.yaml \
    --error-cases-json $IDX --device cuda:$gpu \
    --frame-logits-out $d/fold1_frames_shard0.npz --out $d/fold1_tracked_shard0.json > $d/track.log 2>&1
  $PY scripts/evidential_three_member_yolo.py calibrate --work-dir $d > $d/calibrate.log 2>&1
}
worker() { gpu=$1; shift; for c in "$@"; do run_one "$c" $gpu; done; }
worker 0 "${CONFIGS[0]}" "${CONFIGS[2]}" "${CONFIGS[4]}" &
worker 1 "${CONFIGS[1]}" "${CONFIGS[3]}" "${CONFIGS[5]}" &
wait
$PY scripts/summarize_yolo_sweep.py --dir $OUT
