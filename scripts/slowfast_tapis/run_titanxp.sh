#!/bin/bash
# Run the three SlowFast runs on the two Titan Xp, detached: official split on GPU 0, fold1 then fold2 on GPU 1. Needs ~/sf30 filled and verified by relay_windows_to_titanxp.sh.
# The box has 4 CPUs, so JPEG decoding is the limit, not the GPUs; checkpoints are written every 10 epochs to keep disk use down.
# usage: run_titanxp.sh [BS]      (default batch 6 as in the recipe; pass 4 if 6 does not fit in 12 GB)
set -uo pipefail
export W=$HOME/sf30 PY=$HOME/tapis_venv/bin/python CODE=$HOME/baselines/GraSP/TAPIS_gtbox FRAMES=$HOME/sf30/windows
export BS=${1:-6} CK_EVERY=${CK_EVERY:-10} WORKERS=${WORKERS:-3}
[ -f "$W/windows/_verified" ] || { echo "windows not verified yet (no $W/windows/_verified)"; exit 2; }
mkdir -p "$W/runs" "$W/weights"
[ -e "$W/weights/SLOWFAST_4x16_R50.pkl" ] || ln -s "$HOME/slowfast_build/weights/SLOWFAST_4x16_R50.pkl" "$W/weights/SLOWFAST_4x16_R50.pkl"
R=$W/scripts/rc_slowfast_run.sh
nohup bash -c "CUDA_VISIBLE_DEVICES=0 bash $R official" > "$W/runs/official.driver.log" 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=1 bash $R fold1; CUDA_VISIBLE_DEVICES=1 bash $R fold2" > "$W/runs/folds.driver.log" 2>&1 &
echo "started (batch $BS, checkpoints every $CK_EVERY epochs); logs in $W/runs/*.driver.log and $W/runs/<run>/train.log"
