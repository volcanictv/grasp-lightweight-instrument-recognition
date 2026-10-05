#!/bin/bash
# Runs the SAM3 box-prompt probe (scripts/sam3_boxprompt_probe.py) on 60 evenly spaced fold1 frames once GPU 1 has room, that is after the
# refiner driver (scripts/run_refiner.sh) has dumped its fold2 training crops. Fold1 only, no test cases.
# Usage (titanxp, repo root): nohup scripts/run_sam3_probe.sh > experiments/sam3/probe_driver.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p experiments/sam3
say "waiting for GPU 1 to free up"
until grep -q "training crops" experiments/refiner/driver.log 2>/dev/null; do sleep 60; done
say "GPU 1 free, probing SAM3 (fp16)"
~/sam3_venv/bin/python scripts/sam3_boxprompt_probe.py --frames 60 --crops experiments/refiner/fold1_enc4.npz --device cuda:1 --dtype float16 \
  --out experiments/sam3/probe_fold1.json 2>&1 | grep -v -i warn
say "probe finished"
