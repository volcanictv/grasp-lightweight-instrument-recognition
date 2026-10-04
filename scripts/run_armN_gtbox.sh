#!/bin/bash
# Trains the tracker-style-crop (arm N) members of the four-member evidential ensemble for seeds 42, 43, 44 on the official split, per the
# protocol addendum of 2026-10-04 in docs/reports/gtbox_sam_protocol.md, and writes their ensemble configs (configs/arms/ens4_N_official_s*.yaml).
# Waits for the running gtft_ens pipeline so the two do not fight over the GPUs and the CPU. Finished runs (the seed-42 members of the earlier
# official arm-N run) are skipped. Usage (titanxp, repo root): nohup scripts/run_armN_gtbox.sh > experiments/armN_gtbox.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }
say "waiting for the gtft_ens pipeline"
until [ -f experiments/gtbox_sam/gtft_ens/pipeline.done ] && [ -f experiments/latency.done ]; do sleep 120; done  # the latency suite needs idle GPUs first
say "training arm N, four members, seeds 42 43 44 (official split)"
$PY scripts/run_training_arms.py --fold official --arms N --seeds 42 43 44 --members resnet50_320 resnet50_224 baseline letterbox_crop \
  --neighbour-dir experiments/temporal_neighbors/fold1 experiments/temporal_neighbors/fold2 --data-root "$GRASP_DATA_ROOT"
$PY scripts/make_armN_ensemble_configs.py --arm N --fold official --seeds 42 43 44
say "arm N members trained, ensemble configs written"
echo done > experiments/armN_gtbox.done
