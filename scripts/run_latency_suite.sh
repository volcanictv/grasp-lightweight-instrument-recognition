#!/bin/bash
# Latency window on idle hardware, run right after the gtft_ens pipeline and before the arm N training takes the GPUs:
#   1. TAPIS (Swin-L Mask2Former + MViT, shipped weights) per keyframe, scripts/benchmark_tapis_latency.py
#   2. the GT-box pipeline's own stages (SAM2, SAM3, four-member classifier), scripts/benchmark_gtbox_stage_latency.py
# One at a time, on GPU 0 with nothing else running. Writes docs/reports/latency/{tapis,our_stages}.json and experiments/latency.done.
# Usage (titanxp, repo root): nohup scripts/run_latency_suite.sh > experiments/latency_suite.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p docs/reports/latency
say "waiting for the gtft_ens pipeline"
until [ -f experiments/gtbox_sam/gtft_ens/pipeline.done ]; do sleep 60; done
sleep 30
say "idle check: $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')"
say "TAPIS"
PYTHONPATH=$HOME/baselines/GraSP/TAPIS:$HOME/baselines/GraSP/TAPIS/tapis CUDA_VISIBLE_DEVICES=0 ~/tapis_venv/bin/python scripts/benchmark_tapis_latency.py \
  --reps 100 --warmup 20 --out docs/reports/latency/tapis.json > experiments/latency_tapis.log 2>&1
say "TAPIS done: $(tail -n 1 experiments/latency_tapis.log | cut -c1-100)"
say "our stages"
CUDA_VISIBLE_DEVICES=0 ~/sam3_venv/bin/python scripts/benchmark_gtbox_stage_latency.py --frames 60 --out docs/reports/latency/our_stages.json > experiments/latency_ours.log 2>&1
say "our stages done"
echo done > experiments/latency.done
say "latency window finished"
