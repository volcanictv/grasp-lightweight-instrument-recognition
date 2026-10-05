#!/bin/bash
# Rung A/B segmenter of the real-time addendum: SAM2.1 tiny trained on all eight official training cases with the fixed final-run schedule (5 epochs, last epoch's
# weights, last four Hiera blocks and neck unfrozen; fold1 is inside the training data and only monitored), then rung A with flip-averaged masks.
# The rule of the addendum chose tiny: fastest variant within 0.005 of the large model's fold1 IoU (docs/reports/realtime/segmenter_latency.json, fold1 logs).
# Usage (titanxp, repo root): nohup scripts/run_rt_tiny_all8.sh > experiments/rt_tiny_all8.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SURG=/home/yzx/miniconda3/envs/surgical/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }

$SURG -u scripts/finetune_sam2_gtbox.py --train-split train --dev-split fold1 --fixed-epochs 5 --unfreeze-blocks 4 \
  --sam-checkpoint ~/sam2/checkpoints/sam2.1_hiera_tiny.pt --sam-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --out-dir experiments/sam2_gtbox/tiny_all8 --device cuda:0 > experiments/sam2_gtbox/tiny_all8.log 2>&1
say "tiny trained: $(grep -h 'saved epoch' experiments/sam2_gtbox/tiny_all8.log | tail -1)"
echo done > experiments/rt_tiny_all8.done
