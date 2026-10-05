#!/bin/bash
# Rung S of the real-time addendum (docs/reports/gtbox_sam_protocol.md, 2026-10-05 second): SAM2.1 small (GPU 0) and tiny (GPU 1) fine-tuned on the fold2 cases with
# the recipe of the large model's dev run (8 epochs, last four Hiera blocks and neck unfrozen, checkpoint chosen on fold1 mean mask IoU without flip, flip reported).
# Usage (titanxp, repo root): nohup scripts/run_rt_segmenter_rung.sh > experiments/rt_rung_s.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SURG=/home/yzx/miniconda3/envs/surgical/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }

$SURG -u scripts/finetune_sam2_gtbox.py --train-split fold2 --dev-split fold1 --epochs 8 --unfreeze-blocks 4 --tta \
  --sam-checkpoint ~/sam2/checkpoints/sam2.1_hiera_small.pt --sam-config configs/sam2.1/sam2.1_hiera_s.yaml \
  --out-dir experiments/sam2_gtbox/small_enc4 --device cuda:0 > experiments/sam2_gtbox/small_enc4.log 2>&1 &
$SURG -u scripts/finetune_sam2_gtbox.py --train-split fold2 --dev-split fold1 --epochs 8 --unfreeze-blocks 4 --tta \
  --sam-checkpoint ~/sam2/checkpoints/sam2.1_hiera_tiny.pt --sam-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --out-dir experiments/sam2_gtbox/tiny_enc4 --device cuda:1 > experiments/sam2_gtbox/tiny_enc4.log 2>&1 &
wait
say "rung S training finished"
for v in small tiny; do say "$v: $(grep -h 'best epoch' experiments/sam2_gtbox/${v}_enc4.log | tail -1)"; done
echo done > experiments/rt_rung_s.done
