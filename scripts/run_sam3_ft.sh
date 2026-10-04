#!/bin/bash
# Launches one SAM3 fine-tune of the protocol addendum in docs/reports/gtbox_sam_protocol.md (2026-10-03, third).
#   scripts/run_sam3_ft.sh dec  cuda:0     S3a, decoder and prompt encoder
#   scripts/run_sam3_ft.sh enc4 cuda:1     S3b, plus the neck and the last four ViT layers (waits for the fold1 YOLO scoring to free the GPU)
# Usage (titanxp, repo root): nohup scripts/run_sam3_ft.sh dec cuda:0 > experiments/sam3/dec.log 2>&1 &
set -uo pipefail
VARIANT=${1:?variant dec|enc4}
DEV=${2:-cuda:0}
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/sam3
ARGS="--out-dir experiments/sam3/$VARIANT --device $DEV --epochs 3"
if [ "$VARIANT" = enc4 ]; then
  ARGS="$ARGS --unfreeze-layers 4"
fi
echo "[$(date +%H:%M:%S)] SAM3 fine-tune $VARIANT on $DEV"
~/sam3_venv/bin/python -u scripts/finetune_sam3_gtbox.py $ARGS 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|rope_theta\|sam3_video"
echo "[$(date +%H:%M:%S)] SAM3 fine-tune $VARIANT finished"
