#!/bin/bash
# Extra, disclosed run requested after the SAM3 experiments: the unchanged GT-box pipeline with the SAM2 + SAM3 ensemble as the segmentor
# (variant gtft_ens): test masks from the ensemble on both GPUs, then scripts/run_gtbox_sam_pipeline.sh gtft_ens (logits for the three seeds,
# SAM2-large tracking, gating, scoring). The registered rule had kept SAM2 (fold1 ensemble 0.9131 against the 0.9136 bar), so this is reported
# next to the gtft run and does not replace it.
# Usage (titanxp, repo root): nohup scripts/run_gtbox_sam_ensemble.sh > experiments/gtbox_sam/gtft_ens_driver.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/sam3_venv/bin/python
D=experiments/gtbox_sam/gtft_ens
mkdir -p $D
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "ensemble test masks on two GPUs"
for g in 0 1; do
  $PY -u scripts/gtbox_sam_masks_ensemble.py --split test --offset $g --stride 2 --device cuda:$g --out $D/part$g.pkl > $D/masks_part$g.log 2>&1 &
done
wait
$PY scripts/merge_gtbox_mask_parts.py --parts $D/part0.pkl $D/part1.pkl --variant gtft_ens --out-dir $D > $D/merge.log 2>&1
say "test masks: $($PY -c "import json;d=json.load(open('$D/summary.json'));print('mean mask IoU', round(d['mean_mask_iou'],4), 'IoU>=0.5', round(d['iou_ge_0.5'],4), 'IoU>=0.75', round(d['iou_ge_0.75'],4))")"

scripts/run_gtbox_sam_pipeline.sh gtft_ens
say "pipeline finished"
