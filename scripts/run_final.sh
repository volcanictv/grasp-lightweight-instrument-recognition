#!/bin/bash
# The final run of the protocol addendum of 2026-10-04 (third), after the arm N training:
#   1. SAM2 (GPU 0) and SAM3 (GPU 1) trained on all eight official training cases with the fixed schedules, last-epoch weights
#   2. test masks from the equal-weight ensemble of the two (both GPUs), merged
#   3. the unchanged pipeline as variant "final" with the arm N four-member ensembles (logits for three seeds, tracking, scoring, aggregate)
# Usage (titanxp, repo root): nohup scripts/run_final.sh > experiments/final_run.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SURG=/home/yzx/miniconda3/envs/surgical/bin/python
PY3=~/sam3_venv/bin/python
D=experiments/gtbox_sam/final
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for the arm N training"
until [ -f experiments/armN_gtbox.done ]; do sleep 120; done

say "training SAM2 (cuda:0, 5 epochs) and SAM3 (cuda:1, 3 epochs) on all eight training cases"
$SURG -u scripts/finetune_sam2_gtbox.py --train-split train --dev-split fold1 --fixed-epochs 5 --unfreeze-blocks 4 \
  --out-dir experiments/sam2_gtbox/final_all8 --device cuda:0 > experiments/sam2_gtbox/final_all8.log 2>&1 &
$PY3 -u scripts/finetune_sam3_gtbox.py --train-split train --dev-split fold1 --epochs 3 --unfreeze-layers 4 --save-last --dev-stride 4 --final-stride 8 \
  --out-dir experiments/sam3/final_all8 --device cuda:1 > experiments/sam3/final_all8.log 2>&1 &
wait
say "segmentors trained: $(grep -hE 'saved epoch' experiments/sam2_gtbox/final_all8.log | tail -n 1), $(grep -hE '^\[.*epoch 3/3' experiments/sam3/final_all8.log | cut -c1-90)"

say "final test masks (ensemble of the all-case SAM2 and SAM3)"
mkdir -p $D
for g in 0 1; do
  $PY3 -u scripts/gtbox_sam_masks_ensemble.py --split test --offset $g --stride 2 --device cuda:$g \
    --sam2-weights experiments/sam2_gtbox/final_all8/weights.pt --sam3-weights experiments/sam3/final_all8/weights.pt --out $D/part$g.pkl > $D/masks_part$g.log 2>&1 &
done
wait
$PY3 scripts/merge_gtbox_mask_parts.py --parts $D/part0.pkl $D/part1.pkl --variant final --out-dir $D > $D/merge.log 2>&1
say "final masks: $($PY3 -c "import json;d=json.load(open('$D/summary.json'));print('mean mask IoU', round(d['mean_mask_iou'],4), 'IoU>=0.5', round(d['iou_ge_0.5'],4), 'IoU>=0.75', round(d['iou_ge_0.75'],4))")"

ENS_TPL="configs/arms/ens4_N_official_s%s.yaml" scripts/run_gtbox_sam_pipeline.sh final
say "final pipeline finished"
echo done > experiments/final.done
