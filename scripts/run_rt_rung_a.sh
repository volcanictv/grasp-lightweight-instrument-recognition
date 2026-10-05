#!/bin/bash
# Rung A of the real-time addendum (docs/reports/gtbox_sam_protocol.md, 2026-10-05 second): one SAM2 segmenter, no flip, the arm N four-member evidential ensemble
# of seeds 42, 43 and 44 on its masks, no tracking. Writes docs/reports/gtbox_sam/<variant>_s<seed>.json and <variant>_3seed.json; the "single pass" row is the rung.
# Usage (titanxp, repo root): [TTA=1] scripts/run_rt_rung_a.sh VARIANT WEIGHTS SAM_CHECKPOINT SAM_CONFIG [GPU]   (TTA=1 averages the flipped pass)
#   e.g. scripts/run_rt_rung_a.sh rt_large experiments/sam2_gtbox/final_all8/weights.pt ~/sam2/checkpoints/sam2.1_hiera_large.pt configs/sam2.1/sam2.1_hiera_l.yaml 0
set -uo pipefail
VARIANT=$1; WEIGHTS=$2; CKPT=$3; CFG=$4; GPU=${5:-0}
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/$VARIANT
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $D $D/empty_tracks docs/reports/gtbox_sam

$PY -u scripts/gtbox_sam_masks.py --variant gtft --weights $WEIGHTS --sam-checkpoint $CKPT --sam-config $CFG --split test --device cuda:$GPU ${TTA:+--tta} --out-dir $D > $D/masks.log 2>&1
say "masks: $($PY -c "import json;d=json.load(open('$D/summary.json'));print('mean mask IoU', round(d['mean_mask_iou'],4))")"
for s in 42 43 44; do
  $PY scripts/extract_logits_from_masks.py --masks $D/masks.pkl --ensemble-config configs/arms/ens4_N_official_s$s.yaml --device cuda:$GPU --out $D/logits_s$s.npz >> $D/logits.log 2>&1
  $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $D/empty_tracks --budgets 539 --out docs/reports/gtbox_sam/${VARIANT}_s$s.json >> $D/score.log 2>&1
  say "seed $s scored"
done
$PY scripts/aggregate_gtbox_seeds.py --variant $VARIANT --seeds 42 43 44 | tee $D/aggregate.txt
echo done > $D/rung_a.done
