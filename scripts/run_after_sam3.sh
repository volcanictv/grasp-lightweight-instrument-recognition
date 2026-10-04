#!/bin/bash
# After both SAM3 fine-tunes have finished their final fold1 check (experiments/sam3/{dec,enc4}/tta.json):
#   1. resume the paused YOLO scoring evaluations at low CPU priority (SAM3 work keeps priority)
#   2. pick the SAM3 variant with the higher flip-averaged fold1 IoU (the protocol's rule) and score SAM2, SAM3 and their ensemble on all of
#      fold1, split over the two GPUs (scripts/eval_sam_ensemble.py), then merge into docs/reports/gtbox_sam/sam3_fold1.json
# Usage (titanxp, repo root): nohup scripts/run_after_sam3.sh > experiments/sam3/after.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/sam3_venv/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for the SAM3 final checks"
until [ -f experiments/sam3/dec/tta.json ] && [ -f experiments/sam3/enc4/tta.json ]; do sleep 60; done
say "SAM3 final checks done"

for pid in $(pgrep -f "scripts/[e]val_on_neighbour_crops.py"); do renice -n 10 -p $pid > /dev/null; done
pkill -CONT -f "scripts/[e]val_on_neighbour_crops.py"
say "YOLO scoring resumed at nice 10"

CHOSEN=$($PY - <<'E'
import json
best = {v: json.load(open(f"experiments/sam3/{v}/tta.json"))["flip_tta"]["mean_iou"] for v in ("dec", "enc4")}
print(max(best, key=best.get))
E
)
say "SAM3 variant by flip-averaged fold1 IoU: $CHOSEN"
W=experiments/sam3/$CHOSEN/weights.pt
for g in 0 1; do
  $PY -u scripts/eval_sam_ensemble.py --sam3-weights $W --offset $g --stride 2 --device cuda:$g --out experiments/sam3/ens_fold1_part$g.json > experiments/sam3/ens_part$g.log 2>&1 &
done
wait
say "ensemble scoring done"
$PY scripts/merge_sam_ensemble.py --parts experiments/sam3/ens_fold1_part0.json experiments/sam3/ens_fold1_part1.json --variant $CHOSEN \
  --tta-dec experiments/sam3/dec/tta.json --tta-enc4 experiments/sam3/enc4/tta.json --out docs/reports/gtbox_sam/sam3_fold1.json
say "after-SAM3 steps finished"
