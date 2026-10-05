#!/bin/bash
# Unattended follow-up to scripts/finetune_sam2_gtbox.py: applies the rule fixed in docs/reports/gtbox_sam_protocol.md (addendum
# 2026-10-03) and runs the unchanged GT-box pipeline with the chosen segmentor.
#   1. wait for both fine-tune runs (experiments/sam2_gtbox/dec, enc4) to finish their fold1 flip-average check
#   2. variant = the higher best fold1 mean mask IoU; flip averaging iff it raises that checkpoint's fold1 mean IoU
#   3. one pass of the chosen segmentor over the official test boxes, then scripts/run_gtbox_sam_pipeline.sh gtft
# Usage (titanxp, repo root): nohup scripts/run_gtbox_sam_gtft.sh > experiments/gtbox_sam/gtft_driver.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
R=experiments/sam2_gtbox
D=experiments/gtbox_sam/gtft
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for the fine-tune runs"
until [ -f $R/dec/tta.json ] && [ -f $R/enc4/tta.json ]; do sleep 120; done

read -r CHOSEN TTA < <($PY - <<'E'
import json
best = {}
for v in ("dec", "enc4"):
    log = json.load(open(f"experiments/sam2_gtbox/{v}/log.json"))["log"]
    best[v] = max(e["mean_iou"] for e in log if e["epoch"] >= 1)
chosen = max(best, key=best.get)
t = json.load(open(f"experiments/sam2_gtbox/{chosen}/tta.json"))
print(chosen, "tta" if t["flip_tta"]["mean_iou"] > t["plain"]["mean_iou"] else "plain")
E
)
say "chosen $CHOSEN, $TTA (fold1: $($PY -c "import json;print(json.load(open('$R/$CHOSEN/tta.json')))"))"

mkdir -p $D
ARGS="--variant gtft --weights $R/$CHOSEN/weights.pt --out-dir $D --device cuda:0"
[ "$TTA" = tta ] && ARGS="$ARGS --tta"
$PY scripts/gtbox_sam_masks.py $ARGS > $D/masks.log 2>&1
say "test masks: $($PY -c "import json;d=json.load(open('$D/summary.json'));print('mean mask IoU', round(d['mean_mask_iou'],4), 'IoU>=0.5', round(d['iou_ge_0.5'],4), 'IoU>=0.75', round(d['iou_ge_0.75'],4))")"

scripts/run_gtbox_sam_pipeline.sh gtft
say "pipeline finished"
