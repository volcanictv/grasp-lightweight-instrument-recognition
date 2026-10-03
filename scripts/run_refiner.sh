#!/bin/bash
# Unattended mask-refinement experiment under the protocol addendum in docs/reports/gtbox_sam_protocol.md (2026-10-03, second).
#   1. fold1 dev crops from the fold2-trained SAM (enc4, flip average)
#   2. wait for the fold1-trained SAM (enc4_f1), then fold2 training crops from it
#   3. train the refiner (train fold2, dev fold1), apply it to fold1 at full resolution
#   4. accept iff fold1 refined mean IoU >= 0.9136 and above SAM's own paste-back; only then dump the test crops, refine the test
#      masks and run scripts/run_gtbox_sam_pipeline.sh gtft_ref
# Usage (titanxp, repo root): nohup scripts/run_refiner.sh > experiments/refiner/driver.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
R=experiments/refiner
S=experiments/sam2_gtbox
mkdir -p $R
say() { echo "[$(date +%H:%M:%S)] $*"; }

$PY scripts/dump_refiner_crops.py --weights $S/enc4/weights.pt --tta --split fold1 --device cuda:0 --out $R/fold1_enc4.npz > $R/dump_fold1.log 2>&1
say "dev crops: $(tail -n 1 $R/dump_fold1.log)"

say "waiting for the fold1-trained SAM"
until grep -q "saved epoch 5" $S/enc4_f1.log 2>/dev/null; do sleep 120; done
say "fold1-trained SAM done: $(grep -E 'epoch 5/5' $S/enc4_f1.log | cut -c1-160)"

$PY scripts/dump_refiner_crops.py --weights $S/enc4_f1/weights.pt --tta --split fold2 --device cuda:1 --out $R/fold2_encB.npz > $R/dump_fold2.log 2>&1
say "training crops: $(tail -n 1 $R/dump_fold2.log)"

$PY scripts/train_mask_refiner.py --train $R/fold2_encB.npz --dev $R/fold1_enc4.npz --out-dir $R/run1 --device cuda:0 > $R/train.log 2>&1
say "refiner trained: $(tail -n 1 $R/train.log)"

$PY scripts/apply_mask_refiner.py --crops $R/fold1_enc4.npz --weights $R/run1/weights.pt --split fold1 --out-dir $R/run1 --device cuda:0 > $R/apply_fold1.log 2>&1
VERDICT=$($PY - <<'E'
import json
d = json.load(open("experiments/refiner/run1/fold1_summary.json"))
ok = d["mean_mask_iou"] >= 0.9136 and d["mean_mask_iou"] > d["sam_clipped_mean_mask_iou"]
print("accept" if ok else "reject", round(d["mean_mask_iou"], 4), "refined;", round(d["sam_mean_mask_iou"], 4), "sam;", round(d["sam_clipped_mean_mask_iou"], 4), "sam paste-back")
E
)
say "fold1 full-resolution check: $VERDICT"
case "$VERDICT" in accept*) ;; *) say "refiner rejected by the pre-registered rule, nothing run on the test cases"; exit 0;; esac

$PY scripts/dump_refiner_crops.py --weights $S/enc4/weights.pt --tta --split test --device cuda:0 --out $R/test_enc4.npz > $R/dump_test.log 2>&1
D=experiments/gtbox_sam/gtft_ref
$PY scripts/apply_mask_refiner.py --crops $R/test_enc4.npz --weights $R/run1/weights.pt --split test --save-masks --out-dir $D --device cuda:0 > $R/apply_test.log 2>&1
say "test masks: $($PY -c "import json;d=json.load(open('$D/summary.json'));print('refined', round(d['mean_mask_iou'],4), 'sam', round(d['sam_mean_mask_iou'],4), 'IoU>=0.75', round(d['iou_ge_0.75'],4))")"
scripts/run_gtbox_sam_pipeline.sh gtft_ref
say "pipeline finished"
