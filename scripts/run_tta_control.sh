#!/bin/bash
# The compute-matched TTA control end to end: classify 21 augmented views of every instrument's keyframe crop (scripts/tta_control.py, GPU 1), then score each seed with the same gate and fusion as the
# tracked runs (scripts/gtbox_sam_final_eval.py, tau 1.7e-5 and the top-833 and refine-all budgets). Outputs: experiments/gtbox_sam/final/tracked_tta*, docs/reports/gtbox_sam/final_tta_s<seed>.json,
# experiments/rescore/tta_s<seed>.frames.pkl.
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/final
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p experiments/rescore docs/reports/gtbox_sam
say "classifying the augmented views"
CUDA_VISIBLE_DEVICES=${TTA_GPU:-1} nice -n 5 $PY -u scripts/tta_control.py --variant final --tag tta || { say "tta_control failed"; exit 1; }
for s in 42 43 44; do
  if [ "$s" = 42 ]; then T=$D/tracked_tta; else T=$D/tracked_tta_s$s; fi
  nice -n 10 $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $T --tau 1.7e-5 --budgets 833 2861 \
    --out docs/reports/gtbox_sam/final_tta_s$s.json --save-frames experiments/rescore/tta_s$s.frames.pkl > experiments/rescore/tta_s$s.log 2>&1
  say "seed $s scored: $(grep -E 'gated \(tau|gated \(top 2861' experiments/rescore/tta_s$s.log | cut -c1-110 | tr '\n' '|')"
done
say "done"
