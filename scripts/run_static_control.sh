#!/bin/bash
# The static-box control (docs/reports/gtbox_sam_protocol.md, 2026-10-07): classify the keyframe's own mask region in the neighbouring frames, no propagation, then score with the registered scorer.
# Usage (titanxp, repo root, after the final causal and non-causal runs exist): scripts/run_static_control.sh [GPU]
set -uo pipefail
GPU=${1:-1}
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/final
say() { echo "[$(date +%H:%M:%S)] $*"; }
say "classifying the neighbouring frames"
MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 CUDA_VISIBLE_DEVICES=$GPU $PY scripts/static_mask_control.py --variant final > $D/static_control.log 2>&1 || { say "control failed"; exit 1; }
for W in static_causal static_noncausal; do
  for s in 42 43 44; do
    tdir=$D/tracked_$W; [ $s != 42 ] && tdir=${tdir}_s$s
    $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $tdir --tau 2.85e-4 --budgets 539 833 \
      --out docs/reports/gtbox_sam/final_${W}_s$s.json >> $D/static_score.log 2>&1
    say "$W seed $s scored"
  done
  $PY scripts/aggregate_gtbox_seeds.py --variant final_$W --seeds 42 43 44 | tee $D/static_aggregate_$W.txt
done
echo done > $D/static_control.done
