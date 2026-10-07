#!/bin/bash
# Resumes scripts/run_causal_sam2.sh after the tracking stage: skips tracking (the tracks are in experiments/gtbox_sam/VARIANT/tracked_causal_sam2, backups in
# ~/causal_sam2_backup), then reclassifies the tracks with the seed 43 and 44 members and scores seeds 42, 43, 44, each only if its output is missing.
# Do not edit run_causal_sam2.sh while it runs (bash reads scripts incrementally); this file is separate on purpose.
# Usage (titanxp, repo root): scripts/finish_causal_sam2.sh VARIANT [GPU]
set -uo pipefail
VARIANT=$1; GPU=${2:-0}
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/$VARIANT
T=$D/tracked_causal_sam2
say() { echo "[$(date +%H:%M:%S)] $*"; }
[ -f $T/masks_shard0.pkl ] || { say "no tracks in $T (restore from ~/causal_sam2_backup/$VARIANT first)"; exit 1; }

for s in 43 44; do
  mkdir -p ${T}_s$s
  if [ ! -f ${T}_s$s/frames_shard0.npz ]; then
    $PY scripts/reclassify_tracked_masks.py --masks $T/masks_shard0.pkl --ensemble-config configs/arms/ens4_N_official_s$s.yaml --device cuda:$GPU \
      --out ${T}_s$s/frames_shard0.npz >> $T/track_resume.log 2>&1
    say "seed $s reclassified"
  fi
done
for s in 42 43 44; do
  out=docs/reports/gtbox_sam/${VARIANT}_causalsam2_s$s.json
  tdir=${T}_s$s; [ $s = 42 ] && tdir=$T
  if [ ! -f $out ]; then
    $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $tdir --tau 2.85e-4 --budgets 539 833 --out $out >> $T/score_resume.log 2>&1
    say "seed $s scored"
  fi
done
$PY scripts/aggregate_gtbox_seeds.py --variant ${VARIANT}_causalsam2 --seeds 42 43 44 | tee $T/aggregate.txt
echo done > $D/causal_sam2.done
