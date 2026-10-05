#!/bin/bash
# Resumes scripts/run_gtbox_sam_pipeline.sh after the tracking step: the tracked masks of the other two seeds' members, the per-seed scoring, the three-seed
# aggregate and the done flag. Needed when the main script stopped after tracking (here: it was replaced on disk while it was running, and bash reads
# scripts incrementally). Never edit a pipeline script while it is running; copy it, or wait.
# Usage (titanxp, repo root): scripts/finish_gtbox_pipeline.sh gtft_ens   (ENS_TPL as for the main script)
set -uo pipefail
VARIANT=${1:?variant}
SEEDS="42 43 44"
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
ENS_TPL=${ENS_TPL:-configs/evidential/ens_E_official_s%s_lam0p01a10.yaml}
ens() { printf "$ENS_TPL" "$1"; }
D=experiments/gtbox_sam/$VARIANT
LOG=experiments/gtbox_sam/pipeline_$VARIANT.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "resuming after tracking"
for s in 43 44; do
  $PY scripts/reclassify_tracked_masks.py --masks $D/tracked/masks_shard0.pkl $D/tracked/masks_shard1.pkl --ensemble-config $(ens $s) \
    --device cuda:0 --out $D/tracked_s$s/frames_shard0.npz >> $LOG 2>&1
  say "tracks reclassified with the seed $s members"
done

mkdir -p docs/reports/gtbox_sam
for s in $SEEDS; do
  tdir=$D/tracked_s$s; [ $s = 42 ] && tdir=$D/tracked
  $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_s$s.npz --tracked-dir $tdir --out docs/reports/gtbox_sam/${VARIANT}_s$s.json >> $LOG 2>&1
  say "scored seed $s"
done
$PY scripts/aggregate_gtbox_seeds.py --variant $VARIANT --seeds 42 43 44 >> $LOG 2>&1
say "aggregated"
echo done > $D/pipeline.done
