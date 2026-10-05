#!/bin/bash
# Rung 3 of the ablation ladder (protocol addendum of 2026-10-04, third): the arm N members on the same SAM2 + SAM3 ensemble masks and the same SAM2-large
# tracks as the gtft_ens run. Per seed: single-pass logits of the arm N four-member ensemble, the saved tracked masks reclassified with it, the usual scoring
# (clipped and unclipped rows), then the three-seed aggregate gtft_ens_armN. Flagged instruments without a track fall back to the single pass and are counted.
# Usage (titanxp, repo root): nohup scripts/run_armN_on_ens.sh > experiments/armN_on_ens.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
D=experiments/gtbox_sam/gtft_ens
say() { echo "[$(date +%H:%M:%S)] $*"; }
say "waiting for the arm N members and the gtft_ens pipeline"
until [ -f experiments/armN_gtbox.done ] && [ -f $D/pipeline.done ]; do sleep 120; done
mkdir -p docs/reports/gtbox_sam
for s in 42 43 44; do
  CFG=configs/arms/ens4_N_official_s$s.yaml
  $PY scripts/extract_logits_from_masks.py --masks $D/masks.pkl --ensemble-config $CFG --device cuda:0 --out $D/logits_armN_s$s.npz
  say "arm N single-pass logits, seed $s"
  $PY scripts/reclassify_tracked_masks.py --masks $D/tracked/masks_shard0.pkl $D/tracked/masks_shard1.pkl --ensemble-config $CFG --device cuda:0 \
    --out $D/tracked_armN_s$s/frames_shard0.npz
  say "tracks reclassified, seed $s"
  $PY scripts/gtbox_sam_final_eval.py --masks $D/masks.pkl --logits $D/logits_armN_s$s.npz --tracked-dir $D/tracked_armN_s$s \
    --out docs/reports/gtbox_sam/gtft_ens_armN_s$s.json
  say "scored seed $s"
done
$PY scripts/aggregate_gtbox_seeds.py --variant gtft_ens_armN --seeds 42 43 44
say "gtft_ens_armN aggregated"
echo done > experiments/armN_on_ens.done
