#!/bin/bash
# Fold1 post-processing study of the segmenter side: dumps SAM2 and SAM3 flip-averaged logit windows for every fold1 box on both GPUs, then runs
# scripts/postproc_study.py on the CPU (docs/reports/gtbox_sam/postproc_fold1.json).
# Usage (titanxp, repo root): nohup scripts/run_postproc_study.sh > experiments/postproc/driver.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/sam3_venv/bin/python
mkdir -p experiments/postproc
say() { echo "[$(date +%H:%M:%S)] $*"; }
say "dumping fold1 logit windows on two GPUs"
for g in 0 1; do
  $PY -u scripts/dump_fold_logits.py --split fold1 --offset $g --stride 2 --device cuda:$g --out experiments/postproc/fold1_part$g.pkl > experiments/postproc/dump_part$g.log 2>&1 &
done
wait
say "dump done: $(tail -n 1 experiments/postproc/dump_part0.log), $(tail -n 1 experiments/postproc/dump_part1.log)"
$PY scripts/postproc_study.py --parts experiments/postproc/fold1_part0.pkl experiments/postproc/fold1_part1.pkl --out docs/reports/gtbox_sam/postproc_fold1.json > experiments/postproc/study.log 2>&1
say "study done"
tail -n 30 experiments/postproc/study.log
