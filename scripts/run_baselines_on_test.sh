#!/bin/bash
# Waits for the last cross-entropy member (arm C official, seed 44) and then runs, one after the other: test-crop extraction of all 12 members (GPU), the baseline analysis with paired bootstrap
# (scripts/baselines_on_test.py), and the gate comparison on the refine-all tracks (scripts/gate_baselines_analysis.py). Low priority; logs to ~/baselines_on_test.log.
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }
until [ "$(ls ~/ce_campaign/*.ok 2>/dev/null | wc -l)" -ge 12 ]; do sleep 60; done
say "all 12 baseline members trained; extracting test-crop logits"
CUDA_VISIBLE_DEVICES=${EXTRACT_GPU:-0} nice -n 5 $PY -u scripts/baselines_on_test.py extract --ce-dir ~/ce_campaign || { say "extract failed"; exit 1; }
say "analysing baselines"
nice -n 10 $PY scripts/baselines_on_test.py analyze --out docs/reports/gtbox_sam/baselines_on_test.json || { say "analyze failed"; exit 1; }
say "gate comparison on the refine-all tracks"
nice -n 10 $PY -u scripts/gate_baselines_analysis.py --out docs/reports/gtbox_sam/gate_baselines.json || { say "gate analysis failed"; exit 1; }
say "done"
