#!/bin/bash
# Stage A of the addendum in docs/reports/training_experiments_preregistration.md. Waits for the overnight fold2
# confirmation, checks arm N against the fold2 baseline, and if it holds trains the three official-split members
# (fixed schedule, last epoch, validation inside the training data). No test data is read. Stage B
# (scripts/final_arm_official_eval.sh) is run separately, once, on the user's go.
# Usage (titanxp, repo root): scripts/final_arm_train.sh
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
LOG=experiments/final_arm.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "waiting for the fold2 confirmation"
until [ -f experiments/overnight_arms.done ]; do sleep 60; done
verdict=$($PY - <<'E'
import json
d = json.load(open("docs/reports/training_arms/fold2.json"))["summary"]
n = d["N"]
print("CONFIRMED" if n["delta_accuracy_vs_baseline"] > 0 else "NOT_CONFIRMED", f"{n['delta_accuracy_vs_baseline']:+.4f}")
E
)
say "fold2 verdict for arm N: $verdict"
case "$verdict" in
  CONFIRMED*) ;;
  *) say "not confirmed on fold2: no official-split training"; echo "not confirmed" > experiments/final_arm.done; exit 0;;
esac
say "training the official-split members, arm N, seed 42"
$PY scripts/run_training_arms.py --fold official --arms N --seeds 42 \
  --neighbour-dir experiments/temporal_neighbors/fold1 experiments/temporal_neighbors/fold2 --data-root "$GRASP_DATA_ROOT" >> $LOG 2>&1
$PY - <<'E' >> $LOG 2>&1
import sys
sys.path.insert(0, "scripts")
from evaluate_training_arms import ensemble_config
print("ensemble config:", ensemble_config("N", "official", 42))
E
say "official-split members trained; waiting for the go for stage B"
echo trained > experiments/final_arm.done
