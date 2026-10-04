#!/bin/bash
# Unattended run of docs/reports/training_experiments_yolo_neighbours_preregistration.md. Waits for the cross-fit YOLO
# (trained on the fold1 cases) and the fold1 YOLO-neighbour crops, makes the fold2-case crops, trains arms Y and NY on
# fold1 (3 seeds), scores them, runs the fold2 confirmation (seed 42), and picks the final arm by the pre-registered rule.
# Trains the official-split members of that arm (no test data read). The official evaluation is run separately.
# Usage (titanxp, repo root): scripts/overnight_yolo_arms.sh
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
PY=/home/yzx/miniconda3/envs/surgical/bin/python
YPY=$HOME/yolo26_venv/bin/python
LOG=experiments/overnight_yolo_arms.log
TN=experiments/temporal_neighbors
TNY=experiments/temporal_neighbors_yolo
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

say "waiting for the cross-fit YOLO and the fold1 YOLO crops"
until ls experiments/yolo26_seg_fold2_*/manifest.json >/dev/null 2>&1 && grep -q "^done:" $TNY/gen_fold1.log; do sleep 60; done
W2=$(ls experiments/yolo26_seg_fold2_*/weights/best.pt | head -1)
say "cross-fit YOLO ready: $W2"
$YPY scripts/build_temporal_neighbors_yolo.py --json-split fold2 --yolo-weights $W2 --device cuda:0 --out-dir $TNY/fold2 > $TNY/gen_fold2.log 2>&1
say "fold2-case YOLO crops: $(tail -1 $TNY/gen_fold2.log)"

say "fold1: training arms Y and NY"
$PY scripts/run_training_arms.py --fold fold1 --arms Y NY --seeds 42 43 44 --neighbour-dir $TN/fold2 --yolo-neighbour-dir $TNY/fold2 --data-root "$GRASP_DATA_ROOT" >> $LOG 2>&1
say "fold1: evaluating"
$PY scripts/evaluate_training_arms.py --fold fold1 --arms baseline N Y NY --seeds 42 43 44 --heldout-neighbour-dir $TN/fold1 --heldout-yolo-neighbour-dir $TNY/fold1 \
  --data-root "$GRASP_DATA_ROOT" --out docs/reports/training_arms/yolo_fold1.json >> $LOG 2>&1

say "fold2 confirmation: training arms Y and NY, seed 42"
$PY scripts/run_training_arms.py --fold fold2 --arms Y NY --seeds 42 --neighbour-dir $TN/fold1 --yolo-neighbour-dir $TNY/fold1 --data-root "$GRASP_DATA_ROOT" >> $LOG 2>&1
say "fold2 confirmation: evaluating"
$PY scripts/evaluate_training_arms.py --fold fold2 --arms baseline N Y NY --seeds 42 --heldout-neighbour-dir $TN/fold2 --heldout-yolo-neighbour-dir $TNY/fold2 \
  --data-root "$GRASP_DATA_ROOT" --out docs/reports/training_arms/yolo_fold2.json >> $LOG 2>&1

choice=$($PY - <<'E'
import json
f1 = json.load(open("docs/reports/training_arms/yolo_fold1.json"))["summary"]
f2 = json.load(open("docs/reports/training_arms/yolo_fold2.json"))["summary"]
ok = [a for a in ("Y", "NY") if a in f1 and a in f2 and f1[a]["verdict"] == "IMPROVEMENT" and f2[a]["delta_accuracy_vs_baseline"] > 0]
if not ok:
    print("NONE")
else:
    best = max(ok, key=lambda a: (round(f1[a]["accuracy"][0], 4), a == "NY"))
    print(best)
E
)
say "pre-registered choice for the final model: $choice"
if [ "$choice" = "NONE" ]; then say "no arm confirmed: no official-split training"; echo "none" > experiments/overnight_yolo_arms.done; exit 0; fi
say "training the official-split members, arm $choice, seed 42"
$PY scripts/run_training_arms.py --fold official --arms $choice --seeds 42 --neighbour-dir $TN/fold1 $TN/fold2 --yolo-neighbour-dir $TNY/fold1 $TNY/fold2 --data-root "$GRASP_DATA_ROOT" >> $LOG 2>&1
say "official-split members trained; waiting for the go for the official evaluation"
echo "$choice" > experiments/overnight_yolo_arms.done
