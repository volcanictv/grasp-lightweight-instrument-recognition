#!/bin/bash
# Hard mask crop vs soft-isolated context crop, single ResNet-50 224 member, no neighbour crops, last epoch, dev folds only.
# One GPU per fold; usage: run_ctx_ablation.sh <fold 1|2> <gpu 0|1>
set -uo pipefail
F=$1; G=$2
cd ~/frame_exp
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP" PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/surgical/bin/python
T0=$(date +%s)
for s in 42 43 44; do
  for v in hard soft; do
    echo "[$(date +%H:%M:%S)] fold$F $v s$s"
    nice -n 5 $PY scripts/train.py configs/ctx/ctx_${v}_resnet50_224_fold${F}_s${s}.yaml --device cuda:$G --num-workers 2 --data-root "$GRASP_DATA_ROOT"
  done
done
echo "elapsed $(( $(date +%s)-T0 )) s, exit $?"
