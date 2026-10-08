#!/bin/bash
# Smoke test of the campaign on titanxp: every task of one pseudo-fold on tiny splits, two GPUs, then (second argument "resume") the same output dir again, which must find everything done.
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PY_MAIN=/home/yzx/miniconda3/envs/surgical/bin/python
export PY_SAM3=/home/yzx/sam3_venv/bin/python
export PY_ET=/home/yzx/edgetam_venv/bin/python
export SAM2_CKPT_DIR=/home/yzx/sam2/checkpoints
export EDGETAM_CKPT=/home/yzx/EdgeTAM/checkpoints/edgetam.pt
export MALLOC_ARENA_MAX=2
OUT=${1:-$HOME/cs_smoke}
$PY_MAIN scripts/campaign.py --out "$OUT" --smoke --gpus 0,1 --budget-hours 2 --cpu-slots 2
echo "campaign exit code: $?"
