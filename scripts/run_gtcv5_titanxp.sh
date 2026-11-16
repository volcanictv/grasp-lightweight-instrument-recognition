#!/bin/bash
# Five-fold cross-validation with ground-truth masks on titanxp (both GPUs): the classifier members of the finished campaign (copied to ~/gtcv5_old/cv5_f*/, ensemble configs rewritten to local paths)
# are reused; per fold annotated masks, single-pass logits, SAM2 tracking of the 50% most uncertain instruments and scoring (scripts/campaign_tasks_gtmask.py run by scripts/campaign.py, resumable).
# Usage (titanxp, repo root): nohup bash scripts/run_gtcv5_titanxp.sh > ~/gtcv5.log 2>&1 &
set -uo pipefail
cd ~/grasp_yolo26
export GRASP_DATA_ROOT="$HOME/Desktop/Classification Surgurical Tools/GraSP"
export PY_MAIN=$HOME/miniconda3/envs/surgical/bin/python PY_SAM3=$HOME/miniconda3/envs/surgical/bin/python SAM2_CKPT_DIR=$HOME/sam2/checkpoints
export OLD_CAMPAIGN=$HOME/gtcv5_old TRACK_SHARDS=2 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T0=$(date +%s)
for k in 0 1 2 3 4; do [ -f $OLD_CAMPAIGN/cv5_f$k/ens4.yaml ] || { echo "missing ensemble config of fold $k"; echo "elapsed 0 s, exit 1"; exit 1; }; done
$PY_MAIN scripts/campaign.py --out experiments/gtcv5 --tasks-module campaign_tasks_gtmask --gpus 0,1 --slots-per-gpu 1 --max-share 1 --util-target 50 --budget-hours 24 --cpu-slots 1
rc=$?
echo "campaign exit code: $rc"
echo "elapsed $(( $(date +%s)-T0 )) s, exit $rc"
