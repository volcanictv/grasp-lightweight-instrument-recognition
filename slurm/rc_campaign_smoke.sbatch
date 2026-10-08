#!/bin/bash -l
# SMOKE TEST of the campaign on the cluster (scripts/campaign.py --smoke): every task of one pseudo-fold on tiny splits (one epoch, a few instruments), on TWO A100s of the onboard partition,
# to check before the long job that the environments, the staged data, the offline model caches, the per-GPU binding of the tasks and the whole task graph work here. Expected 15 to 30 minutes, 60 allowed.
# Stage first (laptop): bash scripts/rc_stage_campaign.sh. Then, from ~/grasp_work: CODE_DIR=code_<commit> sbatch --export=ALL,CODE_DIR=$CODE_DIR code_<commit>/slurm/rc_campaign_smoke.sbatch
# Slack notification when it starts, ends or fails (you must be logged into the RC Slack workspace).
#SBATCH --job-name=grasp-campaign-smoke
#SBATCH --comment="smoke test of the 5-fold campaign: tiny splits, 2 a100"
#SBATCH --account=robot-surg
#SBATCH --partition=onboard
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=0-01:00:00
#SBATCH --mail-user=slack:@ab1340
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
export WORK=$HOME/grasp_work
: "${CODE_DIR:?set CODE_DIR to the staged code directory name, e.g. code_4406a60}"
CODE=$WORK/$CODE_DIR
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$WORK/torch_home   # nothing is downloaded on the compute node
export GRASP_DATA_ROOT=$WORK/data/GraSP
export NEIGHBOURS=$WORK/data/temporal_neighbors/fold1:$WORK/data/temporal_neighbors/fold2
export PY_MAIN=$WORK/envs/main/bin/python PY_SAM3=$WORK/envs/main/bin/python PY_ET=$WORK/envs/edgetam/bin/python
export SAM2_CKPT_DIR=$WORK/checkpoints/sam2 EDGETAM_CKPT=$WORK/EdgeTAM/checkpoints/edgetam.pt
export CAMPAIGN_WORKERS=2 TRACK_SHARDS=2 MALLOC_ARENA_MAX=2
OUT=$WORK/campaign_smoke_${SLURM_JOB_ID:-manual}
echo "node $(hostname), job ${SLURM_JOB_ID:-none}, code $CODE_DIR, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
for f in "$GRASP_DATA_ROOT/annotations/cv5_splits.json" "$WORK/torch_home/hub/checkpoints/resnet50-11ad3fa6.pth" "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" "$EDGETAM_CKPT" "$PY_MAIN" "$PY_ET"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
ls "$GRASP_DATA_ROOT/frames-001/frames" | head -n 20
# each GPU task must see exactly one GPU as cuda:0: print what the driver's tasks will get
cd "$CODE" || exit 1
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do CUDA_VISIBLE_DEVICES=$g $PY_MAIN -c "import torch,os;print('binding test: CUDA_VISIBLE_DEVICES=%s -> %d device(s), %s' % (os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count(), torch.cuda.get_device_name(0)))"; done
$PY_MAIN scripts/campaign.py --out "$OUT" --smoke --slots-per-gpu 2 --budget-hours 0.8 --cpu-slots 1
rc=$?
echo "campaign exit code: $rc"
# the utilisation-aware packing of the driver, inside this Slurm allocation: idle sleeper tasks (GPUs stay below 50%), so after two minutes each GPU must take more light tasks, up to 3, and a heavy task must never share one
echo "== packing test"; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader
$PY_MAIN scripts/campaign.py --out "$OUT/packtest" --slots-per-gpu 1 --max-share 3 --budget-hours 0.25 --tasks-module campaign_packtest
echo "packing test exit code: $?"; tail -n 6 "$OUT/packtest/gpu_util.csv" 2>/dev/null
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,AveCPU,State -P 2>/dev/null | head -n 10 || true
exit $rc
