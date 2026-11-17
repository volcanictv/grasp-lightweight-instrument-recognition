#!/bin/bash -l
# GROUND-TRUTH-MASK RUN of the 5-fold cross-validation (recognition paper): the classifier members of the finished campaign ($WORK/campaign_cv5) are reused, nothing is retrained.
# Per fold: annotated masks -> single-pass logits -> the 50% most uncertain instruments -> SAM2-large tracking over +-10 frames -> scoring (scripts/campaign_tasks_gtmask.py, run by scripts/campaign.py).
# Resumable: finished tasks are never redone, so after any stop the SAME command continues:
#   cd ~/grasp_work && sbatch --mail-user=slack:@<RC username> --export=ALL,CODE_DIR=<staged code directory> <that directory>/slurm/rc_campaign_gtmask.sbatch
# Smoke test first (onboard, under an hour):  sbatch --mail-user=slack:@<RC username> --partition=onboard --gres=gpu:a100:1 --time=0-01:00:00 --export=ALL,CODE_DIR=<dir>,SMOKE=1 <dir>/slurm/rc_campaign_gtmask.sbatch
#SBATCH --job-name=grasp-gtmask-cv5
#SBATCH --comment="5-fold CV with ground-truth masks, reused classifier members, 2 a100, resumable"
#SBATCH --account=robot-surg
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48g
#SBATCH --time=0-10:00:00
#SBATCH --mail-user=slack:@YOUR_RC_USERNAME
# YOUR_RC_USERNAME is a placeholder: submit with  sbatch --mail-user=slack:@<your RC username> ...  (the command line overrides this line)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
export WORK=$HOME/grasp_work
: "${CODE_DIR:?set CODE_DIR to the staged code directory name, e.g. code_081fc2b}"
CODE=$WORK/$CODE_DIR
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=$WORK/torch_home   # nothing is downloaded on the compute node
export PY_MAIN=$WORK/envs/main/bin/python PY_SAM3=$WORK/envs/main/bin/python PY_ET=$WORK/envs/edgetam/bin/python
export SAM2_CKPT_DIR=$WORK/checkpoints/sam2 EDGETAM_CKPT=$WORK/EdgeTAM/checkpoints/edgetam.pt
export CAMPAIGN_WORKERS=3 TRACK_SHARDS=2 MALLOC_ARENA_MAX=2
OUT=$WORK/campaign_cv5_gtmask   # fixed name: a resubmission continues in the same directory
OLD_CAMPAIGN=$WORK/campaign_cv5
SMOKE_ARGS=""
if [ "${SMOKE:-0}" = 1 ]; then OUT=$WORK/campaign_gtmask_smoke; OLD_CAMPAIGN=$(ls -d $WORK/campaign_smoke_* | tail -n 1); SMOKE_ARGS="--smoke"; fi
export OLD_CAMPAIGN
HOMEDATA=$WORK/data
LOCAL=/tmp/${USER}_campaign_${SLURM_JOB_ID:-manual}   # node-local scratch (700 GB): the frames are copied here once, so 116,000 small files are not read from the network home directory by every task
say() { echo "[$(date +%H:%M:%S)] $*"; }
echo "node $(hostname), job ${SLURM_JOB_ID:-none}, code $CODE_DIR, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, cpus ${SLURM_CPUS_ON_NODE:-?}"
nvidia-smi -L
CHK=cv5_f0; [ "${SMOKE:-0}" = 1 ] && CHK=cv5_smoke
for f in "$HOMEDATA/GraSP/annotations/cv5_splits.json" "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" "$PY_MAIN" "$OLD_CAMPAIGN/$CHK/ens4.yaml"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
mkdir -p "$OUT" "$LOCAL/GraSP" || exit 1
trap 'case "$LOCAL" in /tmp/${USER}_campaign_*) rm -rf "$LOCAL";; esac' EXIT   # the only deletion: this job's private scratch directory

if [ "${SMOKE:-0}" = 1 ]; then
  export GRASP_DATA_ROOT=$HOMEDATA/GraSP   # the smoke test touches a handful of frames: no copy of the 14 GB
else
  say "copying the dataset to node-local scratch"
  cp -r "$HOMEDATA/GraSP/annotations" "$LOCAL/GraSP/annotations" || exit 1
  mkdir -p "$LOCAL/GraSP/frames-001/frames"
  ls "$HOMEDATA/GraSP/frames-001/frames" | grep '^CASE' | xargs -P 8 -I{} sh -c 'tar -C "$1" -cf - "$3" | tar -C "$2" -xf -' _ "$HOMEDATA/GraSP/frames-001/frames" "$LOCAL/GraSP/frames-001/frames" {} || { echo "dataset copy failed"; exit 1; }
  N=$(find "$LOCAL/GraSP/frames-001/frames" -type f | wc -l)
  [ "$N" -eq 116521 ] || { echo "expected 116521 frame files, found $N"; exit 1; }
  say "dataset on local scratch: $N frames, $(du -sh "$LOCAL" | cut -f1)"
  export GRASP_DATA_ROOT=$LOCAL/GraSP
fi

cd "$CODE" || exit 1
# one hour is kept free at the end of the allocation: no task starts that would not finish before then
$PY_MAIN scripts/campaign.py --out "$OUT" --tasks-module campaign_tasks_gtmask $SMOKE_ARGS --slots-per-gpu 2 --max-share 3 --util-target 50 --budget-hours 8.8 --cpu-slots 2
rc=$?
echo "campaign exit code: $rc"
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,AveCPU,TotalCPU,AllocCPUS,State -P 2>/dev/null | head -n 8 || true
python3 - "$OUT/gpu_util.csv" <<'EOF' 2>/dev/null || true
import csv, sys, collections
by = collections.defaultdict(list)
for r in csv.reader(open(sys.argv[1])):
    by[r[1]].append(float(r[2]))
for g, v in sorted(by.items()):
    print(f"GPU {g}: mean utilisation {sum(v)/len(v):.0f}% over {len(v)} one-minute samples, {100*sum(x >= 50 for x in v)/len(v):.0f}% of the minutes at or above 50%")
EOF
exit $rc
