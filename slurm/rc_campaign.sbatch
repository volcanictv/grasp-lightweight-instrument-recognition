#!/bin/bash -l
# THE CAMPAIGN: one allocation, two A100s, about a day. scripts/campaign.py runs the task graph of scripts/campaign_tasks.py (5-fold case-level cross-validation of the whole pipeline
# plus the extras: tracking-window sweep, segmenter-seed variance), resumable: finished tasks are never redone, so after any stop (time limit, node failure, maintenance) the SAME command
# continues where it left off:   cd ~/grasp_work && sbatch --export=ALL,CODE_DIR=<staged code directory> <that directory>/slurm/rc_campaign.sbatch
# Utilisation: tasks are packed onto the two GPUs (a GPU measured below 50% takes more light tasks, scripts/campaign.py); per-minute GPU utilisation goes to $OUT/gpu_util.csv.
# Sizing: the onboard smoke job (21827048, tiny tasks, 4 tasks at a time) peaked at 13 GB and used about 10% of 4 CPUs; the real tasks are larger and up to 6 run at once (two classifier runs with 3 loader workers each, trainings, tracking), so 64 GB and 12 CPUs, to be adjusted from the first hours of the real run. Slack notification when it starts, ends or fails.
#SBATCH --job-name=grasp-campaign
#SBATCH --comment="5-fold CV of the GT-box pipeline + window sweep + segmenter seeds, 2 a100, resumable"
#SBATCH --account=robot-surg
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH --time=1-06:00:00
#SBATCH --mail-user=slack:@ab1340
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
OUT=$WORK/campaign_cv5          # fixed name: a resubmission continues in the same directory
HOMEDATA=$WORK/data
LOCAL=/tmp/${USER}_campaign_${SLURM_JOB_ID:-manual}   # node-local scratch (700 GB): the frames are copied here once, so 116,000 small files are not read from the network home directory by every task
say() { echo "[$(date +%H:%M:%S)] $*"; }
echo "node $(hostname), job ${SLURM_JOB_ID:-none}, code $CODE_DIR, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, cpus ${SLURM_CPUS_ON_NODE:-?}"
nvidia-smi -L
for f in "$HOMEDATA/GraSP/annotations/cv5_splits.json" "$TORCH_HOME/hub/checkpoints/resnet50-11ad3fa6.pth" "$SAM2_CKPT_DIR/sam2.1_hiera_large.pt" "$EDGETAM_CKPT" "$PY_MAIN" "$PY_ET"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done
mkdir -p "$OUT" "$LOCAL/GraSP" || exit 1
trap 'case "$LOCAL" in /tmp/${USER}_campaign_*) rm -rf "$LOCAL";; esac' EXIT   # the only deletion: this job's private scratch directory

say "copying the dataset to node-local scratch"
cp -r "$HOMEDATA/GraSP/annotations" "$LOCAL/GraSP/annotations" || exit 1
mkdir -p "$LOCAL/GraSP/frames-001/frames"
ls "$HOMEDATA/GraSP/frames-001/frames" | grep '^CASE' | xargs -P 8 -I{} sh -c 'tar -C "$1" -cf - "$3" | tar -C "$2" -xf -' _ "$HOMEDATA/GraSP/frames-001/frames" "$LOCAL/GraSP/frames-001/frames" {} || { echo "dataset copy failed"; exit 1; }
N=$(find "$LOCAL/GraSP/frames-001/frames" -type f | wc -l)
[ "$N" -eq 116521 ] || { echo "expected 116521 frame files, found $N"; exit 1; }
say "dataset on local scratch: $N frames, $(du -sh "$LOCAL" | cut -f1)"
export GRASP_DATA_ROOT=$LOCAL/GraSP
export NEIGHBOURS=$HOMEDATA/temporal_neighbors/fold1:$HOMEDATA/temporal_neighbors/fold2

cd "$CODE" || exit 1
# one hour is kept free at the end of the allocation: no task starts that would not finish before then
$PY_MAIN scripts/campaign.py --out "$OUT" --slots-per-gpu 2 --max-share 3 --util-target 50 --budget-hours 28.8 --cpu-slots 2
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
