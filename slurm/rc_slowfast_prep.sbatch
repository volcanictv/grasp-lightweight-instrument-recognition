#!/bin/bash
# Stage the 30 fps frames the SlowFast loader needs (16 frames per annotated keyframe, about 20 GB) out of the verified per-case archives and build the run directories. CPU only.
# Extracts the 30 fps release annotations first. Refuses to run until every archive has its .ok marker (the download job writes it after a size and gzip check).
#SBATCH --job-name=sf-prep
#SBATCH --account=robot-surg
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=0-04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
W=$HOME/grasp_work/slowfast; G=/shared/rc/robot-surg/grasp30fps; PY=$HOME/grasp_work/tapis/venv/bin/python
RUNS=${RUNS:-official,fold1,fold2}
ANN=$W/data/release_annotations
if [ ! -d "$ANN" ]; then
  [ -f "$G/annotations.tar.gz.ok" ] || { echo "annotations archive not verified yet"; exit 2; }
  mkdir -p "$ANN" && tar -xzf "$G/annotations.tar.gz" -C "$ANN" || exit 3
fi
find "$ANN" -maxdepth 3 -name "*.json" | sort | head -20
"$PY" -B "$W/scripts/rc_slowfast_prep.py" --archives "$G/frames" --release-annotations "$ANN" --data "$W/data" --windows "$G/windows" --runs "$RUNS" --jobs 4
