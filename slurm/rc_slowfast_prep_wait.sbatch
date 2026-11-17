#!/bin/bash
# Same staging as rc_slowfast_prep.sbatch, but it starts without a job dependency and waits (CPU only, up to 3 hours) for every archive to carry its .ok marker, so extraction begins the minute the download ends.
#SBATCH --job-name=sf-prep-wait
#SBATCH --account=robot-surg
#SBATCH --partition=sporc-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=0-05:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
G=${GRASP30FPS_DIR:?set GRASP30FPS_DIR to the project-storage directory that holds the 30 fps archives}
NEED=(annotations.tar.gz)
for c in 001 002 003 004 007 014 015 021 041 047 050 051 053; do NEED+=("frames/CASE$c.tar.gz"); done
for i in $(seq 1 180); do
  ok=1; for f in "${NEED[@]}"; do [ -f "$G/$f.ok" ] || ok=0; done
  [ "$ok" = 1 ] && break; sleep 60
done
[ "$ok" = 1 ] || { echo "archives still not verified after 3 hours"; exit 2; }
echo "all archives verified at $(date +%T); starting staging"
bash "$HOME/grasp_work/slowfast/slurm/rc_slowfast_prep.sbatch"
