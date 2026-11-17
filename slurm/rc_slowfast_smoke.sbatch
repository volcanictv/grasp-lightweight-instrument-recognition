#!/bin/bash
# Smoke test of the whole SlowFast path on real 30 fps frames: window extraction for three already verified test cases, one training epoch, both test modes, scoring. Short partition, under an hour.
# The training part is built from the test split too, because the train annotations come with the last archive; this checks the machinery, not the numbers.
#SBATCH --job-name=sf-smoke
#SBATCH --account=robot-surg
#SBATCH --partition=onboard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
W=$HOME/grasp_work/slowfast; G=/shared/rc/robot-surg/grasp30fps; PY=$HOME/grasp_work/tapis/venv/bin/python
"$PY" -B "$W/scripts/rc_slowfast_prep.py" --archives "$G/frames" --release-annotations "$W/data/GraSP/annotations" --data "$W/data" --windows "$G/windows" --runs smoke --smoke-from test --jobs 3 || exit 2
EPOCHS=1 WARMUP=0 BS=4 CK_EVERY=1 WORKERS=3 bash "$W/scripts/rc_slowfast_run.sh" smoke
