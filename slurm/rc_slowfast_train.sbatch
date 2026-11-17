#!/bin/bash
# SlowFast baseline on the 30 fps GraSP windows, one array task per run: 0 official split, 1 fold1 (held out), 2 fold2 (held out). 1 A100 each.
# Thin wrapper on purpose: the logic lives in ~/grasp_work/slowfast/scripts/rc_slowfast_run.sh, which is read when the task starts, so it can be corrected while the job is still queued.
# Submit after the staging job:  sbatch --dependency=afterok:<prep job id> --mail-user=slack:@<RC username> slurm/rc_slowfast_train.sbatch   (smoke: SMOKE=1, see rc_slowfast_smoke.sbatch)
#SBATCH --job-name=sf-grasp
#SBATCH --account=robot-surg
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=0-08:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -uo pipefail
RUNS=(official fold1 fold2)
bash "$HOME/grasp_work/slowfast/scripts/rc_slowfast_run.sh" "${RUNS[$SLURM_ARRAY_TASK_ID]}"
