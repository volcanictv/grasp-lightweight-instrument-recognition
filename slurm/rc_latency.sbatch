#!/bin/bash -l
# The A100 latency measurements of the paper (component stages and whole pipelines, scripts/rc_latency_run.sh): ONE A100, nothing else on the GPU; expected 25 to 40 minutes, 45 allowed.
# Submit from ~/grasp_work after rc_setup.sbatch finished: sbatch code/slurm/rc_latency.sbatch
# Slack notification when it starts, ends or fails (you must be logged into the RC Slack workspace).
#SBATCH --job-name=grasp-latency
#SBATCH --comment="A100 latency benchmark: stage times and whole pipelines, one gpu"
#SBATCH --account=robot-surg
#SBATCH --partition=onboard
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=0-00:45:00
#SBATCH --mail-user=slack:@ab1340
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export WORK=$HOME/grasp_work HF_HOME=$HOME/grasp_work/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # SAM3 comes from the cache filled by rc_setup.sbatch
export BUDGET_S=2450   # seconds: no step starts or runs past this, the 45 minute limit keeps a margin for the summary and the job epilogue
export PY_MAIN=$WORK/envs/main/bin/python PY_ET=$WORK/envs/edgetam/bin/python
bash "$WORK/code/scripts/rc_latency_run.sh"
