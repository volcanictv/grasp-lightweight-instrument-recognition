#!/bin/bash -l
# The A100 latency measurements of the paper (component stages and whole pipelines, scripts/rc_latency_run.sh): ONE A100, nothing else on the GPU; about 30 minutes expected, 55 allowed.
# Submit from ~/grasp_work after rc_setup.sbatch finished: sbatch code/slurm/rc_latency.sbatch
# Slack notification when it starts, ends or fails (you must be logged into the RC Slack workspace).
#SBATCH --job-name=grasp-latency
#SBATCH --comment="A100 latency benchmark: stage times and whole pipelines, one gpu"
#SBATCH --account=robot-surg
#SBATCH --partition=onboard
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=0-00:55:00
#SBATCH --mail-user=slack:@ab1340
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export WORK=$HOME/grasp_work HF_HOME=$HOME/grasp_work/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # SAM3 comes from the cache filled by rc_setup.sbatch
export PY_MAIN=$WORK/envs/main/bin/python PY_ET=$WORK/envs/edgetam/bin/python
bash "$WORK/code/scripts/rc_latency_run.sh"
