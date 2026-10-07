#!/bin/bash -l
# Setup of the latency run in ~/grasp_work (environments, code, weights, SAM3): CPU only, no GPU requested. Safe to run again: it only does what is missing.
# Submit from ~/grasp_work: sbatch code/slurm/rc_setup.sbatch
# Resources follow the first run (job 21823340: 1 min 52 s, 17 GB of memory used, less than half of 8 cores busy).
# Slack notification when it starts, ends or fails (you must be logged into the RC Slack workspace).
#SBATCH --job-name=grasp-setup
#SBATCH --comment="build python environments and fetch weights in the home directory"
#SBATCH --account=robot-surg
#SBATCH --partition=sporc-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=0-00:30:00
#SBATCH --mail-user=slack:@ab1340
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

bash "$HOME/grasp_work/code/scripts/rc_setup.sh"
