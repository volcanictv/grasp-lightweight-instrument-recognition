#!/bin/bash -l
# Short probe of an RC compute node: GPU, driver, CPU, memory, local disk, internet. Debug partition (debugging only, 1 day max); this one runs a couple of minutes.
#SBATCH --job-name=rc-probe
#SBATCH --comment="probe a100 node: gpu, disk, internet"
#SBATCH --account=robot-surg
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=0-00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "== node"; hostname; date; uname -m; nproc; free -g | head -2
echo "== gpu"; nvidia-smi -L; nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
echo "== cuda toolkit on the node"; ls /usr/local | grep -i cuda; which nvcc 2>&1 | head -1
echo "== local disk"; df -h /tmp /dev/shm 2>&1 | tail -3
echo "== internet from the compute node"
curl -s -m 8 -o /dev/null -w "huggingface %{http_code}\n" https://huggingface.co
curl -s -m 8 -o /dev/null -w "pypi %{http_code}\n" https://pypi.org
curl -s -m 8 -o /dev/null -w "github %{http_code}\n" https://github.com
curl -s -m 8 -o /dev/null -w "download.pytorch.org %{http_code}\n" https://download.pytorch.org
echo "== slurm env"; env | grep -E "^SLURM_(JOB_ID|JOB_GPUS|CPUS_ON_NODE|MEM_PER_NODE|GPUS)" | sort
echo "done"
