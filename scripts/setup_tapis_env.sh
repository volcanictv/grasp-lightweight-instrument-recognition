#!/bin/bash
# Builds ~/tapis_venv for the TAPIS baseline (BCV-Uniandes/GraSP, TAPIS/): Python 3.8, torch 1.9.1 + CUDA 11.1 (the toolkit on titanxp is
# /usr/local/cuda-11.1, which the Titan Xp (sm_61) supports), detectron2 v0.6 compiled against it, and the repository's requirements.
# Runs at low CPU priority so it does not slow the GPU jobs. Usage (titanxp): nohup nice -n 19 scripts/setup_tapis_env.sh > ~/baselines/tapis_setup.log 2>&1 &
set -uo pipefail
say() { echo "[$(date +%H:%M:%S)] $*"; }
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="6.1"
export MAX_JOBS=2
[ -d ~/tapis_venv ] || /usr/bin/python3.8 -m venv ~/tapis_venv
P=~/tapis_venv/bin/pip
$P install -q -U pip wheel setuptools==59.5.0
say "torch"
$P install -q torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
~/tapis_venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
say "requirements"
$P install -q -r ~/baselines/GraSP/TAPIS/requirements.txt
$P install -q 'numpy<1.24' cython
$P install -q 'git+https://github.com/facebookresearch/fvcore' 'git+https://github.com/facebookresearch/fairscale'
say "detectron2 v0.6 (compiles CUDA ops)"
$P install -q 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
~/tapis_venv/bin/python -c "import detectron2, torch;print('detectron2', detectron2.__version__)"
say "setup finished"
