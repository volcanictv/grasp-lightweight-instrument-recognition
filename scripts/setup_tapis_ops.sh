#!/bin/bash
# Second half of the TAPIS environment (after scripts/setup_tapis_env.sh): timm for the Swin backbone and the compiled multi-scale deformable attention
# op of the Mask2Former pixel decoder, built for the Titan Xp (sm_61) with the CUDA 11.1 toolkit.
# Usage (titanxp): nohup nice -n 10 scripts/setup_tapis_ops.sh > ~/baselines/tapis_ops.log 2>&1 &
set -uo pipefail
say() { echo "[$(date +%H:%M:%S)] $*"; }
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="6.1"
export MAX_JOBS=2
~/tapis_venv/bin/pip install -q timm==0.6.12 einops 2>&1 | tail -1
say "timm installed"
cd ~/baselines/GraSP/TAPIS/region_proposals/mask2former/modeling/pixel_decoder/ops
~/tapis_venv/bin/python setup.py build install > ~/baselines/tapis_ops_build.log 2>&1
tail -n 2 ~/baselines/tapis_ops_build.log
~/tapis_venv/bin/python -c "import MultiScaleDeformableAttention;print('ops ok')"
say "ops finished"
