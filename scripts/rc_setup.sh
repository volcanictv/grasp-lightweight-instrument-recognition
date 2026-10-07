#!/bin/bash
# One-time setup of the A100 latency run in the home directory on the RIT cluster. Runs INSIDE a CPU Slurm job (slurm/rc_setup.sbatch), never on the login node.
# Everything is created under $WORK (default ~/grasp_work); nothing else in the home directory is read or changed, and nothing is ever deleted except the temporary Hugging Face token
# file, after the SAM3 download (see below).
#
#   code            the public research repository, branch realtime-gtbox (git clone / fast-forward pull)
#   envs/main       Python 3.11, torch 2.8.0+cu126, SAM-2 (2b90b9f), ultralytics 8.4.168, transformers 5.18.0 (SAM3), the classifier dependencies
#   envs/edgetam    Python 3.11, torch 2.8.0+cu126, EdgeTAM (7711e01) with the one-line perceiver.py patch the titanxp environment also carries
#   checkpoints/sam2   SAM2.1 tiny, small, large from Meta
#   code/weights/evidential_armN_seed44   the four classifier checkpoints from the public Hugging Face repository AryanB005/grasp-instrument-pipeline
#   hf_cache        facebook/sam3 (gated): needs $WORK/.hf_token, a read-only Hugging Face token the user places there; the file is removed when the download is done
# Every download is verified against configs/rc_expected_sha256.txt; the YOLO26s weights and the test-data bundle are copied from the laptop by the user (runbook).
set -euo pipefail
WORK=${WORK:-$HOME/grasp_work}
REPO_URL=${REPO_URL:-https://github.com/volcanictv/grasp-lightweight-instrument-recognition}
BRANCH=${BRANCH:-realtime-gtbox}
say() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "SETUP FAILED: $*" >&2; exit 1; }

case "$WORK" in "$HOME"/*) ;; *) fail "WORK must be inside $HOME";; esac
mkdir -p "$WORK"/{envs,checkpoints/sam2,weights_extra,results,hf_cache,src,logs}
export HF_HOME=$WORK/hf_cache UV_CACHE_DIR=$WORK/uv_cache UV_PYTHON_INSTALL_DIR=$WORK/python PIP_CACHE_DIR=$WORK/pip_cache TMPDIR=${TMPDIR:-/tmp/${USER}_setup_${SLURM_JOB_ID:-manual}}
mkdir -p "$TMPDIR"
say "node $(hostname), free space in home: $(df -h --output=avail "$HOME" | tail -1)"
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 40 ] || fail "less than 40 GB free in $HOME"

say "code"
if [ -d "$WORK/code/.git" ]; then git -C "$WORK/code" fetch --depth 1 origin "$BRANCH" && git -C "$WORK/code" checkout "$BRANCH" && git -C "$WORK/code" merge --ff-only FETCH_HEAD
elif [ -d "$WORK/code/scripts" ]; then say "using the code directory that is already there"
else git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORK/code"; fi
CODE=$WORK/code
git -C "$CODE" rev-parse HEAD > "$WORK/logs/code_commit.txt" 2>/dev/null || echo "no git metadata" > "$WORK/logs/code_commit.txt"

say "uv and Python 3.11 (inside $WORK)"
UV_SHA=9167d72b3319674b6303c4cbe071854bba13ebdf3d76b1a7cbdc175471fb66d6   # uv 0.12.23, x86_64 Linux release tarball; no dependence on the cluster's system python
mkdir -p "$WORK/uvbin"
if [ ! -x "$WORK/uvbin/uv" ]; then
  curl -fsSL -o "$TMPDIR/uv.tar.gz" https://github.com/astral-sh/uv/releases/download/0.12.23/uv-x86_64-unknown-linux-gnu.tar.gz
  echo "$UV_SHA  $TMPDIR/uv.tar.gz" | sha256sum -c - || fail "uv download differs from the known checksum"
  tar xzf "$TMPDIR/uv.tar.gz" -C "$WORK/uvbin" --strip-components=1
fi
UV=$WORK/uvbin/uv
$UV --version
TORCH_ARGS="torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126"  # the CUDA 12.6 build titanxp runs; the A100 driver is far newer than it needs
COMMON="numpy==2.4.6 scipy scikit-learn pillow pyyaml pycocotools matplotlib hydra-core iopath tqdm"

say "environment main"
[ -x "$WORK/envs/main/bin/python" ] || $UV venv --python 3.11 "$WORK/envs/main"   # a second run (for SAM3, once the token file is there) reuses the environment
PY=$WORK/envs/main/bin/python
$UV pip install --python $PY $TORCH_ARGS
$UV pip install --python $PY setuptools wheel $COMMON "ultralytics==8.4.168" "transformers==5.18.0" huggingface_hub
[ -d "$WORK/src/sam2/.git" ] || git clone https://github.com/facebookresearch/sam2 "$WORK/src/sam2"
git -C "$WORK/src/sam2" checkout 2b90b9f
(cd "$WORK/src/sam2" && SAM2_BUILD_CUDA=0 $UV pip install --python $PY --no-build-isolation -e .)  # no nvcc on the nodes; titanxp's install has no CUDA extension either
$UV pip freeze --python $PY > "$WORK/logs/freeze_main.txt"

say "environment edgetam"
[ -x "$WORK/envs/edgetam/bin/python" ] || $UV venv --python 3.11 "$WORK/envs/edgetam"
PYE=$WORK/envs/edgetam/bin/python
$UV pip install --python $PYE $TORCH_ARGS
$UV pip install --python $PYE setuptools wheel $COMMON timm==1.0.30   # timm: EdgeTAM's RepViT image encoder
[ -d "$WORK/EdgeTAM/.git" ] || git clone https://github.com/facebookresearch/EdgeTAM "$WORK/EdgeTAM"
git -C "$WORK/EdgeTAM" checkout 7711e01
sed -i 's/\.expand(B, -1, -1)\.view(-1, 1, C)/.expand(B, -1, -1).reshape(-1, 1, C)/' "$WORK/EdgeTAM/sam2/modeling/perceiver.py"
[ "$(git -C "$WORK/EdgeTAM" diff --numstat | awk '{print $1"+"$2}')" = "1+1" ] || fail "the EdgeTAM patch did not apply as exactly one changed line"
(cd "$WORK/EdgeTAM" && SAM2_BUILD_CUDA=0 $UV pip install --python $PYE --no-build-isolation -e .)
$UV pip freeze --python $PYE > "$WORK/logs/freeze_edgetam.txt"

say "SAM2.1 checkpoints"
for v in tiny small large; do
  [ -s "$WORK/checkpoints/sam2/sam2.1_hiera_$v.pt" ] || curl -fL --retry 3 -o "$WORK/checkpoints/sam2/sam2.1_hiera_$v.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_$v.pt"
done

say "classifier checkpoints (public Hugging Face repository)"
$PY - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("AryanB005/grasp-instrument-pipeline", allow_patterns=["weights/evidential_armN_seed44/*"], local_dir="$CODE")
EOF

say "SAM3 (gated)"
if [ -e "$WORK/hf_cache/hub/models--facebook--sam3/snapshots" ] && [ -n "$(ls -A "$WORK/hf_cache/hub/models--facebook--sam3/snapshots" 2>/dev/null)" ]; then
  say "SAM3 already in the cache, nothing to download"
elif [ -s "$WORK/.hf_token" ]; then
  HF_TOKEN=$(cat "$WORK/.hf_token") $PY - <<EOF
import os
from huggingface_hub import snapshot_download
snapshot_download("facebook/sam3", token=os.environ["HF_TOKEN"])
EOF
  rm -f "$WORK/.hf_token"  # the token leaves the cluster again
  say "SAM3 downloaded, token file removed"
else
  say "no $WORK/.hf_token: SAM3 NOT downloaded (the latency run needs it)"
fi

say "verify checksums"
WORK=$WORK bash "$CODE/scripts/rc_verify.sh" lenient || fail "a downloaded file differs from its known checksum"

say "import smoke tests"
cd "$TMPDIR"  # a neutral directory: a folder named sam2 in the working directory would shadow the installed packages
$PY -c "import torch, sam2, ultralytics, transformers, pycocotools; from transformers import Sam3TrackerModel; print(torch.__version__, torch.version.cuda, transformers.__version__, ultralytics.__version__)"
$PYE -c "import torch, sam2; from sam2.build_sam import build_sam2_video_predictor; print(torch.__version__, sam2.__file__)"
say "model build tests (CPU): every model the latency run loads must construct here, so a missing package fails now and not inside the GPU job"
$PYE -c "
from sam2.build_sam import build_sam2_video_predictor
build_sam2_video_predictor('configs/edgetam.yaml', '$WORK/EdgeTAM/checkpoints/edgetam.pt', device='cpu'); print('EdgeTAM builds')"
HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 $PY -c "
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
for v, c in (('tiny', 't'), ('large', 'l')):
    build_sam2(f'configs/sam2.1/sam2.1_hiera_{c}.yaml', '$WORK/checkpoints/sam2/sam2.1_hiera_' + v + '.pt', device='cpu')
build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', '$WORK/checkpoints/sam2/sam2.1_hiera_large.pt', device='cpu'); print('SAM2 tiny, large and the video predictor build')
from transformers import Sam3TrackerModel, Sam3TrackerProcessor
Sam3TrackerModel.from_pretrained('facebook/sam3'); Sam3TrackerProcessor.from_pretrained('facebook/sam3'); print('SAM3 loads offline from the cache')
from ultralytics import YOLO; print('ultralytics imports')"
case "$TMPDIR" in /tmp/${USER}_setup_*) rm -rf "$TMPDIR";; esac  # this job's private temp directory
say "setup done"
du -sh "$WORK" 2>/dev/null | tail -1
