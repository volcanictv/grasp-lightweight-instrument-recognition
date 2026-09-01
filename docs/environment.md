# Environment

Two separate machines are in play. Keep their status straight — code that
runs on one doesn't confirm anything about the other.

## Dev laptop (this machine, where Milestones 0-2 were built)

- Windows 11, Python 3.14.6.
- **Has a local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, driver
  610.74, compute capability (8, 9) (Ada Lovelace, has tensor cores — unlike
  the workstation's Pascal Titan Xps).** This was wrongly recorded as "no
  local GPU" until 2026-08-31; the hardware was always present, but only a
  CPU-only torch build (`2.13.0+cpu`) had been installed, so
  `torch.cuda.is_available()` returned `False` for reasons unrelated to the
  hardware. Do not trust that check alone on a new machine — verify with a
  real op, as below.
- `torch==2.13.0+cu126`, `torchvision==0.28.0+cu126` installed via
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.
  Ada Lovelace has no legacy sm-support constraint the way the Titan Xp's
  Pascal does (see the workstation section below) — any recent CUDA build
  works; cu126 was picked only for consistency with the workstation's pinned
  version, not because it's required here.
- Verified 2026-08-31 with the same real-tensor-op standard as the
  workstation (matmul + conv2d, not just `is_available()`):
  ```
  2.13.0+cu126 True (8, 9)
  NVIDIA GeForce RTX 4060 Laptop GPU
  matmul ok -798957.25
  conv2d ok (8, 16, 224, 224)
  ```
- Measured training throughput (MobileNetV3-Small, fine-tuned, batch 32,
  4 loader workers, uncached `GraSp` root, loader+compute combined):
  74.4 img/s on CPU vs. 292.6 img/s on the RTX 4060 -- about 4x. Still
  behind the workstation's cached-loader Task A epochs (8.5-11s, vs. an
  extrapolated ~20s here) because this machine doesn't have its own resized
  frame cache built (Milestone 1's cache is workstation-only, see
  README.md) and this comparison didn't tune worker count for this
  machine's CPU. Not apples-to-apples yet -- if this machine becomes a
  real second training node, build its own cache and re-measure before
  drawing conclusions about which machine is actually faster for a given
  run.
- This machine can now train, not just prototype the data pipeline --
  CLAUDE.md's "never assume local GPU access" instruction was written before
  this was known and should be read as "don't assume a GPU on *some*
  machine you haven't checked," not as a permanent constraint on this one.
  It remains true that this machine's disk/CPU numbers (loader throughput,
  etc.) aren't the workstation's numbers and shouldn't be substituted for
  them.

## Titan Xp workstation (2x Titan Xp, Ubuntu) — Milestone -1, done 2026-08-31

Reached over SSH (`ssh titanxp`, alias configured in `~/.ssh/config` on the
dev laptop, key-based auth, password auth should be disabled on the
workstation next time someone's physically at it).

- OS: Ubuntu 20.04.4 LTS, kernel 5.15.0-102-generic.
- GPU driver: 575.57.08, reports CUDA 12.9 as the max supported runtime.
  `nvidia-smi` was already working correctly before any of this — the
  "GPU detection is broken" framing in CLAUDE.md did not match reality by
  the time this was checked. What was actually broken: no PyTorch was
  installed at all, and the system Python (3.8.10) is too old for current
  PyTorch wheels.
- System Python untouched. Installed **Miniconda** self-contained under
  `~/miniconda3` (no sudo, no system package changes — this is shared lab
  hardware, so system-wide Python/apt changes were avoided in favor of a
  fully reversible per-user install). Created a `surgical` conda env with
  the `conda-forge` channel (the `defaults`/`pkgs/main` channels now gate
  behind an Anaconda ToS click-through; conda-forge sidesteps that).
- Python 3.11 (via the `surgical` conda env).
- **`torch==2.8.0+cu126`, `torchvision==0.23.0+cu126`**, installed via:
  ```
  ~/miniconda3/envs/surgical/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu126
  ```
  This combination matters and is not the obvious default: PyTorch dropped
  Pascal/Maxwell/Volta support in the **cu128 and cu129** wheel builds
  starting with the 2.8/2.9 release line (see pytorch/pytorch#157517), but
  the **cu126** build of the same 2.8.0 release still ships sm_61 kernels.
  Picking cu128 or cu129 here — which would otherwise look like the
  "newer, more correct" choice given the driver reports CUDA 12.9 — silently
  produces a torch install that passes `torch.cuda.is_available()` but
  fails at the first real kernel launch. Do not upgrade the CUDA build
  variant without re-verifying with the tensor-op command below.
- cuDNN 9.10.2.21 (bundled via the `nvidia-cudnn-cu12` pip package), enabled.

**Verification actually run** (not just `torch.cuda.is_available()`), both
GPUs, both a raw matmul and a cuDNN conv2d (the op MobileNetV3 will
actually use):

```
~/miniconda3/envs/surgical/bin/python -c "
import torch, torch.nn as nn
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))
a = torch.randn(4096, 4096, device='cuda:0'); b = torch.randn(4096, 4096, device='cuda:0')
c = a @ b; torch.cuda.synchronize(); print('matmul ok', c.sum().item())
for idx in range(torch.cuda.device_count()):
    dev = f'cuda:{idx}'
    x = torch.randn(8, 3, 224, 224, device=dev)
    y = nn.Conv2d(3, 16, 3, padding=1).to(dev)(x)
    torch.cuda.synchronize(dev)
    print(dev, torch.cuda.get_device_name(idx), tuple(y.shape))
"
```

Output:

```
2.8.0+cu126 True (6, 1)
matmul ok -95386.25
cuda:0 NVIDIA TITAN Xp (8, 16, 224, 224)
cuda:1 NVIDIA TITAN Xp (8, 16, 224, 224)
```

Compute capability `(6, 1)` confirms Pascal/sm_61, matching CLAUDE.md's
hardware note. Both GPUs run cuDNN convolutions successfully.

AMP was not benchmarked for throughput per CLAUDE.md's standing note
(Pascal has no tensor cores, so AMP saves memory only, not speed) — don't
report an AMP speedup number without re-reading that note.

**Reproducing this env on a fresh shell:** activate with
`~/miniconda3/envs/surgical/bin/python` directly, or
`source ~/miniconda3/bin/activate surgical`. `conda init` was not run, so
plain interactive shells don't auto-activate anything — deliberate, to
avoid changing this shared machine's default shell behavior for other
users.
