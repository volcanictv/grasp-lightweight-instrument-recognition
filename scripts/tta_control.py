"""Compute-matched test-time-augmentation control for the multi-frame refinement.

Question: does the gain of refinement come from temporal context (the same instrument seen at other moments) or only from averaging more classifier passes? For every instrument of the official test split
the keyframe's own SAM mask is used for 21 augmented VIEWS of the SAME keyframe (as many classifier passes as the 21 frames of the +-10 window; the SAM2 propagation cost is not spent), the four members of
each classifier seed classify every view, and the result is written in the format of the registered tracked runs, so scripts/gtbox_sam_final_eval.py scores it with the same fusion, gate and metrics.

The 21 views = 3 mask variants x 7 transforms. Mask variants: the SAM mask as it is, dilated by 4 px, dilated by 8 px (a looser mask, like the tracker-style masks the members were trained on).
Transforms (on the transformed crop tensor, on the GPU): identity, horizontal flip, zoom in 1.10, zoom out 0.90, shift (+0.06, -0.06), brightness/contrast change, flip + zoom 1.05 + shift (-0.05, +0.05).
View 0 is the plain keyframe crop. All settings are fixed here, before any score is seen.

Usage (titanxp, surgical environment, repo root):
    CUDA_VISIBLE_DEVICES=1 python scripts/tta_control.py --variant final
then for each seed S (42, 43, 44), tracked dir experiments/gtbox_sam/final/tracked_tta[_s43|_s44]:
    python scripts/gtbox_sam_final_eval.py --masks .../masks.pkl --logits .../logits_sS.npz --tracked-dir <dir> --tau 1.7e-5 --budgets 833 2861 --out docs/reports/gtbox_sam/final_tta_sS.json --save-frames experiments/rescore/tta_sS.frames.pkl
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pycocotools import mask as mask_codec
from scipy import ndimage

from evaluate_temporal_track_ensemble import crop_from_box
from evidential_seeds_e2e_eval import ORDER
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

SEEDS = (42, 43, 44)
DILATIONS = (0, 4, 8)
# (flip, zoom, shift_x, shift_y, contrast, brightness) on the normalised crop tensor
TRANSFORMS = [
    (False, 1.00, 0.00, 0.00, 1.00, 0.00),
    (True, 1.00, 0.00, 0.00, 1.00, 0.00),
    (False, 1.10, 0.00, 0.00, 1.00, 0.00),
    (False, 0.90, 0.00, 0.00, 1.00, 0.00),
    (False, 1.00, 0.06, -0.06, 1.00, 0.00),
    (False, 1.00, 0.00, 0.00, 0.85, -0.15),
    (True, 1.05, -0.05, 0.05, 1.00, 0.00),
]


def augment(x: torch.Tensor, flip: bool, zoom: float, sx: float, sy: float, contrast: float, brightness: float) -> torch.Tensor:
    """x: (1, 3, H, W) normalised crop. Zoom > 1 magnifies the centre; shifts are fractions of the half-width; borders are replicated."""
    if flip:
        x = x.flip(-1)
    if zoom != 1.0 or sx != 0.0 or sy != 0.0:
        theta = torch.tensor([[1.0 / zoom, 0.0, sx], [0.0, 1.0 / zoom, sy]], dtype=x.dtype, device=x.device).unsqueeze(0)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return x * contrast + brightness


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="final")
    ap.add_argument("--limit", type=int, default=0, help="first N instruments only (smoke test)")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tag", default="tta")
    args = ap.parse_args()
    dev = "cuda"
    D = REPO_ROOT / "experiments" / "gtbox_sam" / args.variant
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    frames_root = args.data_root / "frames-001" / "frames"
    masks = pickle.loads((D / "masks.pkl").read_bytes())
    idx = sorted(masks)   # every instrument with a final mask, so that any gate (threshold, top-K, refine-all) can be scored
    if args.limit:
        idx = idx[: args.limit]
    print(f"{len(idx)} instruments, {len(DILATIONS) * len(TRANSFORMS)} views each", flush=True)

    ens = {}
    for s in SEEDS:
        cfg = yaml.safe_load((REPO_ROOT / f"configs/arms/ens4_N_official_s{s}.yaml").read_text())
        by_label = {}
        for m in cfg["members"]:
            net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(dev)
            net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=dev), strict=False)
            by_label[m["label"]] = (net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"], m["image_size"])
        ens[s] = [by_label[k] for k in ORDER]
    tf_by_key: dict[tuple, object] = {}
    for s in SEEDS:
        for _net, tf, lb, size in ens[s]:
            tf_by_key.setdefault((size, lb), tf)

    out = {s: {} for s in SEEDS}
    cache: dict[str, np.ndarray] = {}
    t0 = time.time()

    def decode(path: Path) -> np.ndarray:
        key = str(path)
        if key not in cache:
            if len(cache) > 40:
                cache.clear()
            cache[key] = np.array(Image.open(path).convert("RGB"))
        return cache[key]

    for n, i in enumerate(idx):
        file_name, _seg, box, _label = ds.instances[i]
        case, stem = file_name.split("/")
        frame = decode(frames_root / case / stem)
        mask0 = mask_codec.decode(masks[i]).astype(bool)
        views: dict[tuple, list] = {}   # (size, letterbox) -> list of (1,3,S,S) tensors, 21 in a fixed order: mask variant major, transform minor
        for r in DILATIONS:
            mask = mask0 if r == 0 else ndimage.binary_dilation(mask0, iterations=r)
            crops = {lb: crop_from_box(frame, mask, box, lb) for lb in (True, False)}
            for key, tf in tf_by_key.items():
                base = tf(Image.fromarray(crops[key[1]])).unsqueeze(0).to(dev)
                views.setdefault(key, []).extend(augment(base, *t) for t in TRANSFORMS)
        batches = {key: torch.cat(v, dim=0) for key, v in views.items()}   # (21, 3, S, S)
        with torch.no_grad():
            for s in SEEDS:
                cols = [net(batches[(size, lb)]).float().cpu().numpy() for net, _tf, lb, size in ens[s]]
                out[s][f"det_{i}"] = np.stack(cols, axis=1)   # (views, members, classes)
        del views, batches
        if (n + 1) % 100 == 0:
            gc.collect()
            print(f"{n + 1}/{len(idx)}  {time.time() - t0:.0f}s  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for s in SEEDS:
        d = D / (f"tracked_{args.tag}" + ("" if s == 42 else f"_s{s}"))
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "frames_shard0.npz", **out[s])
        print("wrote", d, len(out[s]))


if __name__ == "__main__":
    main()
