"""Dumps, for every ground-truth box of a split, the square crop around the box of (image, SAM mask logits, ground-truth mask) for the
mask-refinement network. SAM is the GT-box-prompted SAM2.1-large of scripts/finetune_sam2_gtbox.py (weights.pt), optionally with
flip averaging. Instrument order is GraspRegionDataset's, the same indices scripts/gtbox_sam_masks.py uses.

Crop: square, side 1.25 x the longer box side, centred on the box, resized to --size, zero padded outside the frame (logits padded
with -12). SAM logits are clipped to [-12, 12] and stored as uint8, the ground-truth crop as soft uint8 (bilinear of the binary mask).

Usage (titanxp, surgical environment):
    python scripts/dump_refiner_crops.py --weights experiments/sam2_gtbox/enc4/weights.pt --tta --split fold1 --out experiments/refiner/fold1_enc4.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from finetune_sam2_gtbox import box_logits, embed
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset

LOGIT_CLIP = 12.0
CROP_SCALE = 1.25


def crop_geometry(box_xywh: tuple, scale: float = CROP_SCALE) -> tuple[float, float, float]:
    x, y, w, h = box_xywh
    return x + w / 2, y + h / 2, max(w, h) * scale


def sample_crop(stack: torch.Tensor, cx: float, cy: float, s: float, size: int) -> torch.Tensor:
    """(1, C, H, W) -> (C, size, size): bilinear crop of the square window, zeros outside the frame."""
    h, w = stack.shape[-2:]
    theta = torch.tensor([[[s / w, 0, 2 * cx / w - 1], [0, s / h, 2 * cy / h - 1]]], dtype=torch.float32, device=stack.device)
    grid = F.affine_grid(theta, (1, stack.shape[1], size, size), align_corners=False)
    return F.grid_sample(stack, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--sam-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(args.sam_config, str(args.sam_checkpoint), device=args.device))
    predictor.model.load_state_dict(torch.load(args.weights, map_location=args.device), strict=False)
    predictor.model.eval()

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames_root = args.data_root / "frames-001" / "frames"
    n = len(ds.instances)
    S = args.size
    img_out = np.zeros((n, S, S, 3), np.uint8)
    logit_out = np.zeros((n, S, S), np.uint8)
    gt_out = np.zeros((n, S, S), np.uint8)
    geom = np.zeros((n, 3), np.float32)
    boxes = np.zeros((n, 4), np.float32)
    done = np.zeros(n, bool)
    base_iou = np.zeros(n)
    for k, (file_name, indices) in enumerate(sorted(by_frame.items())):
        if args.limit_frames is not None and k >= args.limit_frames:
            break
        frame = np.array(Image.open(frames_root / file_name).convert("RGB"))
        xyxy = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (ds.instances[i][2] for i in indices)], dtype=np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embed(predictor, frame, grad=False)
            logits = box_logits(predictor, xyxy)[0].float()
            if args.tta:
                embed(predictor, frame[:, ::-1].copy(), grad=False)
                fb = xyxy.copy()
                fb[:, [0, 2]] = frame.shape[1] - xyxy[:, [2, 0]]
                logits = (logits + box_logits(predictor, fb)[0].float().flip(-1)) / 2
        h, w = frame.shape[:2]
        img_t = torch.from_numpy(frame).to(args.device).permute(2, 0, 1)[None].float()
        for j, idx in enumerate(indices):
            gt = decode_instance_mask(ds.instances[idx][1]).astype(bool)
            lg = logits[j, 0].clamp(-LOGIT_CLIP, LOGIT_CLIP)
            m = lg > 0
            gt_t = torch.from_numpy(gt).to(args.device)
            union = (m | gt_t).sum().item()
            base_iou[idx] = (m & gt_t).sum().item() / union if union else 0.0
            cx, cy, s = crop_geometry(ds.instances[idx][2])
            stack = torch.cat([img_t, (lg + LOGIT_CLIP)[None, None], gt_t[None, None].float()], dim=1)
            crop = sample_crop(stack, cx, cy, s, S)
            img_out[idx] = crop[:3].permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()
            logit_out[idx] = ((crop[3] / (2 * LOGIT_CLIP)).clamp(0, 1) * 255).round().byte().cpu().numpy()
            gt_out[idx] = (crop[4].clamp(0, 1) * 255).round().byte().cpu().numpy()
            geom[idx] = (cx, cy, s)
            boxes[idx] = xyxy[j]
            done[idx] = True
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(by_frame)} frames, mean SAM IoU so far {base_iou[done].mean():.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, img=img_out[done], logit=logit_out[done], gt=gt_out[done], geom=geom[done], boxes=boxes[done],
             idx=np.where(done)[0], base_iou=base_iou[done], size=S)
    print(f"wrote {args.out}: {int(done.sum())} instruments, mean SAM mask IoU {base_iou[done].mean():.4f}")


if __name__ == "__main__":
    main()
