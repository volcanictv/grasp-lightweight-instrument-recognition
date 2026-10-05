"""Applies the mask-refinement network to crops dumped by scripts/dump_refiner_crops.py, pastes the refined masks back at full
resolution inside each ground-truth box, and scores them against the ground truth. The same paste-back is applied to SAM's own
logits as the reference ("sam clipped"), so the comparison isolates what the refiner changes. With --save-masks the refined masks
are written as COCO RLE keyed by instrument index (masks.pkl plus summary.json), the input format of scripts/run_gtbox_sam_pipeline.sh.

Usage (titanxp, surgical environment):
    python scripts/apply_mask_refiner.py --crops experiments/refiner/fold1_enc4.npz --weights experiments/refiner/run1/weights.pt --split fold1
    python scripts/apply_mask_refiner.py --crops experiments/refiner/test_enc4.npz --weights ... --split test --save-masks --out-dir experiments/gtbox_sam/gtft_ref
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_codec

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.models.mask_refiner import LOGIT_CLIP, MaskRefiner, box_channel

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def paste(logit_crop: torch.Tensor, geom: np.ndarray, box: np.ndarray, hw: tuple[int, int], device: str) -> np.ndarray:
    """(S,S) crop logits -> full-frame boolean mask, positive logits inside the box only."""
    h, w = hw
    cx, cy, s = (float(v) for v in geom)
    x0, y0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x1, y1 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
    out = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0:
        return out
    px = (torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5 - cx) / (s / 2)
    py = (torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5 - cy) / (s / 2)
    grid = torch.stack(torch.meshgrid(px, py, indexing="xy"), dim=-1)[None]
    up = F.grid_sample(logit_crop[None, None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]
    xs = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
    ys = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
    inside = ((xs >= box[0]) & (xs <= box[2]))[None] & ((ys >= box[1]) & (ys <= box[3]))[:, None]
    out[y0:y1, x0:x1] = ((up > 0) & inside).cpu().numpy()
    return out


def iou(m: np.ndarray, gt: np.ndarray) -> float:
    u = np.logical_or(m, gt).sum()
    return float(np.logical_and(m, gt).sum() / u) if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    z = np.load(args.crops)
    size = int(z["size"])
    model = MaskRefiner(pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location=args.device))
    model.eval()
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    idxs = z["idx"]
    raw, ref, sam_clip, masks = np.zeros(len(idxs)), np.zeros(len(idxs)), np.zeros(len(idxs)), {}
    raw[:] = z["base_iou"]
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            img = torch.from_numpy(z["img"][i]).to(args.device).permute(2, 0, 1)[None].float() / 255
            sam = (torch.from_numpy(z["logit"][i]).to(args.device).float()[None, None] / 255) * 2 * LOGIT_CLIP - LOGIT_CLIP
            geom, box = z["geom"][i], z["boxes"][i]
            bx = box_channel(torch.from_numpy(box)[None].to(args.device), torch.from_numpy(geom)[None].to(args.device), size)
            out = model(img, sam, bx)[0, 0]
            gt = decode_instance_mask(ds.instances[int(idx)][1]).astype(bool)
            m = paste(out, geom, box, gt.shape, args.device)
            ref[i] = iou(m, gt)
            sam_clip[i] = iou(paste(sam[0, 0], geom, box, gt.shape, args.device), gt)
            if args.save_masks:
                masks[int(idx)] = mask_codec.encode(np.asfortranarray(m.astype(np.uint8)))
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(idxs)}: sam {raw[:i + 1].mean():.4f} sam clipped {sam_clip[:i + 1].mean():.4f} refined {ref[:i + 1].mean():.4f}", flush=True)
    labels = np.array([ds.instances[int(i)][3] for i in idxs])
    summary = {"instruments": int(len(idxs)), "mean_mask_iou": float(ref.mean()), "iou_ge_0.5": float((ref >= 0.5).mean()),
               "iou_ge_0.75": float((ref >= 0.75).mean()), "sam_mean_mask_iou": float(raw.mean()),
               "sam_clipped_mean_mask_iou": float(sam_clip.mean()),
               "per_class_mean_iou": {CLASSES[c]: float(ref[labels == c].mean()) for c in range(7) if (labels == c).any()},
               "per_class_sam_mean_iou": {CLASSES[c]: float(raw[labels == c].mean()) for c in range(7) if (labels == c).any()}}
    print(json.dumps(summary, indent=1))
    if args.save_masks:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "masks.pkl").write_bytes(pickle.dumps(masks))
        (args.out_dir / "summary.json").write_text(json.dumps({"variant": "gtft_ref", **summary}, indent=1))
        full = np.zeros(len(ds.instances))
        full[idxs] = ref
        np.save(args.out_dir / "ious.npy", full)
    elif args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
