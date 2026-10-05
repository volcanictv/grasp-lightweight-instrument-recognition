"""Upstream segmentor stage with the SAM2 + SAM3 ensemble: every ground-truth box of a split prompts the fine-tuned SAM2.1-large (enc4) and the
fine-tuned SAM3 (enc4), each averaged over the image and its horizontal flip, and the two models' mask logits are averaged. Output format as
scripts/gtbox_sam_masks.py (COCO RLE keyed by GraspRegionDataset instrument index), written per part (--offset/--stride, one process per GPU) and
merged by scripts/merge_gtbox_mask_parts.py. An extra, disclosed run: the protocol's registered rule kept SAM2 on fold1 (ensemble 0.9131 against the 0.9136 bar).

Usage (titanxp, sam3_venv):
    ~/sam3_venv/bin/python scripts/gtbox_sam_masks_ensemble.py --split test --offset 0 --stride 2 --device cuda:0 --out experiments/gtbox_sam/gtft_ens/part0.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_codec

import finetune_sam3_gtbox as s3
from finetune_sam2_gtbox import box_logits, embed
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sam2-weights", type=Path, default=Path("experiments/sam2_gtbox/enc4/weights.pt"))
    ap.add_argument("--sam3-weights", type=Path, default=Path("experiments/sam3/enc4/weights.pt"))
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    sam2 = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=args.device))
    sam2.model.load_state_dict(torch.load(args.sam2_weights, map_location=args.device), strict=False)
    sam2.model.eval()
    sam3 = Sam3TrackerModel.from_pretrained("facebook/sam3").to(args.device)
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    sam3.load_state_dict(torch.load(args.sam3_weights, map_location=args.device), strict=False)
    sam3.eval()

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames_root = args.data_root / "frames-001" / "frames"
    names = sorted(by_frame)[args.offset::args.stride]
    if args.limit_frames is not None:
        names = names[: args.limit_frames]
    masks, ious, labels = {}, {}, {}
    for k, file_name in enumerate(names):
        indices = by_frame[file_name]
        image = Image.open(frames_root / file_name).convert("RGB")
        frame = np.array(image)
        boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (ds.instances[i][2] for i in indices)], dtype=np.float32)
        fboxes = boxes.copy()
        fboxes[:, [0, 2]] = frame.shape[1] - boxes[:, [2, 0]]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embed(sam2, frame, grad=False)
            l2 = box_logits(sam2, boxes)[0].float()
            embed(sam2, frame[:, ::-1].copy(), grad=False)
            l2 = (l2 + box_logits(sam2, fboxes)[0].float().flip(-1)) / 2
        with torch.no_grad():
            a, _ = s3.predict(sam3, proc, image, boxes, args.device, grad=False)
            b, _ = s3.predict(sam3, proc, image.transpose(Image.FLIP_LEFT_RIGHT), fboxes, args.device, grad=False)
            l3 = (a.float() + b.float().flip(-1)) / 2
        avg = (l2 + l3) / 2
        for j, idx in enumerate(indices):
            m = (avg[j, 0] > 0).cpu().numpy()
            gt = decode_instance_mask(ds.instances[idx][1]).astype(bool)
            union = np.logical_or(m, gt).sum()
            masks[idx] = mask_codec.encode(np.asfortranarray(m.astype(np.uint8)))
            ious[idx] = float(np.logical_and(m, gt).sum() / union) if union else 0.0
            labels[idx] = int(ds.instances[idx][3])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(names)} frames, mean IoU so far {np.mean(list(ious.values())):.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps({"masks": masks, "ious": ious, "labels": labels}))
    print(f"wrote {args.out}: {len(masks)} instruments, mean mask IoU {np.mean(list(ious.values())):.4f}")


if __name__ == "__main__":
    main()
