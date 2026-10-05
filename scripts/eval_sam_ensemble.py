"""Scores, on a fold, the fine-tuned SAM2 (enc4, flip-averaged), the fine-tuned SAM3 (flip-averaged) and their ensemble (mean of the two
models' flip-averaged mask logits), per instrument, from the same frames and boxes. The protocol addendum (2026-10-03, third) registers the
ensemble as a candidate segmentor. Frames are split by --offset/--stride so two processes (one per GPU) can share the fold; their outputs
are merged by scripts/merge_sam_ensemble.py.

Usage (titanxp, sam3_venv):
    ~/sam3_venv/bin/python scripts/eval_sam_ensemble.py --sam3-weights experiments/sam3/enc4/weights.pt --offset 0 --stride 2 --device cuda:0 --out experiments/sam3/ens_fold1_part0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image

import finetune_sam3_gtbox as s3
from finetune_sam2_gtbox import box_logits, embed, instances
from surgical_ai.data.detection_dataset import GraspDetectionDataset


def iou(m: np.ndarray, gt: np.ndarray) -> float:
    u = np.logical_or(m, gt).sum()
    return float(np.logical_and(m, gt).sum() / u) if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="fold1")
    ap.add_argument("--sam2-weights", type=Path, default=Path("experiments/sam2_gtbox/enc4/weights.pt"))
    ap.add_argument("--sam3-weights", type=Path, required=True)
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
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

    ds = GraspDetectionDataset(args.data_root, args.split, include_masks=True)
    rows = {"sam2": [], "sam3": [], "ensemble": []}
    frames = ds.samples[args.offset::args.stride]
    for k, (file_name, anns) in enumerate(frames):
        if k and k % 100 == 0:
            print(f"  {k}/{len(frames)} frames: sam2 {np.mean(rows['sam2']):.4f} sam3 {np.mean(rows['sam3']):.4f} ensemble {np.mean(rows['ensemble']):.4f}", flush=True)
        boxes, masks = instances(ds, file_name, anns, None)
        if not len(boxes):
            continue
        image = Image.open(ds.frames_root / file_name).convert("RGB")
        frame = np.array(image)
        w = frame.shape[1]
        fboxes = boxes.copy()
        fboxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embed(sam2, frame, grad=False)
            l2 = box_logits(sam2, boxes)[0].float()
            embed(sam2, frame[:, ::-1].copy(), grad=False)
            l2 = (l2 + box_logits(sam2, fboxes)[0].float().flip(-1)) / 2
        with torch.no_grad():
            a, _ = s3.predict(sam3, proc, image, boxes, args.device, grad=False)
            b, _ = s3.predict(sam3, proc, image.transpose(Image.FLIP_LEFT_RIGHT), fboxes, args.device, grad=False)
            l3 = (a.float() + b.float().flip(-1)) / 2
        for j, gt in enumerate(masks):
            for name, lg in (("sam2", l2[j, 0]), ("sam3", l3[j, 0]), ("ensemble", (l2[j, 0] + l3[j, 0]) / 2)):
                rows[name].append(iou((lg > 0).cpu().numpy(), gt))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows))
    print({k: round(float(np.mean(v)), 4) for k, v in rows.items()}, "instruments", len(rows["sam2"]))


if __name__ == "__main__":
    main()
