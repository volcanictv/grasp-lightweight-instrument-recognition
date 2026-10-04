"""Times the stages of one SAM3 box-prompted prediction (processor, image encoder, prompt + decoder, mask post-processing, to numpy) on a few
fold1 frames, to see why scripts/finetune_sam3_gtbox.py is slow. Usage (titanxp, sam3_venv):
    ~/sam3_venv/bin/python scripts/time_sam3_predict.py --device cuda:1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image

from finetune_sam2_gtbox import instances
from surgical_ai.data.detection_dataset import GraspDetectionDataset


def sync(device: str) -> float:
    torch.cuda.synchronize(device)
    return time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--fp32", action="store_true", help="encoder in fp32 without autocast")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(args.device)
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    model.requires_grad_(False)
    if not args.fp32:
        model.vision_encoder.half()
    model.eval()
    ds = GraspDetectionDataset(args.data_root, "fold1", include_masks=True)
    rows = []
    for k, (file_name, anns) in enumerate(ds.samples[: args.frames + 1]):
        boxes, masks = instances(ds, file_name, anns, None)
        if not len(boxes):
            continue
        t0 = sync(args.device)
        image = Image.open(ds.frames_root / file_name).convert("RGB")
        t1 = sync(args.device)
        inputs = proc(images=image, input_boxes=[boxes.tolist()], return_tensors="pt")
        pv = inputs["pixel_values"].to(args.device)
        t2 = sync(args.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=not args.fp32):
            emb = model.get_image_embeddings(pv)
        t3 = sync(args.device)
        emb = [e.float() for e in emb]
        with torch.no_grad():
            out = model(image_embeddings=emb, input_boxes=inputs["input_boxes"].to(args.device).float(), multimask_output=False)
        t4 = sync(args.device)
        logits = proc.post_process_masks(out.pred_masks, inputs["original_sizes"], binarize=False)[0]
        t5 = sync(args.device)
        _ = (logits[:, 0] > 0).cpu().numpy()
        t6 = sync(args.device)
        if k > 0:  # the first frame includes warm-up
            rows.append([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5])
    r = np.array(rows).mean(axis=0)
    print("seconds per frame: load %.2f processor %.2f encoder %.2f decoder %.2f postprocess %.2f to-numpy %.2f total %.2f" % (*r, r.sum()))
    print("pixel_values", tuple(pv.shape), "boxes", len(boxes), "torch threads", torch.get_num_threads())


if __name__ == "__main__":
    main()
