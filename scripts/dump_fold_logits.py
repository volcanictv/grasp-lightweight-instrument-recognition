"""Dumps, for every ground-truth box of a fold, SAM2 (enc4, flip-averaged) and SAM3 (enc4, flip-averaged) mask logits and the ground-truth mask in a
window around the box, so that post-processing and ensembling choices can be studied offline on the CPU (scripts/postproc_study.py).
Windows are the box grown by --margin pixels, clipped to the frame. Frames are split by --offset/--stride (one process per GPU).

Usage (titanxp, sam3_venv):
    ~/sam3_venv/bin/python scripts/dump_fold_logits.py --split fold1 --offset 0 --stride 2 --device cuda:0 --out experiments/postproc/fold1_part0.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="fold1")
    ap.add_argument("--sam2-weights", type=Path, default=Path("experiments/sam2_gtbox/enc4/weights.pt"))
    ap.add_argument("--sam3-weights", type=Path, default=Path("experiments/sam3/enc4/weights.pt"))
    ap.add_argument("--no-sam3", action="store_true", help="SAM2 only (for folds the SAM3 model was trained on or has not seen)")
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2 = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=args.device))
    sam2.model.load_state_dict(torch.load(args.sam2_weights, map_location=args.device), strict=False)
    sam2.model.eval()
    sam3 = proc = None
    if not args.no_sam3:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        sam3 = Sam3TrackerModel.from_pretrained("facebook/sam3").to(args.device)
        proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        sam3.load_state_dict(torch.load(args.sam3_weights, map_location=args.device), strict=False)
        sam3.eval()

    ds = GraspDetectionDataset(args.data_root, args.split, include_masks=True)
    frames = ds.samples[args.offset::args.stride]
    if args.limit_frames is not None:
        frames = frames[: args.limit_frames]
    rows = []
    for k, (file_name, anns) in enumerate(frames):
        boxes, masks = instances(ds, file_name, anns, None)
        if not len(boxes):
            continue
        image = Image.open(ds.frames_root / file_name).convert("RGB")
        frame = np.array(image)
        h, w = frame.shape[:2]
        fboxes = boxes.copy()
        fboxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embed(sam2, frame, grad=False)
            l2 = box_logits(sam2, boxes)[0].float()
            embed(sam2, frame[:, ::-1].copy(), grad=False)
            l2 = (l2 + box_logits(sam2, fboxes)[0].float().flip(-1)) / 2
        l3 = None
        if sam3 is not None:
            with torch.no_grad():
                a, _ = s3.predict(sam3, proc, image, boxes, args.device, grad=False)
                b, _ = s3.predict(sam3, proc, image.transpose(Image.FLIP_LEFT_RIGHT), fboxes, args.device, grad=False)
                l3 = (a.float() + b.float().flip(-1)) / 2
        for j, gt in enumerate(masks):
            x0, y0 = max(int(boxes[j, 0]) - args.margin, 0), max(int(boxes[j, 1]) - args.margin, 0)
            x1, y1 = min(int(np.ceil(boxes[j, 2])) + args.margin, w), min(int(np.ceil(boxes[j, 3])) + args.margin, h)
            rows.append({"frame": file_name, "box": boxes[j].copy(), "win": (x0, y0, x1, y1), "gt": gt[y0:y1, x0:x1].copy(),
                         "l2": l2[j, 0, y0:y1, x0:x1].cpu().numpy().astype(np.float16),
                         "l3": None if l3 is None else l3[j, 0, y0:y1, x0:x1].cpu().numpy().astype(np.float16)})
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(frames)} frames", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps(rows))
    print(f"wrote {args.out}: {len(rows)} instruments")


if __name__ == "__main__":
    main()
