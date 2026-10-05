"""Upstream segmentor stage of the PI's direction: every ground-truth bounding box of the official test prompts a SAM-family
model, which returns that instrument's mask. No class information enters here; the classifier comes after.

Variants: zero-shot SAM2.1-large (no test exposure) and the SAM2.1-large with the mask decoder fine-tuned on GraSP (our best
SAM model, decoder_best.pt from scripts/finetune_sam2_decoder.py; its checkpoint was chosen on validation loss when validation
meant the official test cases, so the zero-shot variant is the clean one). Writes the masks as COCO RLE keyed by instance index
and the per-instrument IoU against the ground-truth mask.

Usage (titanxp, surgical environment):
    python scripts/gtbox_sam_masks.py --variant finetuned --decoder-checkpoint <decoder_best.pt> --out-dir experiments/gtbox_sam/finetuned
    python scripts/gtbox_sam_masks.py --variant zeroshot --out-dir experiments/gtbox_sam/zeroshot
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_codec

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["zeroshot", "finetuned", "gtft"])
    ap.add_argument("--weights", type=Path, default=None, help="gtft: weights.pt of scripts/finetune_sam2_gtbox.py")
    ap.add_argument("--tta", action="store_true", help="gtft: average the mask logits of the image and its horizontal flip")
    ap.add_argument("--decoder-checkpoint", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sam-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(args.sam_config, str(args.sam_checkpoint), device=args.device))
    if args.variant == "finetuned":
        predictor.model.sam_mask_decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=args.device))
    if args.variant == "gtft":
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from finetune_sam2_gtbox import box_logits, embed
        predictor.model.load_state_dict(torch.load(args.weights, map_location=args.device), strict=False)
    predictor.model.eval()

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _seg, _box, _label) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames_root = args.data_root / "frames-001" / "frames"
    masks: dict[int, dict] = {}
    ious = np.zeros(len(ds.instances))
    for n, (file_name, indices) in enumerate(sorted(by_frame.items())):
        if args.limit_frames is not None and n >= args.limit_frames:
            break
        frame = np.array(Image.open(frames_root / file_name).convert("RGB"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.variant == "gtft":
                boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (ds.instances[i][2] for i in indices)], dtype=np.float32)
                embed(predictor, frame, grad=False)
                logits = box_logits(predictor, boxes)[0].float()
                if args.tta:
                    embed(predictor, frame[:, ::-1].copy(), grad=False)
                    fboxes = boxes.copy()
                    fboxes[:, [0, 2]] = frame.shape[1] - boxes[:, [2, 0]]
                    logits = (logits + box_logits(predictor, fboxes)[0].float().flip(-1)) / 2
                frame_masks = {i: (lg[0] > 0).cpu().numpy() for i, lg in zip(indices, logits)}
            else:
                predictor.set_image(frame)
            for idx in indices:
                _fn, seg, (x, y, w, h), _label = ds.instances[idx]
                if args.variant == "gtft":
                    m = frame_masks[idx]
                else:
                    pred, _score, _ = predictor.predict(box=np.array([x, y, x + w, y + h], dtype=np.float32), multimask_output=False)
                    m = pred[0] > 0
                masks[idx] = mask_codec.encode(np.asfortranarray(m.astype(np.uint8)))
                gt = decode_instance_mask(seg).astype(bool)
                union = float(np.logical_or(m, gt).sum())
                ious[idx] = float(np.logical_and(m, gt).sum()) / union if union else 0.0
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(by_frame)} frames, mean IoU so far {ious[ious > 0].mean():.4f}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "masks.pkl").write_bytes(pickle.dumps(masks))
    done = np.array(sorted(masks))
    labels = np.array([ds.instances[i][3] for i in done])
    summary = {"variant": args.variant, "instruments": int(len(done)), "mean_mask_iou": float(ious[done].mean()),
               "iou_ge_0.5": float((ious[done] >= 0.5).mean()), "iou_ge_0.75": float((ious[done] >= 0.75).mean()),
               "per_class_mean_iou": {CLASSES[c]: float(ious[done][labels == c].mean()) for c in range(7) if (labels == c).any()}}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    np.save(args.out_dir / "ious.npy", ious)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
