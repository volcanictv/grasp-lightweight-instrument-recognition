"""Probe: can SAM3 turn a ground-truth box into the instrument's mask, and how good is it zero-shot against what we already have?

Two ways to prompt SAM3 with a box (transformers >= 5.x, facebook/sam3 weights):
    tracker   Sam3TrackerModel, the SAM2-style interactive predictor: one box in, one mask out.
    concept   Sam3Model with the box as a visual exemplar: it returns every instance of that concept; the instance whose predicted
              box overlaps the ground-truth box most is taken (nothing found = empty mask).
Frames are an evenly spaced sample of one split (fold1 by default, never the test cases). Per-instrument mask IoU is compared with the
fine-tuned SAM2 masks of the same instruments, read from a dump of scripts/dump_refiner_crops.py (its base_iou).

Usage (titanxp, sam3_venv):
    ~/sam3_venv/bin/python scripts/sam3_boxprompt_probe.py --frames 25 --crops experiments/refiner/fold1_enc4.npz --out experiments/sam3/probe_fold1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(m: np.ndarray, gt: np.ndarray) -> float:
    u = np.logical_or(m, gt).sum()
    return float(np.logical_and(m, gt).sum() / u) if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--split", default="fold1")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--crops", type=Path, default=None, help="dump with base_iou of the fine-tuned SAM2 on the same split")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--modes", nargs="*", default=["tracker", "concept"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    names = sorted(by_frame)
    pick = [names[i] for i in np.linspace(0, len(names) - 1, args.frames).round().astype(int)]
    frames_root = args.data_root / "frames-001" / "frames"
    ref = None
    if args.crops:
        z = np.load(args.crops)
        ref = dict(zip(z["idx"].tolist(), z["base_iou"].tolist()))

    results: dict = {"frames": len(pick), "modes": {}}
    for mode in args.modes:
        if mode == "tracker":
            model = Sam3TrackerModel.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
            proc = Sam3TrackerProcessor.from_pretrained(args.model)
        else:
            model = Sam3Model.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
            proc = Sam3Processor.from_pretrained(args.model)
        ious, ref_ious, empty, t0 = [], [], 0, time.time()
        for file_name in pick:
            image = Image.open(frames_root / file_name).convert("RGB")
            indices = by_frame[file_name]
            boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (ds.instances[i][2] for i in indices)]
            gts = [decode_instance_mask(ds.instances[i][1]).astype(bool) for i in indices]
            with torch.no_grad():
                if mode == "tracker":
                    inputs = proc(images=image, input_boxes=[boxes], return_tensors="pt").to(args.device, dtype)
                    out = model(**inputs, multimask_output=False)
                    masks = proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"])[0]
                    preds = [masks[j, 0].numpy().astype(bool) for j in range(len(boxes))]
                else:
                    preds = []
                    for box in boxes:
                        inputs = proc(images=image, input_boxes=[[box]], input_boxes_labels=[[1]], return_tensors="pt").to(args.device)
                        inputs = {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inputs.items()}
                        out = model(**inputs)
                        res = proc.post_process_instance_segmentation(out, threshold=0.05, mask_threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
                        res = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in res.items()}
                        best, best_iou = None, 0.0
                        for k in range(len(res["scores"])):
                            o = box_iou(np.array(box), res["boxes"][k].float().cpu().numpy())
                            if o > best_iou:
                                best, best_iou = k, o
                        if best is None or best_iou < 0.3:
                            empty += 1
                            preds.append(np.zeros_like(gts[0]))
                        else:
                            preds.append(res["masks"][best].cpu().numpy().astype(bool))
            for i, p, g in zip(indices, preds, gts):
                ious.append(mask_iou(p, g))
                if ref is not None and i in ref:
                    ref_ious.append(ref[i])
                else:
                    ref_ious.append(np.nan)
        ious, ref_ious = np.array(ious), np.array(ref_ious)
        ok = ~np.isnan(ref_ious)
        results["modes"][mode] = {"instruments": int(len(ious)), "mean_iou": float(ious.mean()), "iou_ge_0.5": float((ious >= 0.5).mean()),
                                  "iou_ge_0.75": float((ious >= 0.75).mean()), "empty": empty,
                                  "finetuned_sam2_mean_iou_same_instruments": float(ref_ious[ok].mean()) if ok.any() else None,
                                  "seconds_per_frame": (time.time() - t0) / len(pick)}
        print(mode, json.dumps(results["modes"][mode]), flush=True)
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
