"""Rung S latency of the real-time addendum (docs/reports/gtbox_sam_protocol.md, 2026-10-05 second): per-frame time of the box-prompted SAM2.1 large, small and
tiny (single pass and flip-averaged, fp32 and bf16) and of the four-member classifier on evenly spaced official-test frames with their ground-truth boxes
(batch of one frame, CUDA-synchronised, after warm-up). Frame decoding is not included.

Usage (titanxp, surgical environment, an idle GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rt_segmenters.py --frames 60 --out docs/reports/realtime/segmenter_latency.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import crop_from_box
from finetune_sam2_gtbox import box_logits, embed
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

CKPT = Path(os.environ.get("SAM2_CKPT_DIR", Path.home() / "sam2" / "checkpoints"))
VARIANTS = {
    "large": (CKPT / "sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    "small": (CKPT / "sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "tiny": (CKPT / "sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
}


def stats(v: list[float]) -> dict:
    v = sorted(v)
    return {"mean_ms": 1000 * statistics.mean(v), "median_ms": 1000 * statistics.median(v), "p95_ms": 1000 * v[int(0.95 * (len(v) - 1))], "n": len(v)}


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--masks", type=Path, default=Path("experiments/gtbox_sam/final/masks.pkl"), help="masks to crop for the classifier stage")
    ap.add_argument("--ensemble-config", type=Path, default=Path("configs/arms/ens4_N_official_s44.yaml"))
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = "cuda"

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictors = {name: SAM2ImagePredictor(build_sam2(cfg, str(ck), device=dev)) for name, (ck, cfg) in VARIANTS.items()}
    for p in predictors.values():
        p.model.eval()
    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = []
    for m in cfg["members"]:
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(dev)
        net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=dev), strict=False)
        members.append((net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))

    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    masks = pickle.loads(args.masks.read_bytes())
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    names = sorted(by_frame)
    pick = [names[i] for i in np.linspace(0, len(names) - 1, args.frames + args.warmup).round().astype(int)]
    frames_root = args.data_root / "frames-001" / "frames"

    rows: dict[str, list[float]] = defaultdict(list)
    n_inst = []
    for k, file_name in enumerate(pick):
        indices = by_frame[file_name]
        frame = np.array(Image.open(frames_root / file_name).convert("RGB"))
        boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (ds.instances[i][2] for i in indices)], dtype=np.float32)
        fboxes = boxes.copy()
        fboxes[:, [0, 2]] = frame.shape[1] - boxes[:, [2, 0]]
        flipped = frame[:, ::-1].copy()
        rec: dict[str, float] = {}

        def sam_pass(name: str, bf16: bool, flip: bool) -> None:
            pred = predictors[name]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                embed(pred, frame, grad=False)
                box_logits(pred, boxes)
                if flip:
                    embed(pred, flipped, grad=False)
                    box_logits(pred, fboxes)

        def classify() -> None:
            with torch.no_grad():
                for idx in indices:
                    if idx not in masks:
                        continue
                    m = mask_codec.decode(masks[idx]).astype(bool)
                    if not m.any():
                        continue
                    for net, transform, letterbox in members:
                        crop = crop_from_box(frame, m, ds.instances[idx][2], letterbox)
                        net(transform(Image.fromarray(crop)).unsqueeze(0).to(dev))

        jobs = []
        for name in VARIANTS:
            jobs += [(f"{name}_fp32_single", lambda n=name: sam_pass(n, False, False)), (f"{name}_bf16_single", lambda n=name: sam_pass(n, True, False)),
                     (f"{name}_fp32_flip", lambda n=name: sam_pass(n, False, True))]
        jobs.append(("classifier_4_members", classify))
        for name, fn in jobs:
            t0 = sync()
            fn()
            rec[name] = sync() - t0
        if k >= args.warmup:
            for name, v in rec.items():
                rows[name].append(v)
            n_inst.append(len(indices))
        print(f"frame {k + 1}/{len(pick)} {len(indices)} instruments", flush=True)
    res = {"gpu": torch.cuda.get_device_name(0), "frames": len(n_inst), "instruments_per_frame_mean": float(np.mean(n_inst)),
           "stages": {n: stats(v) for n, v in rows.items()}}
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
