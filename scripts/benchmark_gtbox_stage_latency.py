"""Stage latencies of the GT-box pipeline's own components on this machine (batch of one frame, CUDA-synchronised, after warm-up), on evenly
spaced official-test frames with their ground-truth boxes: the fine-tuned SAM2 (single pass and flip-averaged), the fine-tuned SAM3 (single and
flip-averaged), and the four-member evidential classifier (crop plus four forwards per instrument, summed per frame). The detector and the
SAM2-large tracker are not re-measured here; their frame-level latencies come from the earlier latency matrix on the same GPU
(docs/reports/causal_realtime/latency_frames/*.json).

Usage (titanxp, sam3_venv, an idle GPU):
    CUDA_VISIBLE_DEVICES=0 ~/sam3_venv/bin/python scripts/benchmark_gtbox_stage_latency.py --frames 40 --out docs/reports/latency/our_stages.json
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

import finetune_sam3_gtbox as s3
from evaluate_temporal_track_ensemble import crop_from_box
from finetune_sam2_gtbox import box_logits, embed
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def stats(v: list[float]) -> dict:
    v = sorted(v)
    return {"mean_ms": 1000 * statistics.mean(v), "median_ms": 1000 * statistics.median(v), "p95_ms": 1000 * v[int(0.95 * (len(v) - 1))], "n": len(v)}


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--masks", type=Path, default=Path("experiments/gtbox_sam/gtft_ens/masks.pkl"), help="masks to crop for the classifier stage")
    ap.add_argument("--ensemble-config", type=Path, default=Path("configs/evidential/ens_E_official_s43_lam0p01a10.yaml"))
    ap.add_argument("--sam2-weights", type=Path, default=Path("experiments/sam2_gtbox/enc4/weights.pt"))
    ap.add_argument("--sam3-weights", type=Path, default=Path("experiments/sam3/enc4/weights.pt"))
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = "cuda"

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    sam2 = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=dev))
    sam2.model.load_state_dict(torch.load(args.sam2_weights, map_location=dev), strict=False)
    sam2.model.eval()
    sam3 = Sam3TrackerModel.from_pretrained("facebook/sam3").to(dev)
    proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    sam3.load_state_dict(torch.load(args.sam3_weights, map_location=dev), strict=False)
    sam3.eval()
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
        image = Image.open(frames_root / file_name).convert("RGB")
        frame = np.array(image)
        boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (ds.instances[i][2] for i in indices)], dtype=np.float32)
        fboxes = boxes.copy()
        fboxes[:, [0, 2]] = frame.shape[1] - boxes[:, [2, 0]]
        flipped = frame[:, ::-1].copy()
        flipped_img = image.transpose(Image.FLIP_LEFT_RIGHT)
        rec: dict[str, float] = {}

        def sam2_pass(bf16: bool, flip: bool) -> None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                embed(sam2, frame, grad=False)
                box_logits(sam2, boxes)
                if flip:
                    embed(sam2, flipped, grad=False)
                    box_logits(sam2, fboxes)

        def sam3_pass(flip: bool) -> None:
            with torch.no_grad():
                s3.predict(sam3, proc, image, boxes, dev, grad=False)
                if flip:
                    s3.predict(sam3, proc, flipped_img, fboxes, dev, grad=False)

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

        for name, fn in (("sam2_bf16_single", lambda: sam2_pass(True, False)), ("sam2_bf16_flip", lambda: sam2_pass(True, True)),
                         ("sam2_fp32_single", lambda: sam2_pass(False, False)), ("sam3_fp32_single", lambda: sam3_pass(False)),
                         ("sam3_fp32_flip", lambda: sam3_pass(True)), ("classifier_4_members", classify)):
            t0 = sync()
            fn()
            rec[name] = sync() - t0
        if k >= args.warmup:
            for name, v in rec.items():
                rows[name].append(v)
            n_inst.append(len(indices))
        print(f"frame {k + 1}/{len(pick)} {file_name} {len(indices)} instruments " + " ".join(f"{n}={1000 * v:.0f}ms" for n, v in rec.items()), flush=True)
    res = {"gpu": torch.cuda.get_device_name(0), "frames": len(n_inst), "instruments_per_frame_mean": float(np.mean(n_inst)),
           "stages": {n: stats(v) for n, v in rows.items()}}
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
