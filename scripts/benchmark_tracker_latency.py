"""Per-stage latency of the tracking pipeline on one GPU, measured with nothing else running
(CLAUDE.md latency rules: warm up, synchronise, report median and p95).

Stages (--stage):
    yolo        YOLO26-seg inference, one frame, imgsz 640 (the per-frame cost of a 1 Hz stream)
    classify    each ensemble member and the whole ensemble on one crop (batch 1), synthetic input of
                the member's size: latency does not depend on pixel content
    match       RLE track matching for one instance from a detections cache (CPU)
    propagate   SAM2 / EdgeTAM mask propagation for one instance over its +/-window, frame copy excluded;
                run with the venv that holds the tracker (--sam2-config / --sam2-checkpoint)

Usage:
    python scripts/benchmark_tracker_latency.py --stage yolo --yolo-weights <best.pt> --out lat_yolo.json
    python scripts/benchmark_tracker_latency.py --stage classify --ensemble-config configs/evidential/ens_E_fold1_s42_lam0p01a10.yaml --out lat_cls.json
    python scripts/benchmark_tracker_latency.py --stage propagate --sam2-config configs/edgetam.yaml --sam2-checkpoint ~/EdgeTAM/checkpoints/edgetam.pt \\
        --error-cases-json experiments/edgetam_fold1/track_shard0.json --split fold1 --out lat_edgetam.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml

from evaluate_temporal_track_ensemble import build_track_frame_nums


def summarize(times_s: list[float]) -> dict:
    ms = np.array(times_s) * 1000
    return {"n": len(ms), "median_ms": float(np.median(ms)), "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(ms.mean())}


def timed(fn, warmup: int, runs: int, device: torch.device | None) -> dict:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        out.append(time.perf_counter() - t0)
    return summarize(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["yolo", "classify", "match", "propagate"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--yolo-weights", type=Path)
    ap.add_argument("--ensemble-config", type=Path)
    ap.add_argument("--sam2-config")
    ap.add_argument("--sam2-checkpoint", type=Path)
    ap.add_argument("--split", default="fold1")
    ap.add_argument("--error-cases-json", type=Path)
    ap.add_argument("--detections-cache", type=Path)
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--window", type=int, default=10, help="frames each side (non-causal) or past frames (--causal; the causal pipeline uses 20)")
    ap.add_argument("--causal", action="store_true", help="propagate backward from the keyframe only, over the past window: what a live system can do")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    result: dict = {"stage": args.stage, "device": args.device, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}

    if args.stage == "yolo":
        from ultralytics import YOLO
        yolo = YOLO(str(args.yolo_weights))
        frame = (np.random.default_rng(0).random((800, 1280, 3)) * 255).astype(np.uint8)
        result["yolo_weights"] = str(args.yolo_weights)
        result["params_m"] = sum(p.numel() for p in yolo.model.parameters()) / 1e6
        result["per_frame"] = timed(lambda: yolo.predict(frame, imgsz=640, retina_masks=True, device=args.device, verbose=False), 20, args.runs, device)

    elif args.stage == "classify":
        from surgical_ai.models import build_model
        cfg = yaml.safe_load(args.ensemble_config.read_text())
        models, inputs = [], []
        for m in cfg["members"]:
            net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device).eval()
            models.append(net)
            inputs.append(torch.randn(1, 3, m["image_size"], m["image_size"], device=device))
        result["members"] = {}
        with torch.no_grad():
            for m, net, x in zip(cfg["members"], models, inputs):
                result["members"][m["label"]] = {"image_size": m["image_size"], **timed(lambda: net(x), 20, args.runs, device)}
            result["all_members_per_crop"] = timed(lambda: [net(x) for net, x in zip(models, inputs)], 20, args.runs, device)

    elif args.stage == "match":
        from evaluate_temporal_track_yolo import walk_track
        from surgical_ai.data.mask_utils import decode_instance_mask
        from surgical_ai.data.region_dataset import GraspRegionDataset
        from evaluate_temporal_track_yolo import encode
        cache = pickle.loads(args.detections_cache.read_bytes())
        ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
        frames_root = args.data_root / "frames-001" / "frames"
        idx = [e["index"] for e in json.loads(args.error_cases_json.read_text())["errors"]][: args.n_instances]
        jobs = []
        for i in idx:
            case, stem = ds.instances[i][0].split("/")
            nums, c = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.window)
            gt = encode(decode_instance_mask(ds.instances[i][1]).astype(bool))
            per = [cache[str(frames_root / case / f"{n:05d}.jpg")] for n in nums]
            jobs.append((gt, per[c + 1:], per[:c][::-1]))
        def run_all():
            for gt, fwd, bwd in jobs:
                walk_track(gt, gt, fwd, 0.1, 2, False)
                walk_track(gt, gt, bwd, 0.1, 2, False)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            run_all()
            times.append((time.perf_counter() - t0) / len(jobs))
        result["per_instance"] = summarize(times)

    else:
        from sam2.build_sam import build_sam2_video_predictor
        from surgical_ai.data.mask_utils import decode_instance_mask
        from surgical_ai.data.region_dataset import GraspRegionDataset
        predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))
        result["params_m"] = sum(p.numel() for p in predictor.parameters()) / 1e6
        result["causal"], result["window"] = args.causal, args.window
        ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
        frames_root = args.data_root / "frames-001" / "frames"
        idx = [e["index"] for e in json.loads(args.error_cases_json.read_text())["errors"]][: args.n_instances]
        tmp = Path(os.environ.get("BENCH_TMP", "/tmp/bench_prop_frames"))  # on a shared machine set BENCH_TMP to a private directory: this directory is deleted and recreated
        per_instance, per_frame = [], []
        for n, i in enumerate(idx):
            case, stem = ds.instances[i][0].split("/")
            nums, c = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.window, 0 if args.causal else None)
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            for k, fn in enumerate(nums):
                shutil.copy(frames_root / case / f"{fn:05d}.jpg", tmp / f"{k:05d}.jpg")
            gt = decode_instance_mask(ds.instances[i][1]).astype(bool)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            state = predictor.init_state(video_path=str(tmp))
            predictor.add_new_mask(state, frame_idx=c, obj_id=1, mask=gt)
            if not args.causal:
                for _ in predictor.propagate_in_video(state):
                    pass
            for _ in predictor.propagate_in_video(state, reverse=True):
                pass
            torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            if n >= 2:  # first two instances warm up kernels and allocators
                per_instance.append(dt)
                per_frame.append(dt / max(1, len(nums) - 1))
        result["per_instance"] = summarize(per_instance)
        result["per_propagated_frame"] = summarize(per_frame)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
