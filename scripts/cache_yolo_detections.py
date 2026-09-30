"""Runs YOLO26-seg once over every frame in the tracking windows of the given instances and stores the
masks as COCO RLE, so matching-option sweeps of evaluate_temporal_track_yolo.py replay detections
instead of re-running the model and re-decoding full-resolution masks.

Usage:
    python scripts/cache_yolo_detections.py --yolo-weights <best.pt> --split fold1 \\
        --error-cases-json experiments/three_member_yolo/fold1_track_shard0.json --out experiments/yolo_sweep/fold1_dets.pkl
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch

from evaluate_temporal_track_ensemble import build_track_frame_nums
from evaluate_temporal_track_yolo import detect
from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--split", default="fold1")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--yolo-conf", type=float, default=0.1)
    ap.add_argument("--yolo-imgsz", type=int, default=640)
    ap.add_argument("--error-cases-json", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    frames_root = args.data_root / "frames-001" / "frames"
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    indices = [e["index"] for e in json.loads(args.error_cases_json.read_text())["errors"]]
    paths: set[str] = set()
    for idx in indices:
        case, stem = ds.instances[idx][0].split("/")
        nums, _ = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.window)
        paths.update(str(frames_root / case / f"{n:05d}.jpg") for n in nums)
    ordered = sorted(paths)
    print(f"{len(ordered)} unique frames for {len(indices)} instances")

    yolo = YOLO(str(args.yolo_weights))
    cache: dict[str, list[dict]] = {}
    for start in range(0, len(ordered), 32):
        chunk = ordered[start:start + 32]
        cache.update(zip(chunk, detect(yolo, chunk, args)))
        if (start // 32) % 20 == 0:
            print(f"{start + len(chunk)}/{len(ordered)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps(cache))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
