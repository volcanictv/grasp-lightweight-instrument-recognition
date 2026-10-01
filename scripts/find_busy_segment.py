"""Finds the stretch of a case with the most instruments per frame (by YOLO detections) and links it into a
directory the real-time harness can replay, so worst-case latency is measured on frames that really hold
several instruments.

Usage:
    python scripts/find_busy_segment.py --case CASE050 --yolo-weights <last.pt> --out-dir /tmp/rt_busy
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--step", type=int, default=3, help="score every step-th frame")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    frames = sorted((args.data_root / "frames-001" / "frames" / args.case).glob("*.jpg"))
    yolo = YOLO(str(args.yolo_weights))
    sampled = list(range(0, len(frames), args.step))
    counts = np.zeros(len(frames))
    for start in range(0, len(sampled), 64):
        chunk = sampled[start:start + 64]
        results = yolo.predict([str(frames[i]) for i in chunk], conf=0.25, imgsz=640, device=args.device, verbose=False)
        for i, r in zip(chunk, results):
            counts[i] = 0 if r.boxes is None else len(r.boxes)
    covered = np.convolve(counts[::args.step], np.ones(args.window // args.step) / (args.window // args.step), mode="valid")
    best = int(covered.argmax()) * args.step
    segment = frames[best:best + args.window]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, f in enumerate(segment):
        link = args.out_dir / f"{n:05d}.jpg"
        if not link.is_symlink():
            os.symlink(f.resolve(), link)
    print(f"{args.case} frames {best} to {best + len(segment) - 1}: mean {covered.max():.2f} detections per frame "
          f"(case mean {counts[::args.step].mean():.2f}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
