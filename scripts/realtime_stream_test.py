"""Replays a frame source through the real-time pipeline at 1 Hz and reports whether it keeps up.

Sources: a video file or a directory of frames (sampled from --native-fps in the config down to the config's
sample_hz), or --synthetic N noise frames for a wiring check. With --realtime frames arrive on a wall-clock
schedule, so a slow stage shows up as latency and deadline misses; without it frames are fed as fast as the
pipeline asks, which measures compute only.

Usage (titanxp, repo root; EdgeTAM runs from ~/edgetam_venv, everything else from ~/yolo26_venv):
    python scripts/realtime_stream_test.py --config configs/realtime_default.yaml --video clip.mp4 \\
        --yolo-weights <best.pt> --sam-checkpoint ~/EdgeTAM/checkpoints/edgetam.pt --realtime --out report.json
    python scripts/realtime_stream_test.py --config configs/realtime_default.yaml --synthetic 60 --stub --out smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import yaml

from surgical_ai.realtime import adapters, stream
from surgical_ai.realtime.pipeline import RealtimePipeline
from surgical_ai.realtime.runner import run_stream


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "realtime_default.yaml")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--frames-dir", type=Path)
    src.add_argument("--synthetic", type=int, metavar="N_FRAMES")
    ap.add_argument("--yolo-weights", type=Path)
    ap.add_argument("--sam-checkpoint", type=Path)
    ap.add_argument("--stub", action="store_true", help="stub detector and classifier: no models needed")
    ap.add_argument("--realtime", action="store_true", help="pace arrivals on the wall clock")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    if args.stub:
        detector, classifier = adapters.StubDetector(), adapters.StubClassifier(members=len(cfg["classifier"]["members"]))
    else:
        detector = adapters.YoloDetector(args.yolo_weights, args.device, cfg["detector"]["conf"], cfg["detector"]["imgsz"])
        classifier = adapters.EnsembleClassifier(REPO_ROOT / cfg["classifier"]["ensemble_config"], cfg["classifier"]["members"], REPO_ROOT, args.device)

    t = cfg["tracker"]
    if t["name"] == "none":
        tracker = None
    elif t["name"] == "match":
        tracker = adapters.MatchTracker(**t["match"])
    elif args.stub:
        tracker = adapters.MatchTracker(**t["match"])
    else:
        tracker = adapters.SamStyleTracker(t["sam_config"], args.sam_checkpoint, args.device)

    pipeline = RealtimePipeline(detector, classifier, np.array(cfg["classifier"]["weights"]), cfg["gate"]["threshold"],
                                tracker, t["look_back"])
    if args.video:
        frames = stream.video_source(args.video)
    elif args.frames_dir:
        frames = stream.directory_source(args.frames_dir)
    else:
        frames = stream.synthetic_source(args.synthetic)
    sampled = stream.sample(frames, cfg["stream"]["native_fps"], cfg["stream"]["sample_hz"])
    hz = cfg["stream"]["sample_hz"]
    arrivals = stream.paced(sampled, hz) if args.realtime else stream.unpaced(sampled)

    start = time.time()
    report = run_stream(pipeline, arrivals, hz, args.max_frames).to_dict()
    report["extras"] = {"wall_seconds": time.time() - start, "paced": args.realtime, "tracker": t["name"], "stub": args.stub}
    try:
        import torch

        if torch.cuda.is_available():
            report["extras"]["gpu"] = torch.cuda.get_device_name(0)
            report["extras"]["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    except ImportError:
        pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("frames", "instances", "gated", "deadline_misses", "frame_latency_ms", "refine_latency_ms")}, indent=1))


if __name__ == "__main__":
    main()
