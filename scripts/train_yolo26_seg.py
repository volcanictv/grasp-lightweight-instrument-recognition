"""Trains YOLO26-seg from a YAML config on the dataset written by
scripts/build_yolo_seg_dataset.py. Augmentation keys follow CLAUDE.md: no
vertical flip, no rotation, no hue shift.

Usage:
    python scripts/train_yolo26_seg.py --config configs/yolo26_seg_fold1.yaml \
        --data experiments/yolo_data/fold1/data.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(args.config.read_text())
    t = cfg["training"]
    run_id = f"{args.config.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = REPO_ROOT / "experiments"

    model = YOLO(cfg["model"]["weights"])
    start = time.time()
    model.train(
        data=str(args.data), imgsz=cfg["data"]["image_size"], epochs=t["epochs"], patience=t["patience"],
        batch=t["batch_size"], seed=t["seed"], val=t.get("val", True), deterministic=True, device=t["device"], workers=t["workers"],
        fliplr=t["fliplr"], flipud=t["flipud"], degrees=t["degrees"], hsv_h=t["hsv_h"],
        project=str(run_dir), name=run_id, exist_ok=False,
    )
    metrics = model.val(data=str(args.data), imgsz=cfg["data"]["image_size"], split="val")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip())
    manifest = {
        "config": cfg, "git_commit": commit, "git_dirty": dirty, "data_yaml": str(args.data),
        "duration_s": time.time() - start, "best_checkpoint": str(run_dir / run_id / "weights" / "best.pt"),
        "box_map50": float(metrics.box.map50), "mask_map50": float(metrics.seg.map50), "mask_map": float(metrics.seg.map),
    }
    (run_dir / run_id / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
