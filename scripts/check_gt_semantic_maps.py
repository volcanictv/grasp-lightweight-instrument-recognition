"""Compares the per-frame ground-truth semantic maps painted from the JSON instance masks (class index + 1, annotation
order) with the dataset's own segmentation PNGs, to decide which one the benchmark IoUs should be computed against.

Usage (titanxp, surgical environment):
    python scripts/check_gt_semantic_maps.py --frames 80
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    by = defaultdict(list)
    for fn, seg, _box, label in ds.instances:
        by[fn].append((seg, label))
    seg_dir = args.data_root / "annotations" / "segmentations"
    checked = identical = missing = 0
    values: set[int] = set()
    worst = []
    for fn, items in list(by.items())[: args.frames]:
        case, stem = fn.split("/")
        png = seg_dir / case / stem.replace(".jpg", ".png")
        if not png.exists():
            missing += 1
            continue
        gt = np.array(Image.open(png))
        gt = gt if gt.ndim == 2 else gt[..., 0]
        values |= set(np.unique(gt).tolist())
        sem = np.zeros(gt.shape, dtype=np.int16)
        for seg, label in items:
            sem[decode_instance_mask(seg).astype(bool)] = label + 1
        checked += 1
        frac = float((sem != gt).mean())
        identical += frac == 0.0
        worst.append((frac, fn))
    print(f"checked {checked} frames ({missing} without a PNG); identical to JSON-painted: {identical}; PNG values: {sorted(values)}")
    print("largest disagreement (fraction of pixels):", sorted(worst, reverse=True)[:4])


if __name__ == "__main__":
    main()
