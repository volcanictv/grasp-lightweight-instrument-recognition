"""Write the ground-truth instance masks of a split as a masks.pkl (COCO RLE keyed by instrument index, same layout as the SAM
masks of scripts/gtbox_sam_masks.py) so the SAM-mask pipeline scripts can be run unchanged with perfect masks.

Usage: python scripts/gt_masks_to_pkl.py --out experiments/gtbox_sam/gtmask/masks.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from pycocotools import mask as mask_codec

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    masks = {i: mask_codec.encode(np.asfortranarray(decode_instance_mask(inst[1]).astype(np.uint8))) for i, inst in enumerate(ds.instances)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps(masks))
    print(f"wrote {len(masks)} ground-truth masks to {args.out}")


if __name__ == "__main__":
    main()
