"""Saves actual misclassified Task B instance crops for a given set of
confusion pairs, for direct visual inspection. Built to answer one
question docs/analyze_classification_errors.py's confusion matrix can't:
is a given confusion pair genuine visual ambiguity (poor lighting, tissue
occlusion, a tip cropped out of frame) or a plain model gap. See
docs/error_analysis.md for what this found on the official test set and
docs/DECISIONS.md, 2026-09-02.

Read-only except for writing PNGs to --out-dir; does not touch
experiments/ or docs/.

Usage:
    python scripts/extract_confusion_crops.py \\
        experiments/region_baseline_20260831-182451/best.pt \\
        --out-dir confusion_crops \\
        --pairs "Bipolar Forceps:Prograsp Forceps" "Large Needle Driver:Bipolar Forceps"
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
import torch
from PIL import Image

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.models import build_model

DEFAULT_PAIRS = [
    "Bipolar Forceps:Prograsp Forceps",
    "Prograsp Forceps:Bipolar Forceps",
    "Large Needle Driver:Bipolar Forceps",
    "Large Needle Driver:Monopolar Curved Scissors",
    "Monopolar Curved Scissors:Suction Instrument",
    "Laparoscopic Grasper:Suction Instrument",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-per-pair", type=int, default=4)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS, help='"TrueClass:PredictedClass" strings')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    top_pairs = {tuple(p.split(":")) for p in args.pairs}

    device = torch.device(args.device)
    ds = GraspRegionDataset(args.data_root, args.split, transform=build_transforms(args.image_size, train=False))
    class_names = ds.class_names_ordered()

    model = build_model("mobilenet_v3_small", num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    missing, _unexpected = model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    assert not missing, missing
    model.eval()

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for images, _labels in loader:
            all_preds.append(model(images.to(device)).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(all_preds)

    saved = defaultdict(int)
    for idx, (file_name, segmentation, (x, y, w, h), label_idx) in enumerate(ds.instances):
        true_name, pred_name = class_names[label_idx], class_names[y_pred[idx]]
        if true_name == pred_name:
            continue
        key = (true_name, pred_name)
        if key not in top_pairs or saved[key] >= args.n_per_pair:
            continue

        frame = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        height, width = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        mask = decode_instance_mask(segmentation)
        masked_crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)
        raw_crop = frame[y0:y1, x0:x1]

        pair_dir = args.out_dir / f"{true_name.replace(' ', '_')}_as_{pred_name.replace(' ', '_')}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        stem = file_name.replace("/", "_").replace(".jpg", "")
        Image.fromarray(masked_crop).save(pair_dir / f"{stem}_masked.png")
        Image.fromarray(raw_crop).save(pair_dir / f"{stem}_raw.png")
        saved[key] += 1
        print(f"saved {key} #{saved[key]}: {file_name} bbox={(x, y, w, h)}")

    print("\ncounts per pair:", dict(saved))
    print(f"output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
