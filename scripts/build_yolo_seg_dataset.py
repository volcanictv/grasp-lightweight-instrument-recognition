"""Writes an Ultralytics YOLO-seg dataset for one GraSP data.split value.

Images are symlinks into the original frames (no duplicated 60 GB tree);
labels come from each annotation's COCO RLE mask. Only annotated frames are
written. Split resolution goes through the official JSONs, so `fold1` trains on
fold2's cases and validates on fold1's, and official test is only touched for
`official`.

A fragmented mask (tissue occluding the middle of an instrument) keeps only its
largest component: YOLO-seg labels are one polygon per instance line.

Usage:
    python scripts/build_yolo_seg_dataset.py --split fold1 --out experiments/yolo_data/fold1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np

from surgical_ai.data import splits, statistics
from surgical_ai.data.mask_utils import decode_instance_mask


def mask_to_polygon(mask: np.ndarray, epsilon_px: float = 1.5) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(largest, epsilon_px, True).reshape(-1, 2)
    return poly if len(poly) >= 3 else None


def write_split(doc: dict, frames_root: Path, out: Path, subset: str, id_to_index: dict[int, int]) -> int:
    img_dir, lbl_dir = out / "images" / subset, out / "labels" / subset
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    images = {i["id"]: i for i in doc["images"]}
    lines: dict[int, list[str]] = {}
    for a in doc["annotations"]:
        seg = a["segmentation"]
        mask = decode_instance_mask(seg)
        poly = mask_to_polygon(mask)
        if poly is None:
            continue
        h, w = seg["size"]
        coords = (poly / np.array([w, h])).clip(0, 1).reshape(-1)
        lines.setdefault(a["image_id"], []).append(
            f"{id_to_index[a['category_id']]} " + " ".join(f"{c:.5f}" for c in coords)
        )
    for image_id, rows in lines.items():
        case, name = images[image_id]["file_name"].split("/")
        stem = f"{case}_{Path(name).stem}"
        link = img_dir / f"{stem}.jpg"
        if not link.is_symlink():
            os.symlink((frames_root / case / name).resolve(), link)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["official", "fold1", "fold2"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = parser.parse_args()

    train_split, val_split = splits.resolve_train_val_split(args.split)
    frames_root = args.data_root / "frames-001" / "frames"
    train_doc = splits.load_short_term(args.data_root, train_split)
    val_doc = splits.load_short_term(args.data_root, val_split)
    id_to_index = statistics.category_id_to_index(train_doc)
    names = statistics.category_names(train_doc)
    ordered = [names[cid] for cid in sorted(names)]

    n_train = write_split(train_doc, frames_root, args.out, "train", id_to_index)
    n_val = write_split(val_doc, frames_root, args.out, "val", id_to_index)
    yaml_lines = [f"path: {args.out.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(ordered)]
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"{args.split}: {n_train} train frames ({train_split}), {n_val} val frames ({val_split}) -> {args.out}")


if __name__ == "__main__":
    main()
