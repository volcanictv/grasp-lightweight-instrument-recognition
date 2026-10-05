"""YOLO-style counterpart of scripts/build_temporal_neighbors.py: neighbour crops of each annotated instrument at
+/-K frames, with the masks produced by the deployed YOLO26s-seg matching tracker (conf 0.1, IoU 0.1, coast 3, centre-frame
fallback, the settings of tracker B) instead of EdgeTAM, same naming, same area guard [1/3, 3], same meta.json layout.

The YOLO weights must come from a model that never trained on the cases being processed (cross-fitting), so the crops
carry the mask quality YOLO has on unseen cases: for the fold2 cases use the YOLO trained on the fold1 cases, and the reverse.

Usage (titanxp, from ~/yolo26_venv):
    python scripts/build_temporal_neighbors_yolo.py --json-split fold1 --yolo-weights <trained on fold2 cases> \\
        --out-dir experiments/temporal_neighbors_yolo/fold1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from PIL import Image
from pycocotools import mask as mask_codec

from build_temporal_neighbors import AREA_HIGH, AREA_LOW, save_crop
from evaluate_temporal_track_ensemble import build_track_frame_nums
from evaluate_temporal_track_yolo import detect, encode, walk_track
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset, instance_key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-split", required=True, choices=["train", "fold1", "fold2"])
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    detect_args = argparse.Namespace(yolo_conf=0.1, yolo_imgsz=640, device=args.device)
    frames_root = args.data_root / "frames-001" / "frames"
    ds = GraspRegionDataset(args.data_root, args.json_split, letterbox=True)
    by_frame: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for file_name, seg, box, _label in ds.instances:
        by_frame[file_name].append((instance_key(file_name, box), seg))
    yolo = YOLO(str(args.yolo_weights))
    crops = args.out_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    meta: dict[str, list[dict]] = {}
    kept = dropped = 0
    for n, (file_name, items) in enumerate(sorted(by_frame.items())):
        if args.max_frames is not None and n >= args.max_frames:
            break
        case, stem = file_name.split("/")
        nums, center = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.k)
        if len(nums) == 1:
            continue
        paths = [str(frames_root / case / f"{num:05d}.jpg") for num in nums]
        per_frame = detect(yolo, paths, detect_args)
        frames = {}
        for obj, (key, seg) in enumerate(items, start=1):
            gt = decode_instance_mask(seg).astype(bool)
            gt_rle = encode(gt)
            found = {}
            for pos, mask in walk_track(gt_rle, gt_rle, per_frame[center + 1:], 0.1, 3, True).items():
                found[center + 1 + pos] = mask
            for pos, mask in walk_track(gt_rle, gt_rle, per_frame[:center][::-1], 0.1, 3, True).items():
                found[center - 1 - pos] = mask
            for local, rle in found.items():
                mask = mask_codec.decode(rle).astype(bool)
                ratio = float(mask.sum()) / max(1.0, float(gt.sum()))
                if not mask.any() or not AREA_LOW <= ratio <= AREA_HIGH:
                    dropped += 1
                    continue
                if local not in frames:
                    frames[local] = np.array(Image.open(paths[local]).convert("RGB"))
                name = f"{case}_{stem[:-4]}_{obj}_{local - center:+d}.jpg"
                save_crop(frames[local], mask, crops / name)
                meta.setdefault(key, []).append({"file": name, "offset": local - center, "area_ratio": round(ratio, 3)})
                kept += 1
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(by_frame)} frames, kept {kept}, dropped {dropped}", flush=True)
            (args.out_dir / "meta.json").write_text(json.dumps(meta))
    (args.out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"done: {len(meta)} instruments with neighbours, kept {kept}, dropped {dropped} by the area guard")


if __name__ == "__main__":
    main()
