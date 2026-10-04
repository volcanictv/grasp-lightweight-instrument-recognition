"""Tracker-style crops of the same physical instrument at neighbouring frames, for training-time augmentation.

For each annotated frame of one annotation split, EdgeTAM propagates every instrument's ground-truth mask
+/-K frames (1 Hz, so K seconds). Each neighbour mask becomes a mask-multiplied bounding-box crop that keeps the
instrument's label by construction (same physical instrument). A neighbour is dropped when the track is empty
or its mask area leaves [1/3, 3] of the annotated mask's area, a guard against a drifted track.
Only run this on training cases: the crops feed GraspRegionDataset's neighbour augmentation.

Usage (titanxp, from ~/edgetam_venv):
    python scripts/build_temporal_neighbors.py --json-split fold2 --out-dir experiments/temporal_neighbors/fold2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from PIL import Image

from evaluate_temporal_track_ensemble import build_track_frame_nums
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset, instance_key

AREA_LOW, AREA_HIGH = 1 / 3, 3.0
MAX_SIDE = 448


def save_crop(frame: np.ndarray, mask: np.ndarray, path: Path) -> None:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = Image.fromarray((frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8))
    if max(crop.size) > MAX_SIDE:
        scale = MAX_SIDE / max(crop.size)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    crop.save(path, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-split", required=True, choices=["train", "fold1", "fold2"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sam-config", default="configs/edgetam.yaml")
    ap.add_argument("--sam-checkpoint", type=Path, default=Path.home() / "EdgeTAM" / "checkpoints" / "edgetam.pt")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many annotated frames (smoke test)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    frames_root = args.data_root / "frames-001" / "frames"
    ds = GraspRegionDataset(args.data_root, args.json_split, letterbox=True)
    by_frame: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for file_name, seg, box, _label in ds.instances:
        by_frame[file_name].append((instance_key(file_name, box), seg))
    predictor = build_sam2_video_predictor(args.sam_config, str(args.sam_checkpoint), device=args.device)
    crops = args.out_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"/tmp/neighbors_{args.json_split}_{os.getpid()}")
    meta: dict[str, list[dict]] = {}
    kept = dropped = 0
    for n, (file_name, items) in enumerate(sorted(by_frame.items())):
        if args.max_frames is not None and n >= args.max_frames:
            break
        case, stem = file_name.split("/")
        nums, center = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.k)
        if len(nums) == 1:
            continue
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for local, num in enumerate(nums):
            shutil.copy(frames_root / case / f"{num:05d}.jpg", tmp / f"{local:05d}.jpg")
        gt = [decode_instance_mask(seg).astype(bool) for _key, seg in items]
        state = predictor.init_state(video_path=str(tmp))
        for obj, mask in enumerate(gt, start=1):
            predictor.add_new_mask(state, frame_idx=center, obj_id=obj, mask=mask)
        tracked: dict[tuple[int, int], np.ndarray] = {}
        for reverse in (False, True):
            for local, obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
                if local != center:
                    for slot, obj in enumerate(obj_ids):
                        tracked[(local, int(obj))] = (logits[slot, 0] > 0).cpu().numpy()
        for (local, obj), mask in tracked.items():
            area = float(mask.sum())
            ratio = area / max(1.0, float(gt[obj - 1].sum()))
            if area == 0 or not AREA_LOW <= ratio <= AREA_HIGH:
                dropped += 1
                continue
            key = items[obj - 1][0]
            name = f"{case}_{stem[:-4]}_{obj}_{local - center:+d}.jpg"
            save_crop(np.array(Image.open(tmp / f"{local:05d}.jpg").convert("RGB")), mask, crops / name)
            meta.setdefault(key, []).append({"file": name, "offset": local - center, "area_ratio": round(ratio, 3)})
            kept += 1
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(by_frame)} frames, kept {kept}, dropped {dropped}", flush=True)
            (args.out_dir / "meta.json").write_text(json.dumps(meta))
    (args.out_dir / "meta.json").write_text(json.dumps(meta))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"done: {len(meta)} instruments with neighbours, kept {kept}, dropped {dropped} by the area guard")


if __name__ == "__main__":
    main()
