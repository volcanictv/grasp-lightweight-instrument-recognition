"""Classifies already-tracked masks (evaluate_temporal_track_ensemble.py --masks-out) with a different ensemble, writing the same
per-frame member logits the tracking script writes (det_<index>, shaped frames x members x classes). Lets the seed-43 and seed-44
members score the exact tracks the seed-42 run produced, without repeating the slow SAM2 propagation.

Usage (titanxp, surgical environment):
    python scripts/reclassify_tracked_masks.py --masks experiments/gtbox_sam/finetuned/tracked/masks_shard*.pkl \\
        --ensemble-config configs/evidential/ens_E_official_s43_lam0p01a10.yaml --out experiments/gtbox_sam/finetuned/tracked_s43/frames.npz
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import crop_from_box, crop_from_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, nargs="+", required=True)
    ap.add_argument("--ensemble-config", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = []
    for m in cfg["members"]:
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        members.append((net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))

    tracks: dict[int, dict] = {}
    case_of: dict[int, str] = {}
    for path in args.masks:
        blob = pickle.loads(path.read_bytes())
        tracks.update(blob["masks"])
        case_of.update(blob["case_of"])
    frames_root = args.data_root / "frames-001" / "frames"
    out = {}
    for n, (idx, frames) in enumerate(sorted(tracks.items())):
        box = ds.instances[idx][2]
        offsets = sorted(frames)
        crops = {}
        for off in offsets:
            frame_num, rle = frames[off]
            mask = mask_codec.decode(rle).astype(bool)
            frame = np.array(Image.open(frames_root / case_of[idx] / f"{frame_num:05d}.jpg").convert("RGB"))
            for letterbox in (True, False):
                crop = crop_from_box(frame, mask, box, letterbox) if off == 0 else crop_from_mask(frame, mask, letterbox)
                crops[(off, letterbox)] = crop
        valid = [off for off in offsets if crops[(off, True)] is not None]
        per_member = []
        for net, transform, letterbox in members:
            batch = torch.stack([transform(Image.fromarray(crops[(off, letterbox)])) for off in valid]).to(device)
            with torch.no_grad():
                per_member.append(net(batch).float().cpu().numpy())
        out[f"det_{idx}"] = np.stack(per_member, axis=1)
        out[f"center_{idx}"] = np.array(valid.index(0) if 0 in valid else 0)
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(tracks)} tracks", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {len(tracks)} tracks")


if __name__ == "__main__":
    main()
