"""Single-pass member logits of every official-test instrument when the mask comes from a segmentor (the SAM masks of
scripts/gtbox_sam_masks.py) and not from the ground truth. Same crop recipe as the ground-truth path: the given box times the
mask, padded to a square for the letterbox members. Writes y and det_<member> like scripts/evidential_extract.py.

Usage (titanxp, surgical environment):
    python scripts/extract_logits_from_masks.py --masks experiments/gtbox_sam/finetuned/masks.pkl \\
        --ensemble-config configs/evidential/ens_E_official_s42_lam0p01a10.yaml --out experiments/gtbox_sam/finetuned/logits.npz
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import crop_from_box
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--ensemble-config", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)

    masks = pickle.loads(args.masks.read_bytes())
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = []
    for m in cfg["members"]:
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        members.append((m["label"], net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))

    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    frames_root = args.data_root / "frames-001" / "frames"
    logits = {label: np.zeros((len(ds.instances), 7), dtype=np.float32) for label, *_ in members}
    done = np.zeros(len(ds.instances), dtype=bool)
    for n, (file_name, indices) in enumerate(sorted(by_frame.items())):
        frame = np.array(Image.open(frames_root / file_name).convert("RGB"))
        for idx in indices:
            if idx not in masks:
                continue
            mask = mask_codec.decode(masks[idx]).astype(bool)
            if not mask.any():
                continue
            box = ds.instances[idx][2]
            for label, net, transform, letterbox in members:
                crop = crop_from_box(frame, mask, box, letterbox)
                with torch.no_grad():
                    logits[label][idx] = net(transform(Image.fromarray(crop)).unsqueeze(0).to(device)).float().cpu().numpy()[0]
            done[idx] = True
        if (n + 1) % 200 == 0:
            print(f"{n + 1}/{len(by_frame)} frames", flush=True)
    y = np.array([inst[3] for inst in ds.instances])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, y=y, done=done, **{f"det_{label}": v for label, v in logits.items()})
    print(f"wrote {args.out}: {int(done.sum())} of {len(done)} instruments classified (empty masks skipped)")


if __name__ == "__main__":
    main()
