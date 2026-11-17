"""Per-instrument visibility and image-condition descriptors for the challenge-stratified TAPIS comparison (train and test cases, no model predictions involved).

One row per annotated instrument, in dataset order. Everything is computed from the frame, the instrument's ground-truth mask and box, and the other masks of the same
frame, so the strata cannot be chosen after seeing who gets the instrument wrong. Thresholds are derived from the train cases only by scripts/challenge_strata_eval.py.

    python scripts/challenge_strata_features.py --split train --out <csv>
    python scripts/challenge_strata_features.py --split test --out <csv>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy.ndimage import minimum_filter

from surgical_ai.data.mask_utils import decode_instance_mask, is_likely_occluded
from surgical_ai.data.region_dataset import GraspRegionDataset

SCALE = 4
BORDER_PX = 4
OUTER_BAND = 0.05
RING = 0.25
DARK_PATCH = 15


def dark_channel(rgb: np.ndarray) -> np.ndarray:
    return minimum_filter(rgb.min(axis=2), size=DARK_PATCH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()

    ds = GraspRegionDataset(args.data_root, args.split)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for i, inst in enumerate(ds.instances):
        by_frame[inst[0]].append(i)

    rows: dict[int, dict] = {}
    for n, (file_name, idx) in enumerate(sorted(by_frame.items())):
        frame = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        H, W = frame.shape[:2]
        small = np.array(Image.fromarray(frame).resize((W // SCALE, H // SCALE), Image.BOX))
        h, w = small.shape[:2]
        hsv = rgb_to_hsv(small / 255.0)
        hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
        dark = dark_channel(small).astype(np.float32) / 255
        gray = small.astype(np.float32).mean(axis=2) / 255
        masks = {i: np.array(Image.fromarray(decode_instance_mask(ds.instances[i][1]) * 255).resize((w, h), Image.NEAREST)) > 127 for i in idx}
        any_mask = np.zeros((h, w), bool)
        for m in masks.values():
            any_mask |= m
        band = np.zeros((h, w), bool)
        bw, bh = max(1, int(round(w * OUTER_BAND))), max(1, int(round(h * OUTER_BAND)))
        band[:bh], band[-bh:], band[:, :bw], band[:, -bw:] = True, True, True, True
        for i in idx:
            _f, seg, (x, y, bw_px, bh_px), label = ds.instances[i]
            m = masks[i]
            area = int(m.sum())
            others = np.zeros((h, w), bool)
            for j, mj in masks.items():
                if j != i:
                    others |= mj
            x0, y0, x1, y1 = x // SCALE, y // SCALE, (x + bw_px) // SCALE, (y + bh_px) // SCALE
            box = np.zeros((h, w), bool)
            box[y0:y1 + 1, x0:x1 + 1] = True
            ex, ey = int(RING * bw_px / SCALE), int(RING * bh_px / SCALE)
            ring_box = np.zeros((h, w), bool)
            ring_box[max(0, y0 - ey):y1 + ey + 1, max(0, x0 - ex):x1 + ex + 1] = True
            ring = ring_box & ~any_mask
            specular = m & (val >= 0.92) & (sat <= 0.20)
            blood = ring & ((hue <= 12) | (hue >= 345)) & (sat >= 0.65) & (val <= 0.65)
            native_m = decode_instance_mask(seg)
            touch = {"left": x <= BORDER_PX, "top": y <= BORDER_PX, "right": x + bw_px >= W - BORDER_PX, "bottom": y + bh_px >= H - BORDER_PX}
            rows[i] = {
                "instrument": i, "file": file_name, "case": file_name.split("/")[0], "label": label,
                "box_x": x, "box_y": y, "box_w": bw_px, "box_h": bh_px,
                "area_frac": area / (h * w),
                "n_in_frame": len(idx),
                "other_in_box": float((box & others).sum() / max(1, box.sum())),
                "fragmented": int(is_likely_occluded(native_m)),
                "touch_left": int(touch["left"]), "touch_top": int(touch["top"]), "touch_right": int(touch["right"]), "touch_bottom": int(touch["bottom"]),
                "n_borders": sum(touch.values()),
                "edge_frac": float((m & band).sum() / max(1, area)),
                "specular_frac": float(specular.sum() / max(1, area)),
                "blood_frac": float(blood.sum() / max(1, ring.sum())),
                "haze_dark": float(dark[ring].mean()) if ring.any() else float("nan"),
                "ring_sat": float(sat[ring].mean()) if ring.any() else float("nan"),
                "ring_contrast": float(gray[ring].std()) if ring.any() else float("nan"),
            }
        if n % 200 == 0:
            print(f"{n}/{len(by_frame)} frames", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for i in sorted(rows):
            wr.writerow(rows[i])
    print(f"wrote {len(rows)} instruments to {args.out}")


if __name__ == "__main__":
    main()
