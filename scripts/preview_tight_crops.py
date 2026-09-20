"""Audit and preview of the tight-crop training data (docs/reports/tight_crop/).

1. Geometry audit over every instance of a split (mask-only, no image reads):
   the instrument-outnumbers-band invariant, the band radius reached, how many
   instances hit the radius cap or get no band at all, and other-instrument
   pixel counts inside the kept region.
2. A preview grid: standard crop | tight crop | tight crop after the masklight
   augmentation, for a seeded sample across classes, to check by eye.

Usage:
    python scripts/preview_tight_crops.py --split fold1 --out-dir OUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset, _pad_to_square
from surgical_ai.data.tight_crop import TightCropDataset, tight_band_crop
from surgical_ai.data.tight_transforms import build_tight_transforms
from surgical_ai.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="fold2", help="annotation split to audit (fold2 is fold1's training half)")
    parser.add_argument("--band-frac", type=float, default=0.5)
    parser.add_argument("--r-cap", type=int, default=24)
    parser.add_argument("--n-preview", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))

    tight = TightCropDataset(data_root, args.split, transform=None, band_frac=args.band_frac, r_cap=args.r_cap)
    standard = GraspRegionDataset(data_root, args.split, transform=None, letterbox=True)
    names = tight.class_names_ordered()

    ratios, radii, empty, capped, no_band = [], [], 0, 0, 0
    for idx, (file_name, seg, _box, _label) in enumerate(tight.instances):
        mask = decode_instance_mask(seg).astype(bool)
        others = np.zeros_like(mask)
        for j in tight._by_file[file_name]:
            if j != idx:
                others |= decode_instance_mask(tight.instances[j][1]).astype(bool)
        dummy = np.ones(mask.shape + (3,), dtype=np.uint8)
        _crop, info = tight_band_crop(dummy, mask, others, args.band_frac, args.r_cap)
        if info["empty"]:
            empty += 1
            continue
        assert info["band_px"] < info["instrument_px"], "instrument must outnumber the band"
        ratios.append(info["band_px"] / info["instrument_px"])
        radii.append(info["radius"])
        capped += info["radius"] >= args.r_cap - 1
        no_band += info["band_px"] == 0
    ratios, radii = np.array(ratios), np.array(radii)
    audit = {
        "split": args.split, "n_instances": len(tight), "empty_masks": empty,
        "invariant_band_lt_instrument_holds_for_all": True,
        "band_to_instrument_ratio": {"mean": float(ratios.mean()), "median": float(np.median(ratios)),
                                     "p10": float(np.percentile(ratios, 10)), "p90": float(np.percentile(ratios, 90))},
        "band_radius_px": {"mean": float(radii.mean()), "median": float(np.median(radii)), "p90": float(np.percentile(radii, 90))},
        "hit_radius_cap": int(capped), "no_band_at_all": int(no_band),
        "band_frac": args.band_frac, "r_cap": args.r_cap,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1))

    rng = np.random.default_rng(0)
    labels = np.array([lbl for _f, _s, _b, lbl in tight.instances])
    picks = [int(rng.choice(np.where(labels == c)[0])) for c in range(len(names))]
    picks += [int(i) for i in rng.choice(len(tight), size=max(0, args.n_preview - len(picks)), replace=False)]
    aug = build_tight_transforms(224, train=True, augmentation="masklight")
    mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]

    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.5 * len(picks)))
    for row, idx in enumerate(picks):
        std_crop = standard[idx][0]
        tight_crop, info = tight.crop_and_info(idx)
        aug_img = aug(Image.fromarray(_pad_to_square(tight_crop))).numpy() * std + mean
        for col, (img, title) in enumerate([
            (np.asarray(std_crop), f"standard ({names[labels[idx]]})"),
            (_pad_to_square(tight_crop), f"tight band, r={info['radius']:.0f}px, band/inst={info['band_px'] / max(info['instrument_px'], 1):.2f}"),
            (np.clip(aug_img.transpose(1, 2, 0), 0, 1), "tight + masklight aug"),
        ]):
            axes[row, col].imshow(img)
            axes[row, col].set_title(title, fontsize=6)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(args.out_dir / "preview.png", dpi=110)
    print(f"wrote {args.out_dir / 'audit.json'} and {args.out_dir / 'preview.png'}")


if __name__ == "__main__":
    main()
