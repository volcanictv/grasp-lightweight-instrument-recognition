"""Crops and metadata for the error gallery: every official-test instance that at least one of the three
configurations (A: EdgeTAM, B: YOLO26s-seg, S: SAM2-large offline) gets wrong. Each crop is the mask-multiplied,
letterboxed box the classifier sees, resized to 160 px. Run where the frames live (titanxp).

Usage:
    python scripts/build_error_gallery_data.py --predictions docs/reports/causal_realtime/official_predictions.json \\
        --out-dir docs/reports/gallery_assets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset, _pad_to_square

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--size", type=int, default=160)
    args = ap.parse_args()

    p = json.loads(args.predictions.read_text())
    y = np.array(p["y"])
    final = {k: np.array(p[k]["pred"]) for k in ("A", "B", "S")}
    wrong = np.zeros(len(y), bool)
    for pred in final.values():
        wrong |= pred != y
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    crops = args.out_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    records = []
    for i in np.where(wrong)[0]:
        file_name, seg, box, label = ds.instances[int(i)]
        frame = np.array(Image.open(args.data_root / "frames-001" / "frames" / file_name).convert("RGB"))
        mask = decode_instance_mask(seg).astype(bool)
        x, yy, w, h = box
        crop = (frame[yy:yy + h, x:x + w] * mask[yy:yy + h, x:x + w, None]).astype(np.uint8)
        Image.fromarray(_pad_to_square(crop)).resize((args.size, args.size)).save(crops / f"{int(i)}.jpg", quality=85)
        rec = {"index": int(i), "file": file_name, "true": CLASSES[int(y[i])], "s1": p["s1_3member"][int(i)],
               "base3": CLASSES[p["base3"][int(i)]]}
        for k in ("A", "B", "S"):
            rec[k] = {"pred": CLASSES[int(final[k][i])], "gated": bool(p[k]["gated"][int(i)]), "correct": bool(final[k][i] == y[i])}
        records.append(rec)
    (args.out_dir / "data.json").write_text(json.dumps(records))
    print(f"{len(records)} instances wrong in at least one configuration; crops in {crops}")


if __name__ == "__main__":
    main()
