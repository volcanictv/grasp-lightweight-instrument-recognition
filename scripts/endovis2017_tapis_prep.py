"""EndoVis 2017, the standard 4-fold cross-validation, in the layout of the TAPIS / MATIS pipeline, and a per-instance table for scoring with our scorer.

Source: our endovis2017.zip (per-frame type masks, .bmp). Only the TRAINING RELEASE is used (train/: sequences 1 to 8, frames 0 to 224); the official test frames (val1..val10) are never read, so the held-out
sequences of a fold cannot leak into its training. Folds as in TernausNet / ISINet: validation sequences {1,3}, {2,5}, {4,8}, {6,7}. Instances are the connected components (4-connectivity, at least 500 pixels)
of each instrument type (1 BF, 2 PF, 3 LND, 4 VS, 5 GR, 6 MCS, 7 Other/UP), the same rule as scripts/endovis_gtbox_classifiers_grasp_e17.py.

Writes <out>/images/seq{N}_frame{NNN}.png, <out>/annotations/Fold{k}/{train,val}.json (COCO, polygons + the RLE in an extra field) and <out>/instances2017.pkl
(list of dicts: seq, frame, cls 0..6, rle, box x0 y0 x1 y1, area, fold = the fold in which its sequence is held out).

Usage (titanxp, tapis_venv for OpenCV):
    ~/tapis_venv/bin/python scripts/endovis2017_tapis_prep.py --zip ~/Desktop/endovis_data/endovis2017.zip --out ~/endovis2017_tapis
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_codec
from scipy import ndimage

NAMES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Vessel Sealer", "Grasping Retractor", "Monopolar Curved Scissors", "Other (Ultrasound Probe)"]
FOLDS = {0: (1, 3), 1: (2, 5), 2: (4, 8), 3: (6, 7)}
MIN_AREA = 500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many frames")
    args = ap.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.zip)
    names = set(zf.namelist())
    labels = sorted(n for n in names if re.search(r"/train/label/seq_\d+_frame\d+\.bmp$", n) and n.replace("/label/", "/image/") in names)
    if args.limit:
        labels = labels[: args.limit]
    fold_of = {seq: k for k, seqs in FOLDS.items() for seq in seqs}
    images: dict[tuple[int, int], dict] = {}
    anns_by_frame: dict[tuple[int, int], list] = {}
    table: list[dict] = []
    for n in labels:
        m = re.search(r"seq_(\d+)_frame(\d+)\.bmp$", n)
        seq, frame = int(m.group(1)), int(m.group(2))
        assert 1 <= seq <= 8 and frame < 225, n
        lab = np.array(Image.open(io.BytesIO(zf.read(n))))
        lab = lab[..., 0] if lab.ndim == 3 else lab
        img = np.array(Image.open(io.BytesIO(zf.read(n.replace("/label/", "/image/")))).convert("RGB"))
        name = f"seq{seq}_frame{frame:03d}.png"
        if not (args.out / "images" / name).exists():
            cv2.imwrite(str(args.out / "images" / name), img[..., ::-1])
        h, w = lab.shape
        images[(seq, frame)] = {"file_name": name, "width": w, "height": h, "video_name": f"seq{seq}", "frame_num": frame}
        anns: list[dict] = []
        for t in range(1, 8):
            comp, k = ndimage.label(lab == t)
            for j in range(1, k + 1):
                mask = comp == j
                area = int(mask.sum())
                if area < MIN_AREA:
                    continue
                ys, xs = np.nonzero(mask)
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                rle = mask_codec.encode(np.asfortranarray(mask.astype(np.uint8)))
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = [c.ravel().tolist() for c in contours if len(c) >= 3]
                anns.append({"category_id": t, "iscrowd": 0, "segmentation": polys, "rle": {"size": [h, w], "counts": rle["counts"].decode()},
                             "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]], "area": area})
                table.append({"seq": seq, "frame": frame, "cls": t - 1, "rle": rle, "box": box, "area": area, "fold": fold_of[seq]})
        anns_by_frame[(seq, frame)] = anns
    cats = [{"id": i + 1, "name": nme, "supercategory": "instrument"} for i, nme in enumerate(NAMES)]
    for k, val_seqs in FOLDS.items():
        d = args.out / "annotations" / f"Fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            keys = sorted(key for key in images if (key[0] in val_seqs) == (split == "val"))
            imgs, anns = [], []
            for key in keys:
                imgs.append({"id": len(imgs) + 1, **images[key]})
                for a in anns_by_frame[key]:
                    anns.append({"id": len(anns) + 1, "image_id": len(imgs), **a})
            (d / f"{split}.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
            per_class = np.bincount([a["category_id"] - 1 for a in anns], minlength=7).tolist()
            print(f"fold {k} {split}: sequences {sorted({key[0] for key in keys})}, {len(imgs)} images, {len(anns)} instances, per class {per_class}", flush=True)
    (args.out / "instances2017.pkl").write_bytes(pickle.dumps(table))
    print(f"wrote {len(images)} frames and {len(table)} instances to {args.out}")


if __name__ == "__main__":
    main()
