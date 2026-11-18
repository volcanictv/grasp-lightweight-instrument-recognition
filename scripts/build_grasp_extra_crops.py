"""Extra training crops for GraSP's region classifier from EndoVis 2018 (all 15 annotated sequences) and EndoVis 2017, as a pickle read by
data.extra_crops (see GraspRegionDataset).

Class mapping onto GraSP's order (BF, PF, LND, MCS, SI, CA, LG): EndoVis 2018 BF, PF, LND, MCS, SI, CA keep their class; its Ultrasound Probe is
dropped. EndoVis 2017 types 1, 2, 3, 6 are BF, PF, LND, MCS; vessel sealer, grasping retractor and "Other" are dropped. GraSP's Laparoscopic Grasper
has no counterpart, so it gets no extra examples. Each class is capped per source (random, fixed seed) so the outside data cannot swamp GraSP's own.
Crops are mask-multiplied and not padded (the dataset letterboxes them like its own).

Usage (titanxp): python scripts/build_grasp_extra_crops.py --endovis18 ~/endovis_isinet --zip2017 ~/Desktop/endovis_data/endovis2017.zip --cap 400 --out <pkl>
"""
from __future__ import annotations

import argparse
import io
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_codec
from scipy import ndimage

E18_TO_GRASP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5}  # BF PF LND MCS SI CA; class 4 (Ultrasound Probe) has no GraSP counterpart
E17_TO_GRASP = {1: 0, 2: 1, 3: 2, 6: 3}
MIN_AREA = 500


def crop(img: np.ndarray, mask: np.ndarray, box: tuple) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    return (img[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)


def from_2018(root: Path) -> list[tuple[np.ndarray, int]]:
    table = pickle.loads((root / "instances.pkl").read_bytes())
    out, cache = [], {}
    for inst in table["train"] + table["val"]:
        if inst["cls"] not in E18_TO_GRASP:
            continue
        key = (inst["seq"], inst["frame"])
        if key not in cache:
            cache.clear()
            cache[key] = np.array(Image.open(root / "frames" / f"{key[0]}_frame{key[1]:03d}.jpg").convert("RGB"))
        out.append((crop(cache[key], mask_codec.decode(inst["rle"]).astype(bool), inst["box"]), E18_TO_GRASP[inst["cls"]]))
    return out


def from_2017(zip_path: Path) -> list[tuple[np.ndarray, int]]:
    zf = zipfile.ZipFile(zip_path)
    names = set(zf.namelist())
    out = []
    for n in sorted(x for x in names if "/label/" in x and x.endswith(".bmp") and x.replace("/label/", "/image/") in names):
        lab = np.array(Image.open(io.BytesIO(zf.read(n))))
        lab = lab[..., 0] if lab.ndim == 3 else lab
        img = np.array(Image.open(io.BytesIO(zf.read(n.replace("/label/", "/image/")))).convert("RGB"))
        for v, cls in E17_TO_GRASP.items():
            comp, k = ndimage.label(lab == v)
            for j in range(1, k + 1):
                mask = comp == j
                if mask.sum() < MIN_AREA:
                    continue
                ys, xs = np.nonzero(mask)
                out.append((crop(img, mask, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)), cls))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endovis18", type=Path, required=True)
    ap.add_argument("--zip2017", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=400, help="maximum crops per class and source")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    crops, y = [], []
    for name, items in (("EndoVis 2018", from_2018(args.endovis18)), ("EndoVis 2017", from_2017(args.zip2017))):
        by_class: dict[int, list] = {}
        for c, lab in items:
            by_class.setdefault(lab, []).append(c)
        kept = {}
        for lab, cs in sorted(by_class.items()):
            idx = rng.permutation(len(cs))[: args.cap]
            kept[lab] = len(idx)
            crops += [cs[i] for i in idx]
            y += [lab] * len(idx)
        print(f"{name}: available {({k: len(v) for k, v in sorted(by_class.items())})}, kept {kept}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps({"crops": crops, "y": np.array(y), "cap": args.cap, "note": "BF PF LND MCS SI CA; no LG"}))
    print("total", len(y), "per GraSP class (BF PF LND MCS SI CA LG):", np.bincount(np.array(y), minlength=7).tolist())


if __name__ == "__main__":
    main()
