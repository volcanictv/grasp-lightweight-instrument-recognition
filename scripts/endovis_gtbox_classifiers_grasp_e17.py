"""The GraSP-recipe four-member evidential ensemble of endovis_gtbox_classifiers_grasp.py, trained on EndoVis 2018 plus the complete EndoVis 2017.

Only the training data changes. 2017 instances come from the release's per-frame type masks (connected components of at least 500 pixels, masked and
letterboxed exactly like the 2018 crops): types 1 BF, 2 PF, 3 LND, 6 MCS keep their class, type 7 ("Other") becomes Ultrasound Probe, and types 4 (Vessel
Sealer) and 5 (Grasping Retractor) are dropped because 2018 has no counterpart. Frames whose image is missing from the release (sequence 9) are skipped.
2017 crops have no tracker-style neighbours, so those samples are always used as they are. ISINet also trained its 2018 model on the full 2017 set.

Usage (titanxp, surgical environment, ENDOVIS_GT/ENDOVIS_ISINET exported):
    python scripts/endovis_gtbox_classifiers_grasp_e17.py --seed 42 --neighbours <dir> --zip2017 <endovis2017.zip> --out-dir <dir> [--only resnet50_224 ...]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

import endovis_evidential_stage1 as es1
import endovis_gtbox_classifiers_grasp as G
import endovis_gtbox_common as C

T2017 = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
MIN_AREA = 500


def crops_2017(zip_path: Path) -> tuple[dict, np.ndarray]:
    zf = zipfile.ZipFile(zip_path)
    names = set(zf.namelist())
    labels = sorted(n for n in names if "/label/" in n and n.endswith(".bmp") and n.replace("/label/", "/image/") in names)
    crops: dict[bool, list] = {True: [], False: []}
    y: list[int] = []
    for n in labels:
        lab = np.array(Image.open(io.BytesIO(zf.read(n))))
        lab = lab[..., 0] if lab.ndim == 3 else lab
        img = np.array(Image.open(io.BytesIO(zf.read(n.replace("/label/", "/image/")))).convert("RGB"))
        for v, cls in T2017.items():
            comp, k = ndimage.label(lab == v)
            for j in range(1, k + 1):
                mask = comp == j
                if mask.sum() < MIN_AREA:
                    continue
                ys, xs = np.nonzero(mask)
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                for lb in crops:
                    crops[lb].append(C.masked_crop(img, mask, box, lb))
                y.append(cls)
    return crops, np.array(y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--neighbours", type=Path, required=True)
    ap.add_argument("--zip2017", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    insts = C.load_instances("train")
    y18 = np.array([i["cls"] for i in insts])
    meta = json.loads((args.neighbours / "meta.json").read_text())
    options = {int(k): [args.neighbours / "crops" / it["file"] for it in v] for k, v in meta.items() if int(k) < len(insts)}
    crops = {lb: [] for lb in (True, False)}
    cache: dict[tuple[str, int], np.ndarray] = {}
    for inst in insts:
        key = (inst["seq"], inst["frame"])
        if key not in cache:
            cache.clear()
            cache[key] = C.load_frame(*key)
        mask = C.decode(inst["rle"])
        for lb in crops:
            crops[lb].append(C.masked_crop(cache[key], mask, inst["box"], lb))
    c17, y17 = crops_2017(args.zip2017)
    for lb in crops:
        crops[lb] += c17[lb]
    y = np.concatenate([y18, y17])
    print(f"seed {args.seed}: {len(y18)} 2018 + {len(y17)} 2017 train instances, per class {np.bincount(y, minlength=7).tolist()}, "
          f"{len(options)} with neighbour crops", flush=True)
    log: list[dict] = []
    for cfg in es1.MEMBERS:
        if args.only and cfg["label"] not in args.only:
            continue
        model = G.train_member(cfg, crops[cfg["letterbox"]], y, options, torch.device(args.device), args.seed, log)
        torch.save(model.state_dict(), args.out_dir / f"{cfg['label']}.pt")
        del model
        torch.cuda.empty_cache()
    tag = "_".join(args.only) if args.only else "all"
    (args.out_dir / f"train_log_{tag}.json").write_text(json.dumps({"epochs": G.EPOCHS, "anneal": G.ANNEAL, "lambda": G.LAMBDA, "clamp": G.CLAMP,
                                                                    "neighbour_prob": G.NEIGHBOUR_PROB, "extra_data": "EndoVis 2017 (types 1,2,3,6,7->UP)",
                                                                    "train_per_class": np.bincount(y, minlength=7).tolist(), "log": log}))


if __name__ == "__main__":
    main()
