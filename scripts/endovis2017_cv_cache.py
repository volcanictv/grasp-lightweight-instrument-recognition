"""EndoVis 2017 four-fold cross-validation caches in the format of the EndoVis 2018 GT-box pipeline (scripts/endovis_gtbox_common.py), so that pipeline's scripts run unchanged per fold with ENDOVIS_GT=<out>/fold<k>.

Class scheme of 2017 (type - 1): 0 BF, 1 PF, 2 LND, 3 VS (Vessel Sealer), 4 GR (Grasping Retractor), 5 MCS, 6 Other (ultrasound probe). Per fold:
  val   the instances of the two held-out sequences (TernausNet / ISINet folds {1,3}, {2,5}, {4,8}, {6,7}; scripts/endovis2017_tapis_prep.py), annotation masks
  train the instances of the other six sequences (training release only, no official test frames) plus the EndoVis 2018 training instances (ISINet split), mapped into the 2017 scheme:
        BF, PF, LND keep their class, MCS -> 5, Ultrasound Probe -> 6, Suction Instrument and Clip Applier have no 2017 counterpart and are dropped
        (2018 instances come first, so the tracker-style neighbour crops of the 2018 run still apply; 2017 instances have none)
Writes <out>/frames (2017 frames as seq_17NN_frameNNN.jpg, 2018 frames linked), <out>/fold<k>/{instances.pkl, frames -> ../frames, masks.pkl, neighbours/{meta.json, crops -> the 2018 neighbour crops}}.

Usage (titanxp): python scripts/endovis2017_cv_cache.py --prep ~/endovis2017_tapis --cache18 ~/endovis_isinet --neigh18 ~/grasp_yolo26/experiments/endovis_isinet_e17/neighbours --out ~/endovis2017_cv
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

MAP18 = {0: 0, 1: 1, 2: 2, 3: 5, 4: 6}
NAMES = ["BF", "PF", "LND", "VS", "GR", "MCS", "Other"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--cache18", type=Path, required=True)
    ap.add_argument("--neigh18", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    # frames: 2017 as JPEG with a sequence id that cannot clash with 2018, 2018 linked
    t17 = pickle.loads((args.prep / "instances2017.pkl").read_bytes())
    for seq, frame in sorted({(d["seq"], d["frame"]) for d in t17}):
        dst = frames / f"seq_{1700 + seq}_frame{frame:03d}.jpg"
        if not dst.exists():
            Image.open(args.prep / "images" / f"seq{seq}_frame{frame:03d}.png").convert("RGB").save(dst, quality=95)
    for src in (args.cache18 / "frames").glob("*.jpg"):
        dst = frames / src.name
        if not dst.exists():
            os.symlink(src, dst)

    # 2018 training instances in the 2017 scheme
    tab18 = pickle.loads((args.cache18 / "instances.pkl").read_bytes())["train"]
    keep18, new_of_old = [], {}
    for d in tab18:
        if d["cls"] in MAP18:
            new_of_old[d["i"]] = len(keep18)
            keep18.append({**d, "cls": MAP18[d["cls"]]})
    meta18 = json.loads((args.neigh18 / "meta.json").read_text())
    print(f"2018 training instances: {len(tab18)}, kept {len(keep18)} (Suction and Clip Applier dropped)", flush=True)

    inst17 = [{"seq": f"seq_{1700 + d['seq']}", "frame": d["frame"], "cls": d["cls"], "box": d["box"], "rle": d["rle"], "area": d["area"], "fold": d["fold"]} for d in t17]
    for k in range(4):
        fold = args.out / f"fold{k}"
        fold.mkdir(exist_ok=True)
        if not (fold / "frames").exists():
            os.symlink(Path("..") / "frames", fold / "frames")
        val = [{**d, "i": n} for n, d in enumerate(d for d in inst17 if d["fold"] == k)]
        tr17 = [d for d in inst17 if d["fold"] != k]
        train = [{**d, "i": n} for n, d in enumerate(keep18 + tr17)]
        (fold / "instances.pkl").write_bytes(pickle.dumps({"train": train, "val": val}))
        (fold / "masks.pkl").write_bytes(pickle.dumps({d["i"]: d["rle"] for d in val}))
        ndir = fold / "neighbours"
        ndir.mkdir(exist_ok=True)
        if not (ndir / "crops").exists():
            os.symlink(args.neigh18 / "crops", ndir / "crops")
        remapped = {str(new_of_old[int(k2)]): v for k2, v in meta18.items() if int(k2) in new_of_old}
        (ndir / "meta.json").write_text(json.dumps(remapped))
        c_tr, c_va = np.bincount([d["cls"] for d in train], minlength=7), np.bincount([d["cls"] for d in val], minlength=7)
        print(f"fold {k}: train {len(train)} ({len(keep18)} from 2018 + {len(tr17)} from 2017), per class {c_tr.tolist()}; val {len(val)}, per class {c_va.tolist()}; neighbour crops for {len(remapped)} instances", flush=True)


if __name__ == "__main__":
    main()
