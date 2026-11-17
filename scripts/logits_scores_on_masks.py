"""mIoU / IoU / mcIoU, accuracy and macro-F1 of saved single-model test logits (instances x 7, official test order) painted onto a
mask source. For the DINO classifiers of the recognition ablation, which have no segmentation scores of their own.

Usage: python scripts/logits_scores_on_masks.py --masks M.pkl --logits a.npy b.npy --names a b --out result.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from pycocotools import mask as mask_codec
from sklearn.metrics import f1_score

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--logits", type=Path, nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    masks = pickle.loads(args.masks.read_bytes())
    y = np.array([inst[3] for inst in ds.instances])
    logits = [np.load(p) for p in args.logits]
    preds = [l.argmax(1) for l in logits]
    conf = [(lambda e: (e / e.sum(1, keepdims=True)).max(1))(np.exp(l - l.max(1, keepdims=True))) for l in logits]
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, *_rest) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    per = [[] for _ in preds]
    for file_name, indices in by_frame.items():
        shape = tuple(ds.instances[indices[0]][1]["size"])
        gt = paint(shape, [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y[i]) + 1, 0.0) for i in indices])
        used = {i: mask_codec.decode(masks[i]).astype(bool) for i in indices}
        for k in range(len(preds)):
            per[k].append(frame_class_ious(paint(shape, [(used[i], int(preds[k][i]) + 1, float(conf[k][i])) for i in indices]), gt))
    out = {}
    for k, name in enumerate(args.names):
        r = aggregate(per[k])
        r["instance_accuracy"] = float((preds[k] == y).mean())
        r["instance_macro_f1"] = float(f1_score(y, preds[k], average="macro", labels=range(7)))
        out[name] = r
        print(f"{name:<28}" + "".join(f"{100 * r[m]:>9.2f}" for m in ("mIoU", "IoU", "mcIoU", "instance_accuracy", "instance_macro_f1")))
    args.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
