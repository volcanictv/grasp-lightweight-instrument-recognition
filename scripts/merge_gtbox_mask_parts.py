"""Merges the per-GPU parts of scripts/gtbox_sam_masks_ensemble.py into the masks.pkl / summary.json / ious.npy layout that
scripts/run_gtbox_sam_pipeline.sh reads. Usage: python scripts/merge_gtbox_mask_parts.py --parts p0.pkl p1.pkl --variant gtft_ens --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", type=Path, nargs="+", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    masks, ious, labels = {}, {}, {}
    for p in args.parts:
        d = pickle.loads(p.read_bytes())
        masks.update(d["masks"])
        ious.update(d["ious"])
        labels.update(d["labels"])
    idx = np.array(sorted(masks))
    v = np.array([ious[i] for i in idx])
    lab = np.array([labels[i] for i in idx])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "masks.pkl").write_bytes(pickle.dumps(masks))
    full = np.zeros(int(idx.max()) + 1)
    full[idx] = v
    np.save(args.out_dir / "ious.npy", full)
    summary = {"variant": args.variant, "instruments": int(len(idx)), "mean_mask_iou": float(v.mean()), "iou_ge_0.5": float((v >= 0.5).mean()),
               "iou_ge_0.75": float((v >= 0.75).mean()),
               "per_class_mean_iou": {CLASSES[c]: float(v[lab == c].mean()) for c in range(7) if (lab == c).any()}}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
