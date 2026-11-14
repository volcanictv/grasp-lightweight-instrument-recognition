"""Write the annotated instance masks of the EndoVis 2018 held-out set as masks_ens.pkl ({instrument index: COCO RLE}), the layout of the predicted masks of
scripts/endovis_gtbox_masks.py, so the single-pass, tracking and scoring scripts run unchanged with annotation masks (the variant keeps the name "ens" for the scorer).

Usage (titanxp): ENDOVIS_GT=$HOME/endovis_isinet ENDOVIS_ISINET=1 python scripts/endovis_gt_masks_pkl.py --out experiments/endovis_isinet_gtmask/masks/masks_ens.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import endovis_gtbox_common as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    insts = C.load_instances(args.split)
    masks = {inst["i"]: inst["rle"] for inst in insts}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps(masks))
    print(f"wrote {len(masks)} annotation masks to {args.out}")


if __name__ == "__main__":
    main()
