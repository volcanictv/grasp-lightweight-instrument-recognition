"""Checks that the handoff package (grasp_classifier.evidential) reproduces the research pipeline's single-pass logits on real official-test
instruments: same checkpoints, same masks (the final run's masks.pkl), compared with the logits that run saved (final/logits_s<seed>.npz).

Usage (titanxp, surgical environment, the package copied to ~/handoff_test with the members under weights/evidential_armN_seed44/):
    python scripts/parity_check_handoff.py --package ~/handoff_test --seed 44 --n 60
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image
from pycocotools import mask as mask_codec

from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()
    sys.path.insert(0, str(args.package))
    from grasp_classifier.evidential import EvidentialEnsembleClassifier

    clf = EvidentialEnsembleClassifier(args.package / "evidential_config.yaml", device=args.device)
    ref = np.load(REPO_ROOT / "experiments" / "gtbox_sam" / "final" / f"logits_s{args.seed}.npz")
    masks = pickle.loads((REPO_ROOT / "experiments" / "gtbox_sam" / "final" / "masks.pkl").read_bytes())
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    idxs = np.linspace(0, len(ds.instances) - 1, args.n).round().astype(int)
    frames_root = args.data_root / "frames-001" / "frames"
    diffs, agree = [], []
    for idx in idxs:
        if int(idx) not in masks:
            continue
        file_name, _seg, box, _label = ds.instances[int(idx)]
        frame = np.array(Image.open(frames_root / file_name).convert("RGB"))
        mask = mask_codec.decode(masks[int(idx)]).astype(bool)
        if not mask.any():
            continue
        mine = clf.member_logits(frame, tuple(int(v) for v in box), mask)
        theirs = np.stack([ref[f"det_{label}"][int(idx)] for label in clf.member_labels])
        diffs.append(float(np.abs(mine - theirs).max()))
        agree.append(int(mine.argmax(axis=1).tolist() == theirs.argmax(axis=1).tolist()))
    print(f"{len(diffs)} instruments compared; member order {clf.member_labels}")
    print(f"max abs logit difference: max {max(diffs):.2e}, median {float(np.median(diffs)):.2e}; every member's argmax agrees on {sum(agree)}/{len(agree)}")


if __name__ == "__main__":
    main()
