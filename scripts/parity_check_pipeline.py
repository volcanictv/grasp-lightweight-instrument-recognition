"""Checks the handoff package's segmenter and tracker against the research outputs of the final run.

Segmenter: BoxSegmenter (fine-tuned SAM2 plus SAM3, flip-averaged) on real official-test frames with their ground-truth boxes, against the masks the
research run saved (experiments/gtbox_sam/final/masks.pkl): per-instrument IoU between the two masks.
Tracker: EvidentialTracker on instruments the research run tracked, against the class the research run's saved per-frame logits fuse to
(experiments/gtbox_sam/final/tracked_s<seed>/frames_shard0.npz).

Usage (titanxp, surgical environment, package staged in ~/handoff_test with all weights):
    python scripts/parity_check_pipeline.py --package ~/handoff_test --seed 44 --frames 12 --tracks 6
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from PIL import Image
from pycocotools import mask as mask_codec

from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--tracks", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-tracking", action="store_true")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()
    sys.path.insert(0, str(args.package))
    from grasp_classifier.evidential import EvidentialEnsembleClassifier, fuse_frames
    from grasp_pipeline.segment import BoxSegmenter
    from grasp_pipeline.track import EvidentialTracker

    W = args.package / "weights"
    final = REPO_ROOT / "experiments" / "gtbox_sam" / "final"
    masks = pickle.loads((final / "masks.pkl").read_bytes())
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    by_frame = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    names = sorted(by_frame)
    frames_root = args.data_root / "frames-001" / "frames"

    segmenter = BoxSegmenter(W / "sam2.1_hiera_large.pt", sam2_delta=W / "sam2_delta.pt", sam3_delta=W / "sam3_delta.pt", device=args.device)
    ious = []
    for name in [names[i] for i in np.linspace(0, len(names) - 1, args.frames).round().astype(int)]:
        frame = np.array(Image.open(frames_root / name).convert("RGB"))
        idxs = by_frame[name]
        boxes = [ds.instances[i][2] for i in idxs]
        mine = segmenter.segment(frame, boxes)
        for i, m in zip(idxs, mine):
            ref = mask_codec.decode(masks[i]).astype(bool)
            union = np.logical_or(m, ref).sum()
            ious.append(float(np.logical_and(m, ref).sum() / union) if union else 1.0)
    ious = np.array(ious)
    print(f"segmenter: {len(ious)} instruments, mask IoU against the research masks: mean {ious.mean():.5f}, min {ious.min():.5f}, identical {(ious == 1.0).mean():.3f}")
    if args.skip_tracking:
        return

    tracked = np.load(final / f"tracked_s{args.seed}" / "frames_shard0.npz")
    keys = [int(k[4:]) for k in tracked.files if k.startswith("det_")]
    classifier = EvidentialEnsembleClassifier(args.package / "evidential_config.yaml", device=args.device)
    tracker = EvidentialTracker(classifier, W / "sam2.1_hiera_large.pt", device=args.device, window=10)
    rng = np.random.default_rng(0)
    agree, shown = [], 0
    for idx in rng.permutation(keys)[: args.tracks]:
        file_name, _seg, box, _label = ds.instances[int(idx)]
        case_dir = frames_root / file_name.split("/")[0]
        files = sorted(p for p in case_dir.iterdir() if p.suffix.lower() == ".jpg")
        center = [p.name for p in files].index(file_name.split("/")[1])
        lo, hi = max(0, center - 10), min(len(files) - 1, center + 10)
        frames = [np.array(Image.open(p).convert("RGB")) for p in files[lo:hi + 1]]
        mask = mask_codec.decode(masks[int(idx)]).astype(bool)
        pred = tracker.track(frames, center - lo, mask, tuple(int(v) for v in box))
        ref_idx, ref_share, _ = fuse_frames(tracked[f"det_{int(idx)}"], classifier.weights)
        from grasp_classifier.classes import CLASS_NAMES
        agree.append(pred.class_name == CLASS_NAMES[ref_idx])
        print(f"  instrument {int(idx)}: package {pred.class_name} ({pred.belief[pred.class_name]:.3f}), research {CLASS_NAMES[ref_idx]} ({ref_share:.3f})")
    print(f"tracker: fused class agrees with the research run on {sum(agree)}/{len(agree)} tracked instruments")


if __name__ == "__main__":
    main()
