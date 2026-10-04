"""Official-test evaluation of the PI's pipeline: ground-truth box -> SAM-family mask -> evidential classifier -> evidence gate ->
SAM2 tracking of the gated instruments' predicted masks -> label, scored with the benchmark's three semantic IoUs (mIoU, IoU,
mcIoU) against the JSON-painted ground truth (identical to the dataset's PNG maps on the training cases).

Configurations, each painted into per-frame semantic maps with SAM's masks:
    single pass          label of the four-member single pass, nothing tracked
    gated (tau)          instruments with S1 >= tau take the tracked, evidence-weighted label (tau fixed on fold1 before the test)
    gated (budget N)     the N instruments with the highest S1 are tracked
    oracle classes       SAM's masks with the ground-truth class: the ceiling the segmentor alone allows

The published numbers (GraSP paper, Table 4, test set) are included for the comparison; they come from detectors that find the
instruments themselves, while this pipeline is given the ground-truth boxes.

Usage (titanxp, surgical environment):
    python scripts/gtbox_sam_final_eval.py --masks experiments/gtbox_sam/finetuned/masks.pkl --logits experiments/gtbox_sam/finetuned/logits.npz \\
        --tracked-dir experiments/gtbox_sam/finetuned/tracked --out docs/reports/gtbox_sam/finetuned.json
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, ORDER, alpha_mix
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

PUBLISHED_TEST = {  # GraSP paper (arXiv 2401.11174v3), Table 4, instrument segmentation, test set
    "TAPIS (Mask2Former Swin-L + video transformer)": {"mAP@0.5 box": 89.85, "mAP@0.5 segm": 89.10, "mIoU": 86.61, "IoU": 83.38, "mcIoU": 77.42},
    "TAPIS-VST": {"mAP@0.5 box": 90.29, "mAP@0.5 segm": 89.58, "mIoU": 86.36, "IoU": 83.51, "mcIoU": 77.54},
    "SlowFast": {"mAP@0.5 box": 74.33, "mAP@0.5 segm": 71.32, "mIoU": 77.16, "IoU": 72.26, "mcIoU": 58.75},
}
PUBLISHED_CROSSVAL = {  # same paper, Table 10, cross-validation set (a different split from the test set)
    "ISINet (R50)": {"mAP@0.5 box": 79.85, "mAP@0.5 segm": 78.29, "mIoU": 78.44, "IoU": 70.85, "mcIoU": 56.67},
}


def load_tracked(tracked_dir: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for path in sorted(tracked_dir.glob("*frames*.npz")):
        z = np.load(path)
        for key in z.files:
            if key.startswith("det_"):
                out[int(key[4:])] = z[key]
    return out


def fuse(det_frames: np.ndarray, weights: dict) -> tuple[int, float]:
    """Frame-weighted evidential fusion of one track (frames, members, classes): the class and its share of the combined mass."""
    alpha = sum(w * alpha_from_logits(det_frames[:, ORDER.index(k), :]) for k, w in weights.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    combined = (mu.max(axis=1)[:, None] * mu).sum(axis=0)
    return int(combined.argmax()), float(combined.max() / combined.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--tracked-dir", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=1.7e-5, help="fold1-chosen evidential gate threshold of the shipped pipeline")
    ap.add_argument("--budgets", type=int, nargs="*", default=[539, 833])
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    weights = CONFIGS["four"]
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    masks = pickle.loads(args.masks.read_bytes())
    z = np.load(args.logits)
    y, done = z["y"], z["done"]
    alpha = alpha_mix(lambda k: z[f"det_{k}"], weights)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    base = mu.argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    tracked = load_tracked(args.tracked_dir)

    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    sam = {idx: mask_codec.decode(rle).astype(bool) for idx, rle in masks.items()}

    def clip_to_box(mask: np.ndarray, box: tuple) -> np.ndarray:
        """The mask restricted to the given ground-truth box (pixel centres inside it); the box is an input of the pipeline, so this costs nothing."""
        x, y, w, h = box
        inside = ((np.arange(mask.shape[0]) + 0.5 >= y) & (np.arange(mask.shape[0]) + 0.5 <= y + h))[:, None] & \
                 ((np.arange(mask.shape[1]) + 0.5 >= x) & (np.arange(mask.shape[1]) + 0.5 <= x + w))[None, :]
        return mask & inside

    sam_clip = {idx: clip_to_box(m, ds.instances[idx][2]) for idx, m in sam.items()}

    def final_labels(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        pred, score, missing = base.copy(), mu.max(axis=1).copy(), 0
        for i in np.where(flag & done)[0]:
            if int(i) in tracked:
                pred[i], score[i] = fuse(tracked[int(i)], weights)
            else:
                missing += 1
        return pred, score, missing

    def score_config(pred: np.ndarray, score: np.ndarray, clip: bool = False) -> dict:
        used = sam_clip if clip else sam
        frames = []
        for file_name, indices in by_frame.items():
            if any(i not in masks for i in indices):  # a partial run: only frames whose instruments were all segmented
                continue
            shape = tuple(ds.instances[indices[0]][1]["size"])
            gt = paint(shape, [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y[i]) + 1, 0.0) for i in indices])
            insts = [(used[i], int(pred[i]) + 1, float(score[i])) for i in indices if i in used and done[i]]
            frames.append(frame_class_ious(paint(shape, insts), gt))
        out = aggregate(frames)
        keep = done
        out["instance_accuracy"] = float((pred[keep] == y[keep]).mean())
        out["instance_macro_f1"] = float(f1_score(y[keep], pred[keep], average="macro", labels=range(7)))
        return out

    results = {"instruments": int(len(y)), "classified": int(done.sum())}
    def record(name: str, pred: np.ndarray, score: np.ndarray, **extra) -> None:
        """Each configuration twice: the segmentor's masks as they are, and clipped to the given box ("+box clip")."""
        results[name] = {**score_config(pred, score), **extra}
        results[f"{name} +box clip"] = {**score_config(pred, score, clip=True), **extra}

    record("single pass", base, mu.max(axis=1))
    record("oracle classes", y, np.ones(len(y)))
    flag = s1 >= args.tau
    pred, score, missing = final_labels(flag)
    record(f"gated (tau {args.tau:.1e})", pred, score, tracked=int(flag.sum()), tracks_missing=missing)
    rank = np.argsort(-s1, kind="stable")
    for budget in args.budgets:
        flag = np.zeros(len(y), bool)
        flag[rank[:budget]] = True
        pred, score, missing = final_labels(flag)
        record(f"gated (top {budget})", pred, score, tracked=budget, tracks_missing=missing)
    results["published_test"] = PUBLISHED_TEST
    results["published_crossval"] = PUBLISHED_CROSSVAL

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"{'configuration':<26}{'mIoU':>8}{'IoU':>8}{'mcIoU':>8}{'inst acc':>10}{'macro-F1':>10}")
    for name, r in results.items():
        if isinstance(r, dict) and "mIoU" in r:
            print(f"{name:<26}{100 * r['mIoU']:>8.2f}{100 * r['IoU']:>8.2f}{100 * r['mcIoU']:>8.2f}{r['instance_accuracy']:>10.4f}{r['instance_macro_f1']:>10.4f}")
    for name, r in {**PUBLISHED_TEST, **PUBLISHED_CROSSVAL}.items():
        print(f"{'published: ' + name:<26}{r['mIoU']:>8.2f}{r['IoU']:>8.2f}{r['mcIoU']:>8.2f}")


if __name__ == "__main__":
    main()
