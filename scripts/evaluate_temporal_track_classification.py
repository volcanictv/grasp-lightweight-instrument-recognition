"""Tests whether SAM2 mask propagation, initialized from a real ground-truth
mask, can make Task B classification more robust on the hard-case
population (small mask-area instances -- `docs/error_analysis.md` found
Task B's error rate scales >3x from the largest to smallest instance crop
quartile).

Mechanism (not training-data augmentation -- see `docs/DECISIONS.md`
2026-09-15 for why that's a different, deliberately deferred idea): for
each hard instance, propagate its GT mask forward and backward across a
window of real, already-existing video frames (no new annotation), crop
each propagated frame the same way `GraspRegionDataset` does, classify
every frame in the track with the existing trained checkpoint (unchanged
weights), and aggregate the per-frame predictions into one track-level
label. The hypothesis: a single hard frame (tiny area, motion blur, mid-
occlusion) has ambiguous signal, but neighboring frames of the same
physical instrument often don't -- if so, aggregation should recover
accuracy the single center frame alone can't.

Usage:
    python scripts/evaluate_temporal_track_classification.py \\
        --split fold1 \\
        --checkpoint experiments/region_letterbox_resnet50_320_fold1_20260903-012254/best.pt \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt \\
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset, _pad_to_square
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="fold1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--letterbox-min-aspect", type=float, default=1.0)
    parser.add_argument("--area-threshold", type=float, default=0.04, help="same area-fraction definition used by the abstention gate and oversampling work")
    parser.add_argument("--window", type=int, default=10, help="frames each side of the GT frame -- pilot validated 5-15 frame gaps as reliable")
    parser.add_argument("--max-instances", type=int, default=40)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/sam2_track_frames"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "temporal_track_classification.json")
    return parser.parse_args()


def crop_from_mask(frame: np.ndarray, mask: np.ndarray, letterbox: bool, letterbox_min_aspect: float) -> np.ndarray | None:
    """Same bbox+mask-multiply(+letterbox) recipe as GraspRegionDataset's
    default crop_mode, but the bbox comes from the mask itself -- a
    propagated frame has no separate GT bbox annotation to reuse."""
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)
    if letterbox:
        ch, cw = crop.shape[:2]
        aspect = max(ch, cw) / max(1, min(ch, cw))
        if aspect >= letterbox_min_aspect:
            crop = _pad_to_square(crop)
    return crop


def build_track_frame_nums(frames_root: Path, case: str, center_num: int, window: int) -> tuple[list[int], int]:
    """Widest contiguous run of existing frame files centered on center_num,
    clipped at case boundaries or any gap. Returns (frame_nums, center_local_index)."""
    frame_nums = [center_num]
    for offset in range(1, window + 1):
        if (frames_root / case / f"{center_num - offset:05d}.jpg").exists():
            frame_nums.insert(0, center_num - offset)
        else:
            break
    center_idx = len(frame_nums) - 1
    for offset in range(1, window + 1):
        if (frames_root / case / f"{center_num + offset:05d}.jpg").exists():
            frame_nums.append(center_num + offset)
        else:
            break
    return frame_nums, center_idx


def main() -> None:
    args = parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(args.device)
    frames_root = args.data_root / "frames-001" / "frames"

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True, letterbox_min_aspect=args.letterbox_min_aspect)
    class_names = ds.class_names_ordered()

    area_fracs = np.array([
        decode_instance_mask(seg).sum() / (seg["size"][0] * seg["size"][1])
        for _fn, seg, _box, _label in ds.instances
    ])
    hard_order = np.argsort(area_fracs)
    hard_indices = [i for i in hard_order if area_fracs[i] < args.area_threshold][: args.max_instances]
    print(f"selected {len(hard_indices)} hard instances (area < {args.area_threshold*100:.0f}% of frame) from {len(ds.instances)} total in split={args.split}")

    model = build_model(args.model, num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()
    transform = build_transforms(args.image_size, train=False)

    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    results = []
    for n, idx in enumerate(hard_indices):
        file_name, segmentation, box, label = ds.instances[idx]
        case, frame_stem = file_name.split("/")
        center_num = int(frame_stem.replace(".jpg", ""))
        gt_mask = decode_instance_mask(segmentation).astype(bool)

        frame_nums, center_idx = build_track_frame_nums(frames_root, case, center_num, args.window)

        tmp_dir = args.tmp_dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        for local_idx, fn in enumerate(frame_nums):
            shutil.copy(frames_root / case / f"{fn:05d}.jpg", tmp_dir / f"{local_idx:05d}.jpg")

        state = predictor.init_state(video_path=str(tmp_dir))
        predictor.add_new_mask(state, frame_idx=center_idx, obj_id=1, mask=gt_mask)

        track_masks: dict[int, np.ndarray] = {center_idx: gt_mask}
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx != center_idx:
                track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=True):
            if frame_idx != center_idx:
                track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()

        sorted_local_idxs = sorted(track_masks)
        crops = []
        frame_area_fracs = []
        for local_idx in sorted_local_idxs:
            frame = np.array(Image.open(tmp_dir / f"{local_idx:05d}.jpg").convert("RGB"))
            mask = track_masks[local_idx]
            crop = crop_from_mask(frame, mask, letterbox=True, letterbox_min_aspect=args.letterbox_min_aspect)
            crops.append(transform(Image.fromarray(crop)) if crop is not None else None)
            frame_area_fracs.append(float(mask.sum() / (mask.shape[0] * mask.shape[1])))

        valid = [(i, c) for i, c in enumerate(crops) if c is not None]
        if not valid:
            print(f"[{n}] {case}/{frame_stem} {class_names[label]}: no usable track frames, skipping")
            continue

        batch = torch.stack([c for _i, c in valid]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()

        center_pos = [i for i, (li, _c) in enumerate(valid) if li == sorted_local_idxs.index(center_idx)][0]
        single_pred = int(probs[center_pos].argmax())

        per_frame_preds = probs.argmax(axis=1)
        majority_pred = int(Counter(per_frame_preds.tolist()).most_common(1)[0][0])
        avg_softmax_pred = int(probs.mean(axis=0).argmax())

        results.append({
            "case": case, "frame": frame_stem, "true": class_names[label], "true_idx": int(label),
            "area_pct": float(area_fracs[idx] * 100), "track_len": len(valid),
            "single_frame_pred": class_names[single_pred], "single_frame_correct": single_pred == label,
            "majority_vote_pred": class_names[majority_pred], "majority_vote_correct": majority_pred == label,
            "avg_softmax_pred": class_names[avg_softmax_pred], "avg_softmax_correct": avg_softmax_pred == label,
            # per-frame data for training a learnable aggregator downstream (docs/DECISIONS.md 2026-09-15):
            # offset is real frame distance from the GT center frame, softmax/area_frac let an aggregator
            # learn per-frame reliability instead of the fixed uniform-average rule used above.
            "frames": [
                {
                    "offset": sorted_local_idxs[li] - center_idx,
                    "area_frac": frame_area_fracs[li],
                    "softmax": probs[i].tolist(),
                }
                for i, (li, _c) in enumerate(valid)
            ],
        })
        print(f"[{n}] {case}/{frame_stem} {class_names[label]} (area={area_fracs[idx]*100:.2f}%, track_len={len(valid)}): "
              f"single={'OK' if single_pred == label else class_names[single_pred]} "
              f"majority={'OK' if majority_pred == label else class_names[majority_pred]} "
              f"avg_softmax={'OK' if avg_softmax_pred == label else class_names[avg_softmax_pred]}")

    n_total = len(results)
    single_acc = sum(r["single_frame_correct"] for r in results) / n_total if n_total else float("nan")
    majority_acc = sum(r["majority_vote_correct"] for r in results) / n_total if n_total else float("nan")
    avg_acc = sum(r["avg_softmax_correct"] for r in results) / n_total if n_total else float("nan")

    summary = {
        "split": args.split, "checkpoint": str(args.checkpoint), "area_threshold": args.area_threshold,
        "window": args.window, "n_instances": n_total,
        "single_frame_accuracy": single_acc, "majority_vote_accuracy": majority_acc, "avg_softmax_accuracy": avg_acc,
        "instances": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"\nhard-case subset (n={n_total}, area<{args.area_threshold*100:.0f}%, split={args.split}):")
    print(f"  single-frame accuracy:   {single_acc:.4f}")
    print(f"  majority-vote accuracy:  {majority_acc:.4f}  (delta {majority_acc - single_acc:+.4f})")
    print(f"  avg-softmax accuracy:    {avg_acc:.4f}  (delta {avg_acc - single_acc:+.4f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
