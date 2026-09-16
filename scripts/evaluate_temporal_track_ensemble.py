"""SAM2-propagated temporal track aggregation, using the full 4-way Task B
ensemble (`configs/region_ensemble.yaml`, macro-F1 0.8929, this project's
reported best classifier) instead of a single checkpoint, on the full
official test set -- not just the hard-case subset
`evaluate_temporal_track_classification.py` targets.

Same mechanism as that script (propagate the real GT mask +/-`--window`
frames via SAM2, classify every frame, aggregate), generalized to combine
all 4 ensemble members per frame using the ensemble's own weighting before
the track-level aggregation. Every instance gets a GT-mask-initialized
track regardless of its own area -- this is the full-population number,
not a hard-case-only one.

Supports sharding (`--shard-id`/`--num-shards`) so the ~2,861-instance
official test set can run split across both Titan Xps in parallel, and
checkpoints its own progress to the output JSON every `--save-every`
instances (this run is long enough that losing all progress to a late
crash would be a real cost, unlike the smaller fold1/fold2 runs).

Usage:
    python scripts/evaluate_temporal_track_ensemble.py \\
        --ensemble-config configs/region_ensemble.yaml \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt \\
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \\
        --shard-id 0 --num-shards 2 --device cuda:0
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
import yaml
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset, _pad_to_square
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--error-cases-json", type=Path, default=None,
                         help="restrict to the instance indices in this file's 'errors' list "
                              "(scripts/find_ensemble_error_cases.py output) instead of sharding the whole split")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/sam2_track_ensemble"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def crop_from_box(frame: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], letterbox: bool, letterbox_min_aspect: float = 1.0) -> np.ndarray | None:
    """Center-frame crop only -- matches GraspRegionDataset's own bbox+mask
    convention exactly (the annotated box, not the mask's own pixel extent),
    so the single-frame prediction used as this script's baseline is
    identical to the official evaluation's, not a different crop
    methodology (docs/DECISIONS.md 2026-09-16 -- a real, measured 4.6%
    disagreement rate came from this mismatch before the fix)."""
    x, y, w, h = box
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)
    if letterbox:
        ch, cw = crop.shape[:2]
        aspect = max(ch, cw) / max(1, min(ch, cw))
        if aspect >= letterbox_min_aspect:
            crop = _pad_to_square(crop)
    return crop


def crop_from_mask(frame: np.ndarray, mask: np.ndarray, letterbox: bool, letterbox_min_aspect: float = 1.0) -> np.ndarray | None:
    """Propagated (non-center) frames only -- no original annotation box
    exists for these, so the mask's own pixel extent is the only option."""
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

    ensemble_config = yaml.safe_load(args.ensemble_config.read_text())
    members_cfg = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]
    w_rest = (1 - weight_320) / (len(members_cfg) - 1)

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    class_names = ds.class_names_ordered()

    models, transforms, weights = [], [], []
    for m in members_cfg:
        model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()
        models.append(model)
        transforms.append((build_transforms(m["image_size"], train=False), m["letterbox"]))
        weights.append(weight_320 if m["label"] == "resnet50_320" else w_rest)
    print(f"loaded {len(models)} ensemble members, weights={weights}")

    if args.error_cases_json is not None:
        error_data = json.loads(args.error_cases_json.read_text())
        indices = [e["index"] for e in error_data["errors"]]
    else:
        indices = list(range(len(ds.instances)))[args.shard_id :: args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(indices)} of {len(ds.instances)} instances")

    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    results = []
    for n, idx in enumerate(indices):
        file_name, segmentation, box, label = ds.instances[idx]
        case, frame_stem = file_name.split("/")
        center_num = int(frame_stem.replace(".jpg", ""))
        gt_mask = decode_instance_mask(segmentation).astype(bool)
        area_pct = float(gt_mask.sum() / (segmentation["size"][0] * segmentation["size"][1]) * 100)

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
        frame_arrays = {li: np.array(Image.open(tmp_dir / f"{li:05d}.jpg").convert("RGB")) for li in sorted_local_idxs}

        # two crop variants per frame -- letterbox=True (3 of 4 members) and letterbox=False (baseline mobilenet)
        crop_cache: dict[tuple[int, bool], np.ndarray | None] = {}

        def get_crop(local_idx: int, letterbox: bool) -> np.ndarray | None:
            key = (local_idx, letterbox)
            if key not in crop_cache:
                if local_idx == center_idx:
                    crop_cache[key] = crop_from_box(frame_arrays[local_idx], track_masks[local_idx], box, letterbox)
                else:
                    crop_cache[key] = crop_from_mask(frame_arrays[local_idx], track_masks[local_idx], letterbox)
            return crop_cache[key]

        valid_local_idxs = [li for li in sorted_local_idxs if get_crop(li, True) is not None]
        if not valid_local_idxs:
            print(f"[{n}] {case}/{frame_stem} {class_names[label]}: no usable track frames, skipping")
            continue

        per_frame_ensemble_probs = []
        for li in valid_local_idxs:
            combined = np.zeros(len(class_names), dtype=np.float64)
            for model, (transform, letterbox), weight in zip(models, transforms, weights):
                crop = get_crop(li, letterbox)
                image = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(image), dim=1).cpu().numpy()[0]
                combined += weight * probs
            per_frame_ensemble_probs.append(combined)
        per_frame_ensemble_probs = np.stack(per_frame_ensemble_probs)

        center_pos = valid_local_idxs.index(center_idx) if center_idx in valid_local_idxs else 0
        single_pred = int(per_frame_ensemble_probs[center_pos].argmax())

        per_frame_preds = per_frame_ensemble_probs.argmax(axis=1)
        majority_pred = int(Counter(per_frame_preds.tolist()).most_common(1)[0][0])
        avg_softmax_pred = int(per_frame_ensemble_probs.mean(axis=0).argmax())

        results.append({
            "index": idx, "case": case, "frame": frame_stem, "true": class_names[label], "area_pct": area_pct,
            "track_len": len(valid_local_idxs),
            "single_frame_pred": class_names[single_pred], "single_frame_correct": single_pred == label,
            "majority_vote_pred": class_names[majority_pred], "majority_vote_correct": majority_pred == label,
            "avg_softmax_pred": class_names[avg_softmax_pred], "avg_softmax_correct": avg_softmax_pred == label,
        })
        print(f"[{n}/{len(indices)}] {case}/{frame_stem} {class_names[label]} (area={area_pct:.2f}%, track_len={len(valid_local_idxs)}): "
              f"single={'OK' if single_pred == label else class_names[single_pred]} "
              f"avg_softmax={'OK' if avg_softmax_pred == label else class_names[avg_softmax_pred]}")

        if (n + 1) % args.save_every == 0 or n == len(indices) - 1:
            _write_summary(args, results)

    _write_summary(args, results)


def _write_summary(args: argparse.Namespace, results: list[dict]) -> None:
    n_total = len(results)
    single_acc = sum(r["single_frame_correct"] for r in results) / n_total if n_total else float("nan")
    majority_acc = sum(r["majority_vote_correct"] for r in results) / n_total if n_total else float("nan")
    avg_acc = sum(r["avg_softmax_correct"] for r in results) / n_total if n_total else float("nan")
    summary = {
        "split": args.split, "shard_id": args.shard_id, "num_shards": args.num_shards,
        "window": args.window, "n_instances": n_total,
        "single_frame_accuracy": single_acc, "majority_vote_accuracy": majority_acc, "avg_softmax_accuracy": avg_acc,
        "instances": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
