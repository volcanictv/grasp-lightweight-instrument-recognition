"""Same temporal track aggregation as evaluate_temporal_track_ensemble.py, with
SAM2 mask propagation replaced by a trained YOLO26-seg model.

SAM2 is prompted with the ground-truth mask; YOLO cannot be, so it segments
every frame in the +/-window and the track is built by walking outward from the
center frame, at each step taking the detection whose mask has the highest IoU
with the previous frame's track mask (class-agnostic: the classifier, not YOLO,
decides the label). A direction ends when no detection reaches --min-iou, so a
lost track shortens the window instead of inheriting a wrong instrument.

Output is the same tracked_shard JSON the scorers already read
(evaluate_mc_gate_tracking.py), plus per-instance wall time and track lengths
so the tracker can be compared with SAM2 on cost as well as accuracy.

Usage:
    python scripts/evaluate_temporal_track_yolo.py \\
        --yolo-weights experiments/<run>/weights/best.pt \\
        --error-cases-json docs/reports/tracking_mcgate/track_indices_shard0.json \\
        --out experiments/yolo_tracking/tracked_shard0.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import build_track_frame_nums, crop_from_box, crop_from_mask
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--causal", action="store_true", help="past frames only")
    parser.add_argument("--min-iou", type=float, default=0.3)
    parser.add_argument("--detections-cache", type=Path, default=None,
                        help="pickle from scripts/cache_yolo_detections.py; skips YOLO inference")
    parser.add_argument("--coast", type=int, default=0, help="consecutive missed frames a direction survives")
    parser.add_argument("--center-fallback", action="store_true", help="also match detections against the center-frame mask")
    parser.add_argument("--yolo-conf", type=float, default=0.1)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--error-cases-json", type=Path, default=None,
                        help="restrict to the instance indices in this file's 'errors' list instead of the whole split")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame-logits-out", type=Path, default=None,
                        help="npz of raw per-frame member logits, det_<index> shaped (frames, members, classes), "
                             "the layout the evidential scorers read")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def mask_iou(a: dict, b: dict) -> float:
    return float(mask_codec.iou([a], [b], [0])[0][0])


def encode(mask: np.ndarray) -> dict:
    return mask_codec.encode(np.asfortranarray(mask.astype(np.uint8)))


def walk_track(ref: dict, center: dict, frame_masks: list[list[dict]], min_iou: float,
               coast: int, center_fallback: bool) -> dict[int, dict]:
    """Masks are COCO RLE. frame_masks is ordered outward from the center; returns {position: mask}. A
    direction ends after more than `coast` consecutive frames with no detection at or above min_iou. With
    center_fallback a detection also qualifies by its IoU with the center-frame mask, which recovers an
    instrument the chain lost to one bad frame."""
    track: dict[int, dict] = {}
    misses = 0
    for pos, candidates in enumerate(frame_masks):
        scored = [(max(mask_iou(ref, m), mask_iou(center, m) if center_fallback else 0.0), m) for m in candidates]
        best = max(scored, key=lambda s: s[0], default=(0.0, None))
        if best[1] is not None and best[0] >= min_iou:
            ref = best[1]
            track[pos] = ref
            misses = 0
        else:
            misses += 1
            if misses > coast:
                break
    return track


def detect(yolo, paths: list[str], args: argparse.Namespace) -> list[list[dict]]:
    results = yolo.predict(paths, conf=args.yolo_conf, imgsz=args.yolo_imgsz, retina_masks=True,
                           device=args.device, verbose=False, stream=False)
    return [[] if r.masks is None else [encode(m) for m in r.masks.data.cpu().numpy().astype(bool)] for r in results]


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    device = torch.device(args.device)
    frames_root = args.data_root / "frames-001" / "frames"

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members_cfg = cfg["members"]
    weight_320 = cfg["weight_resnet50_320"]
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

    if args.error_cases_json is not None:
        indices = [e["index"] for e in json.loads(args.error_cases_json.read_text())["errors"]]
    else:
        indices = list(range(len(ds.instances)))
    print(f"{len(indices)} of {len(ds.instances)} instances")

    cache = pickle.loads(args.detections_cache.read_bytes()) if args.detections_cache else None
    yolo = None if cache is not None else YOLO(str(args.yolo_weights))
    results = []
    frame_store: dict[str, np.ndarray] = {}
    for n, idx in enumerate(indices):
        start = time.time()
        file_name, segmentation, box, label = ds.instances[idx]
        case, frame_stem = file_name.split("/")
        center_num = int(frame_stem.replace(".jpg", ""))
        gt_mask = decode_instance_mask(segmentation).astype(bool)
        area_pct = float(gt_mask.sum() / (segmentation["size"][0] * segmentation["size"][1]) * 100)

        frame_nums, center_idx = build_track_frame_nums(frames_root, case, center_num, args.window, 0 if args.causal else None)
        paths = [str(frames_root / case / f"{fn:05d}.jpg") for fn in frame_nums]
        per_frame = [cache[p] for p in paths] if cache is not None else detect(yolo, paths, args)

        gt_rle = encode(gt_mask)
        forward = walk_track(gt_rle, gt_rle, per_frame[center_idx + 1:], args.min_iou, args.coast, args.center_fallback)
        backward = walk_track(gt_rle, gt_rle, per_frame[:center_idx][::-1], args.min_iou, args.coast, args.center_fallback)
        track_masks = {center_idx: gt_mask}
        track_masks.update({center_idx + 1 + i: mask_codec.decode(m).astype(bool) for i, m in forward.items()})
        track_masks.update({center_idx - 1 - i: mask_codec.decode(m).astype(bool) for i, m in backward.items()})
        sorted_local_idxs = sorted(track_masks)
        frame_arrays = {li: np.array(Image.open(paths[li]).convert("RGB")) for li in sorted_local_idxs}

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
        per_frame_probs, per_frame_logits = [], []
        for li in valid_local_idxs:
            combined = np.zeros(len(class_names), dtype=np.float64)
            member_logits = []
            for model, (transform, letterbox), weight in zip(models, transforms, weights):
                image = transform(Image.fromarray(get_crop(li, letterbox))).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(image)
                member_logits.append(logits.float().cpu().numpy()[0])
                combined += weight * torch.softmax(logits, dim=1).cpu().numpy()[0]
            per_frame_probs.append(combined)
            per_frame_logits.append(np.stack(member_logits))
        per_frame_probs = np.stack(per_frame_probs)
        frame_store[f"det_{idx}"] = np.stack(per_frame_logits)

        center_pos = valid_local_idxs.index(center_idx)
        single_pred = int(per_frame_probs[center_pos].argmax())
        majority_pred = int(Counter(per_frame_probs.argmax(axis=1).tolist()).most_common(1)[0][0])
        avg_softmax_pred = int(per_frame_probs.mean(axis=0).argmax())

        results.append({
            "index": idx, "case": case, "frame": frame_stem, "true": class_names[label], "area_pct": area_pct,
            "track_len": len(valid_local_idxs), "frames_forward": len(forward), "frames_backward": len(backward),
            "window_frames": len(frame_nums) - 1, "seconds": time.time() - start,
            "single_frame_pred": class_names[single_pred], "single_frame_correct": single_pred == label,
            "majority_vote_pred": class_names[majority_pred], "majority_vote_correct": majority_pred == label,
            "avg_softmax_pred": class_names[avg_softmax_pred], "avg_softmax_correct": avg_softmax_pred == label,
        })
        print(f"[{n}/{len(indices)}] {case}/{frame_stem} {class_names[label]} track_len={len(valid_local_idxs)} "
              f"single={'OK' if single_pred == label else class_names[single_pred]} "
              f"avg_softmax={'OK' if avg_softmax_pred == label else class_names[avg_softmax_pred]} ({results[-1]['seconds']:.1f}s)")

        if (n + 1) % args.save_every == 0 or n == len(indices) - 1:
            write_summary(args, results)
            if args.frame_logits_out is not None:
                args.frame_logits_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.frame_logits_out, **frame_store)


def write_summary(args: argparse.Namespace, results: list[dict]) -> None:
    n = len(results)
    acc = lambda key: sum(r[key] for r in results) / n if n else float("nan")
    summary = {
        "tracker": "yolo26", "yolo_weights": str(args.yolo_weights), "min_iou": args.min_iou, "coast": args.coast, "center_fallback": args.center_fallback, "yolo_conf": args.yolo_conf,
        "split": args.split, "shard_id": 0, "num_shards": 1, "window": args.window, "causal": args.causal, "n_instances": n,
        "single_frame_accuracy": acc("single_frame_correct"), "majority_vote_accuracy": acc("majority_vote_correct"),
        "avg_softmax_accuracy": acc("avg_softmax_correct"), "mean_seconds": acc("seconds"), "instances": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
