"""Real end-to-end inference latency for the SAM2-propagated temporal-track
pipeline combined with the 4-way Task B ensemble (`docs/DECISIONS.md`
2026-09-15's combination test). Per CLAUDE.md's rule that an accuracy
result is incomplete without its latency cost, and per this project's own
standard (`src/surgical_ai/evaluation/benchmarking.py`): warm up first,
`torch.cuda.synchronize()` around every timed GPU section, report median
and p95, Titan Xp.

Deliberately measures on real instances end-to-end (SAM2 propagation +
frame I/O + all 4 ensemble members' forward passes), not a synthetic
dummy-tensor loop like the plain classifier benchmarks use -- SAM2's cost
depends on real track length and real disk reads, which a dummy tensor
can't represent. Uses fewer repeats than the usual >=200-run convention
(20 real instances, not 200) because each repeat costs seconds, not
milliseconds -- 200 real SAM2 propagations would itself take close to an
hour just to benchmark.

Usage:
    python scripts/benchmark_temporal_track_ensemble_latency.py \\
        --ensemble-config configs/region_ensemble.yaml \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt \\
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
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
from surgical_ai.evaluation.benchmarking import benchmark_gpu_latency
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/sam2_track_latency"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "temporal_track_ensemble_latency.json")
    return parser.parse_args()


def crop_from_mask(frame: np.ndarray, mask: np.ndarray, letterbox: bool, letterbox_min_aspect: float = 1.0) -> np.ndarray | None:
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


def run_one_instance(predictor, models, transforms, ds, idx, frames_root, window, tmp_dir, device):
    file_name, segmentation, _box, _label = ds.instances[idx]
    case, frame_stem = file_name.split("/")
    center_num = int(frame_stem.replace(".jpg", ""))
    gt_mask = decode_instance_mask(segmentation).astype(bool)
    frame_nums, center_idx = build_track_frame_nums(frames_root, case, center_num, window)

    t0 = time.perf_counter()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for local_idx, fn in enumerate(frame_nums):
        shutil.copy(frames_root / case / f"{fn:05d}.jpg", tmp_dir / f"{local_idx:05d}.jpg")
    t_io = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    state = predictor.init_state(video_path=str(tmp_dir))
    predictor.add_new_mask(state, frame_idx=center_idx, obj_id=1, mask=gt_mask)
    track_masks = {center_idx: gt_mask}
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
        if frame_idx != center_idx:
            track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=True):
        if frame_idx != center_idx:
            track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_sam2 = time.perf_counter() - t1

    t2 = time.perf_counter()
    for li in sorted(track_masks):
        frame = np.array(Image.open(tmp_dir / f"{li:05d}.jpg").convert("RGB"))
        for model, (transform, letterbox) in zip(models, transforms):
            crop = crop_from_mask(frame, track_masks[li], letterbox)
            if crop is None:
                continue
            image = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
            with torch.no_grad():
                model(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_classify = time.perf_counter() - t2

    return t_io, t_sam2, t_classify, len(track_masks)


def main() -> None:
    args = parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(args.device)
    frames_root = args.data_root / "frames-001" / "frames"

    ensemble_config = yaml.safe_load(args.ensemble_config.read_text())
    members_cfg = ensemble_config["members"]

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    class_names = ds.class_names_ordered()

    models, transforms = [], []
    for m in members_cfg:
        model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()
        models.append(model)
        transforms.append((build_transforms(m["image_size"], train=False), m["letterbox"]))

    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    total_needed = args.num_warmup + args.num_runs
    indices = list(range(min(total_needed, len(ds.instances))))

    print(f"warming up ({args.num_warmup} instances)...")
    for i in range(args.num_warmup):
        run_one_instance(predictor, models, transforms, ds, indices[i], frames_root, args.window, args.tmp_dir, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"timing {args.num_runs} real instances...")
    io_times, sam2_times, classify_times, track_lens = [], [], [], []
    for i in range(args.num_warmup, args.num_warmup + args.num_runs):
        t_io, t_sam2, t_classify, track_len = run_one_instance(
            predictor, models, transforms, ds, indices[i], frames_root, args.window, args.tmp_dir, device
        )
        io_times.append(t_io * 1000)
        sam2_times.append(t_sam2 * 1000)
        classify_times.append(t_classify * 1000)
        track_lens.append(track_len)
        print(f"  [{i - args.num_warmup}] io={t_io*1000:.1f}ms sam2={t_sam2*1000:.1f}ms classify={t_classify*1000:.1f}ms track_len={track_len}")

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    total_times = [io + s + c for io, s, c in zip(io_times, sam2_times, classify_times)]

    print("\nper-frame ensemble forward latency (dummy-tensor, standard convention):")
    per_frame_ensemble_ms = 0.0
    for m, model in zip(members_cfg, models):
        stats, _vram = benchmark_gpu_latency(model, device, m["image_size"])
        print(f"  {m['label']}: median={stats.median_ms:.3f}ms p95={stats.p95_ms:.3f}ms")
        per_frame_ensemble_ms += stats.median_ms

    result = {
        "n_runs": args.num_runs, "window": args.window,
        "frame_io_ms": {"median": float(np.median(io_times)), "p95": float(np.percentile(io_times, 95))},
        "sam2_propagation_ms": {"median": float(np.median(sam2_times)), "p95": float(np.percentile(sam2_times, 95))},
        "ensemble_classification_ms": {"median": float(np.median(classify_times)), "p95": float(np.percentile(classify_times, 95))},
        "total_per_instance_ms": {"median": float(np.median(total_times)), "p95": float(np.percentile(total_times, 95))},
        "mean_track_len": float(np.mean(track_lens)),
        "peak_vram_mb": peak_vram_mb,
        "per_frame_ensemble_forward_ms": per_frame_ensemble_ms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n=== summary (n={args.num_runs} real instances, official test) ===")
    print(f"frame I/O:              median={result['frame_io_ms']['median']:.1f}ms  p95={result['frame_io_ms']['p95']:.1f}ms")
    print(f"SAM2 propagation:       median={result['sam2_propagation_ms']['median']:.1f}ms  p95={result['sam2_propagation_ms']['p95']:.1f}ms")
    print(f"ensemble classification: median={result['ensemble_classification_ms']['median']:.1f}ms  p95={result['ensemble_classification_ms']['p95']:.1f}ms")
    print(f"TOTAL per instance:      median={result['total_per_instance_ms']['median']:.1f}ms  p95={result['total_per_instance_ms']['p95']:.1f}ms")
    print(f"mean track length: {result['mean_track_len']:.1f} frames")
    print(f"peak VRAM: {peak_vram_mb:.1f} MB")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
