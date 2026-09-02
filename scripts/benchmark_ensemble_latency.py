"""Per-frame inference latency for the accuracy-first pipeline built
2026-09-01/02 (docs/DECISIONS.md): three Mask R-CNN checkpoints plus a
box detector + fine-tuned SAM2 decoder, combined via weighted box+mask
fusion. Accuracy-first priority explicitly defers efficiency to last
(CLAUDE.md), but the actual cost of the current best result (4-way
ensemble, AP50_segm 0.8594) should be known, not assumed.

Follows this project's existing latency convention exactly (CLAUDE.md,
src/surgical_ai/evaluation/benchmarking.py): warm up before timing,
torch.cuda.synchronize() around each timed call, report median and p95
over >=200 warm runs, single image at a time (batch size 1). Mask R-CNN
timings reuse benchmark_detection_gpu_latency unchanged (same
torchvision detection-model interface as Milestone 8's own detectors);
SAM2 has a different interface (image encoder + box-prompted decoder)
and is timed separately, then combined arithmetically -- SAM2's per-box
decode cost is genuinely per-instance, unlike a Mask R-CNN's single
forward pass, so a flat number would misrepresent it.

Usage:
    python scripts/benchmark_ensemble_latency.py \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.evaluation.benchmarking import LatencyStats, benchmark_detection_gpu_latency  # noqa: E402
from surgical_ai.models.detectors.registry import build_detector  # noqa: E402

# GraSP official test: 2861 instances / 1125 keyframes (docs/dataset_report.md).
AVG_INSTANCES_PER_FRAME = 2861 / 1125


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--num-runs", type=int, default=200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1280)
    return parser.parse_args()


@torch.no_grad()
def benchmark_sam2_encoder(predictor, height: int, width: int, num_warmup: int, num_runs: int, device: torch.device):
    dummy = (np.random.rand(height, width, 3) * 255).astype(np.uint8)

    for _ in range(num_warmup):
        predictor.set_image(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        predictor.set_image(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    return LatencyStats(median_ms=float(np.median(times_ms)), p95_ms=float(np.percentile(times_ms, 95))), peak_vram_mb


@torch.no_grad()
def _sam2_box_decode(predictor, box: np.ndarray):
    """Box-prompted mask decode, inference-only (no gradient tracking needed
    for benchmarking) -- the public predict() detaches to numpy but that
    cost is real deployment overhead too, so it's included here on purpose,
    unlike finetune_sam2_decoder.py's training-time differentiable_predict.
    """
    return predictor.predict(box=box, multimask_output=False)


def benchmark_sam2_decode(predictor, height: int, width: int, num_warmup: int, num_runs: int, device: torch.device) -> LatencyStats:
    dummy = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    predictor.set_image(dummy)  # one encoding, reused for every timed decode call below
    box = np.array([width * 0.3, height * 0.3, width * 0.6, height * 0.6])

    for _ in range(num_warmup):
        _sam2_box_decode(predictor, box)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _sam2_box_decode(predictor, box)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)

    return LatencyStats(median_ms=float(np.median(times_ms)), p95_ms=float(np.percentile(times_ms, 95)))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    num_classes = 7

    print(f"GraSP average instances/frame (official test): {AVG_INSTANCES_PER_FRAME:.2f}\n")

    results = {}
    mask_rcnn_configs = [
        ("maskrcnn_mobilenet_v3 (MobileNetV3, official/from-scratch share this arch.)", "maskrcnn_mobilenet_v3"),
        ("maskrcnn_resnet50_coco (COCO-pretrained ResNet-50)", "maskrcnn_resnet50_coco"),
        ("fasterrcnn_mobilenet_v3 (box detector, also SAM2's prompt source)", "fasterrcnn_mobilenet_v3"),
    ]
    for label, registry_name in mask_rcnn_configs:
        model = build_detector(registry_name, num_classes=num_classes, pretrained=False).to(device)
        stats, peak_vram = benchmark_detection_gpu_latency(
            model, device, args.height, args.width, args.num_warmup, args.num_runs
        )
        results[label] = stats
        print(f"[{label}] median={stats.median_ms:.2f}ms p95={stats.p95_ms:.2f}ms peak_vram={peak_vram:.1f}MB")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=str(device)))
    encoder_stats, encoder_vram = benchmark_sam2_encoder(predictor, args.height, args.width, args.num_warmup, args.num_runs, device)
    print(f"[SAM2 image encoder, once/frame] median={encoder_stats.median_ms:.2f}ms p95={encoder_stats.p95_ms:.2f}ms peak_vram={encoder_vram:.1f}MB")

    decode_stats = benchmark_sam2_decode(predictor, args.height, args.width, args.num_warmup, args.num_runs, device)
    print(f"[SAM2 mask decode, once/instance] median={decode_stats.median_ms:.2f}ms p95={decode_stats.p95_ms:.2f}ms")

    sam2_per_frame_median = encoder_stats.median_ms + decode_stats.median_ms * AVG_INSTANCES_PER_FRAME
    print(f"[SAM2 pipeline, per frame @ {AVG_INSTANCES_PER_FRAME:.2f} instances] median={sam2_per_frame_median:.2f}ms (encoder once + decode x avg instances)")

    print("\n--- 4-way ensemble estimate (sum of components; fusion itself is pure-Python, <1ms, not separately measured) ---")
    box_detector_label = "fasterrcnn_mobilenet_v3 (box detector, also SAM2's prompt source)"
    mrcnn1_label = "maskrcnn_mobilenet_v3 (MobileNetV3, official/from-scratch share this arch.)"
    mrcnn2_label = "maskrcnn_resnet50_coco (COCO-pretrained ResNet-50)"
    total_median = (
        results[mrcnn1_label].median_ms * 2  # official + from-scratch, same architecture, run twice
        + results[mrcnn2_label].median_ms
        + results[box_detector_label].median_ms
        + sam2_per_frame_median
    )
    print(f"4 model forward passes + SAM2 pipeline, single frame, single GPU: ~{total_median:.1f}ms median (~{1000/total_median:.1f} fps sequential)")
    print("Note: the two same-architecture MobileNetV3 Mask R-CNN passes (official + from-scratch) could run as one batch-of-2 "
          "forward pass instead of two sequential ones, and all 4 models could run in parallel across GPUs/processes -- "
          "this total is a naive sequential upper bound, not an optimized deployment number.")


if __name__ == "__main__":
    main()
