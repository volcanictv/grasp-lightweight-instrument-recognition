"""Per-frame inference latency for the Task B ensemble
(region_baseline + region_letterbox_crop, docs/error_analysis.md/
DECISIONS.md 2026-09-02, macro-F1 0.848). An accuracy win reported without
its latency cost is incomplete per this project's own rule (see the
segmentation ensemble's benchmark_ensemble_latency.py for the same
reasoning) -- this ensemble hadn't been measured yet.

Follows this project's standard convention (CLAUDE.md,
src/surgical_ai/evaluation/benchmarking.py): warm up before timing,
torch.cuda.synchronize() around each timed call, median/p95 over >=200
warm runs, single image (batch size 1), Titan Xp plus the portable ONNX
CPU number.

Usage:
    python scripts/benchmark_region_ensemble.py \\
        --checkpoint-a experiments/region_baseline_20260831-182451/best.pt \\
        --checkpoint-b experiments/region_letterbox_crop_20260902-152750/best.pt \\
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

from surgical_ai.evaluation.benchmarking import (  # noqa: E402
    benchmark_gpu_latency, benchmark_onnx_cpu_latency, count_macs_params,
    export_onnx, model_file_size_bytes,
)
from surgical_ai.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--model-a", default="mobilenet_v3_small")
    parser.add_argument("--model-b", default="mobilenet_v3_small")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--onnx-dir", type=Path, default=Path("/tmp"))
    return parser.parse_args()


@torch.no_grad()
def benchmark_combined_gpu_latency(
    model_a: torch.nn.Module, model_b: torch.nn.Module, device: torch.device, image_size: int,
    num_warmup: int = 50, num_runs: int = 200,
) -> tuple[float, float, float]:
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    for _ in range(num_warmup):
        model_a(dummy)
        model_b(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model_a(dummy)
        model_b(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)

    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    )
    return float(np.median(times_ms)), float(np.percentile(times_ms, 95)), peak_vram


def load_model(checkpoint: Path, model_name: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_model(model_name, num_classes=num_classes, pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model_a = load_model(args.checkpoint_a, args.model_a, args.num_classes, device)
    model_b = load_model(args.checkpoint_b, args.model_b, args.num_classes, device)

    macs_a, params_a = count_macs_params(model_a, args.image_size, device)
    macs_b, params_b = count_macs_params(model_b, args.image_size, device)
    size_a = model_file_size_bytes(args.checkpoint_a)
    size_b = model_file_size_bytes(args.checkpoint_b)
    print(f"model A ({args.model_a}): params={params_a} macs={macs_a} size={size_a/1024/1024:.2f}MB")
    print(f"model B ({args.model_b}): params={params_b} macs={macs_b} size={size_b/1024/1024:.2f}MB")
    print(f"combined (2 models): params={params_a+params_b} size={(size_a+size_b)/1024/1024:.2f}MB\n")

    stats_a, vram_a = benchmark_gpu_latency(model_a, device, args.image_size)
    print(f"Model A alone: median={stats_a.median_ms:.3f}ms p95={stats_a.p95_ms:.3f}ms peak_vram={vram_a:.1f}MB")

    stats_b, vram_b = benchmark_gpu_latency(model_b, device, args.image_size)
    print(f"Model B alone: median={stats_b.median_ms:.3f}ms p95={stats_b.p95_ms:.3f}ms peak_vram={vram_b:.1f}MB")

    median_c, p95_c, vram_c = benchmark_combined_gpu_latency(model_a, model_b, device, args.image_size)
    print(f"\nCombined ensemble (sequential, single frame): median={median_c:.3f}ms p95={p95_c:.3f}ms peak_vram={vram_c:.1f}MB")
    print(f"Naive sum of individual medians: {stats_a.median_ms + stats_b.median_ms:.3f}ms")

    if device.type == "cuda":
        args.onnx_dir.mkdir(parents=True, exist_ok=True)
        export_onnx(model_a, args.image_size, args.onnx_dir / "region_a.onnx")
        export_onnx(model_b, args.image_size, args.onnx_dir / "region_b.onnx")
        onnx_a = benchmark_onnx_cpu_latency(args.onnx_dir / "region_a.onnx", args.image_size)
        onnx_b = benchmark_onnx_cpu_latency(args.onnx_dir / "region_b.onnx", args.image_size)
        print(f"\nONNX CPU -- model A: median={onnx_a.median_ms:.3f}ms p95={onnx_a.p95_ms:.3f}ms")
        print(f"ONNX CPU -- model B: median={onnx_b.median_ms:.3f}ms p95={onnx_b.p95_ms:.3f}ms")
        print(f"ONNX CPU -- combined (sum): {onnx_a.median_ms + onnx_b.median_ms:.3f}ms")


if __name__ == "__main__":
    main()
