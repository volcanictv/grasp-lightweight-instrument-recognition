"""Runtime benchmarking: params, FLOPs/MACs, model size, GPU latency, peak
VRAM, ONNX CPU latency. Titan Xp numbers here are not a deployment claim
(CLAUDE.md) — the ONNX CPU figure is what's portable to hardware we don't own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LatencyStats:
    median_ms: float
    p95_ms: float


def count_macs_params(model: nn.Module, image_size: int, device: torch.device) -> tuple[int, int]:
    from thop import profile

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    model_copy = model
    macs, params = profile(model_copy, inputs=(dummy,), verbose=False)
    return int(macs), int(params)


def model_file_size_bytes(checkpoint_path: Path) -> int:
    return checkpoint_path.stat().st_size


@torch.no_grad()
def benchmark_gpu_latency(
    model: nn.Module, device: torch.device, image_size: int, num_warmup: int = 50, num_runs: int = 200
) -> tuple[LatencyStats, float]:
    """Single-image latency, warmed up, torch.cuda.synchronize()'d around
    each timed call per CLAUDE.md. Returns (latency stats, peak VRAM in MB).
    """
    model.eval().to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    for _ in range(num_warmup):
        model(dummy)
    torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    )
    return LatencyStats(median_ms=float(np.median(times_ms)), p95_ms=float(np.percentile(times_ms, 95))), peak_vram_mb


@torch.no_grad()
def benchmark_batch_throughput(
    model: nn.Module, device: torch.device, image_size: int, batch_size: int, num_batches: int = 20
) -> float:
    model.eval().to(device)
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    for _ in range(5):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(num_batches):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return (batch_size * num_batches) / elapsed


@torch.no_grad()
def benchmark_detection_gpu_latency(
    model: nn.Module, device: torch.device, height: int, width: int, num_warmup: int = 50, num_runs: int = 200
) -> tuple[LatencyStats, float]:
    """Single-image latency for a torchvision detection model, which takes
    a *list* of tensors in eval mode (not a batched tensor like a
    classifier) -- otherwise identical to `benchmark_gpu_latency`.
    `height`/`width` default to GraSP's native 800x1280 (CLAUDE.md) rather
    than a square guess, since the detection dataset never resizes frames
    (torchvision's internal GeneralizedRCNNTransform does its own resizing
    from whatever is passed in).

    FLOPs/MACs (via thop) and ONNX export are deliberately not attempted
    here: both are unreliable for two-stage detectors with a
    dynamic-length RPN/ROI-heads pipeline (proposal count varies per
    image), and a wrong or misleading number would be worse than an
    honestly-missing one, per CLAUDE.md's standard against overclaiming.
    """
    model.eval().to(device)
    dummy = [torch.randn(3, height, width, device=device)]

    for _ in range(num_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times_ms = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    )
    return LatencyStats(median_ms=float(np.median(times_ms)), p95_ms=float(np.percentile(times_ms, 95))), peak_vram_mb


def export_onnx(model: nn.Module, image_size: int, out_path: Path) -> None:
    model.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, str(out_path), input_names=["image"], output_names=["logits"],
        dynamo=False,
    )


def benchmark_onnx_cpu_latency(
    onnx_path: Path, image_size: int, num_warmup: int = 20, num_runs: int = 200
) -> LatencyStats:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    times_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times_ms.append((time.perf_counter() - start) * 1000)

    return LatencyStats(median_ms=float(np.median(times_ms)), p95_ms=float(np.percentile(times_ms, 95)))
