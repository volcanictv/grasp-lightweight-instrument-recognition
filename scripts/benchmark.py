"""Benchmarking entry point.

`loader` mode (Milestone 1) measures pure dataloader throughput (images/sec)
with no model involved, per CLAUDE.md's "dataloader bound, not GPU bound"
warning.

`model` mode (Milestone 3, classification tasks only) measures params,
FLOPs/MACs, model file size, Titan Xp GPU latency (median/p95 over warm
runs) and peak VRAM, batch throughput, and ONNX Runtime CPU latency — the
number that's portable to hardware we don't own. Titan Xp numbers are
never a deployment claim on their own (CLAUDE.md).

`detection` mode (Milestone 8) measures params, model file size, and
Titan Xp GPU latency for a torchvision detection model (list-of-tensors
calling convention, not the classifiers' batched-tensor one). FLOPs and
ONNX export are deliberately not attempted for detection -- both are
unreliable for two-stage detectors with a dynamic-length RPN/ROI-heads
pipeline; see `evaluation/benchmarking.py`.

Usage:
    python scripts/benchmark.py loader [--data-root PATH] [--split train]
        [--image-size 224] [--batch-size 32] [--num-batches 20]
        [--workers 0,2,4,8]
    python scripts/benchmark.py model configs/baseline_frozen.yaml
        experiments/<run_id>/best.pt [--device cuda:0]
    python scripts/benchmark.py detection configs/detection_baseline.yaml
        experiments/<run_id>/best.pt [--device cuda:0] [--height 800] [--width 1280]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from surgical_ai.data.dataset import GraspMultiLabelDataset  # noqa: E402
from surgical_ai.data.transforms import build_transforms  # noqa: E402
from surgical_ai.evaluation import benchmarking  # noqa: E402
from surgical_ai.models import build_model  # noqa: E402
from surgical_ai.models.detectors.registry import build_detector  # noqa: E402
from surgical_ai.training.trainer import count_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    loader_p = sub.add_parser("loader", help="Pure dataloader throughput benchmark.")
    loader_p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")),
    )
    loader_p.add_argument("--split", default="train", choices=["train", "test", "fold1", "fold2"])
    loader_p.add_argument("--image-size", type=int, default=224)
    loader_p.add_argument("--batch-size", type=int, default=32)
    loader_p.add_argument(
        "--num-batches", type=int, default=20, help="Timed batches per worker config, after warmup."
    )
    loader_p.add_argument("--warmup-batches", type=int, default=3)
    loader_p.add_argument(
        "--workers", default="0,2,4,8", help="Comma-separated num_workers values to try."
    )

    model_p = sub.add_parser("model", help="Params/FLOPs/latency/VRAM/ONNX benchmark.")
    model_p.add_argument("config", type=Path)
    model_p.add_argument("checkpoint", type=Path)
    model_p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    model_p.add_argument("--num-warmup", type=int, default=50)
    model_p.add_argument("--num-runs", type=int, default=200)
    model_p.add_argument("--batch-size", type=int, default=None, help="Defaults to config's training.batch_size.")
    model_p.add_argument("--out", type=Path, default=None, help="Defaults to <checkpoint dir>/benchmark.json.")

    detection_p = sub.add_parser(
        "detection", help="Params/GPU-latency benchmark for a detection model (Milestone 8)."
    )
    detection_p.add_argument("config", type=Path)
    detection_p.add_argument("checkpoint", type=Path)
    detection_p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    detection_p.add_argument(
        "--height", type=int, default=800, help="GraSP native frame height (CLAUDE.md: 800x1280)."
    )
    detection_p.add_argument("--width", type=int, default=1280)
    detection_p.add_argument("--num-warmup", type=int, default=50)
    detection_p.add_argument("--num-runs", type=int, default=200)
    detection_p.add_argument("--out", type=Path, default=None, help="Defaults to <checkpoint dir>/benchmark.json.")

    return parser.parse_args()


def time_dataloader(loader: DataLoader, warmup_batches: int, num_batches: int) -> float:
    it = iter(loader)
    for _ in range(warmup_batches):
        next(it)

    n_images = 0
    start = time.perf_counter()
    for _ in range(num_batches):
        images, _ = next(it)
        n_images += images.shape[0]
    elapsed = time.perf_counter() - start
    return n_images / elapsed


def run_loader_benchmark(args: argparse.Namespace) -> None:
    transform = build_transforms(args.image_size, train=(args.split == "train"))
    dataset = GraspMultiLabelDataset(args.data_root, args.split, transform=transform)
    print(f"dataset: {len(dataset)} samples, split={args.split}, image_size={args.image_size}")

    total_needed = args.batch_size * (args.warmup_batches + args.num_batches)
    if total_needed > len(dataset):
        print(
            f"warning: need {total_needed} samples per config but dataset has "
            f"{len(dataset)}; DataLoader will wrap via a fresh iterator per config, "
            "not a real epoch boundary. Fine for a throughput measurement."
        )

    print(f"{'num_workers':>12}{'images/sec':>14}")
    for workers in [int(w) for w in args.workers.split(",")]:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            persistent_workers=False,
        )
        try:
            throughput = time_dataloader(loader, args.warmup_batches, args.num_batches)
            print(f"{workers:>12}{throughput:>14.1f}")
        finally:
            del loader


def run_model_benchmark(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    image_size = config["data"]["image_size"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    device = torch.device(args.device)

    # 7 is the GraSP instrument class count (see splits/categories); not
    # read from the dataset here since this script only needs the model,
    # not the data pipeline.
    model = build_model(
        config["model"]["name"], num_classes=7,
        pretrained=False, freeze_backbone=config["model"]["freeze_backbone"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    trainable, total = count_parameters(model)

    macs, thop_params = benchmarking.count_macs_params(model, image_size, torch.device("cpu"))
    size_bytes = benchmarking.model_file_size_bytes(args.checkpoint)

    results = {
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "image_size": image_size,
        "trainable_params": trainable,
        "total_params": total,
        "macs": macs,
        "model_file_size_bytes": size_bytes,
        "model_file_size_mb": size_bytes / (1024 * 1024),
    }

    if device.type == "cuda":
        gpu_latency, peak_vram_mb = benchmarking.benchmark_gpu_latency(
            model, device, image_size, args.num_warmup, args.num_runs
        )
        throughput = benchmarking.benchmark_batch_throughput(model, device, image_size, batch_size)
        results.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_latency_median_ms": gpu_latency.median_ms,
            "gpu_latency_p95_ms": gpu_latency.p95_ms,
            "gpu_peak_vram_mb": peak_vram_mb,
            "gpu_batch_throughput_img_s": throughput,
            "gpu_batch_size": batch_size,
        })
    else:
        print("device is CPU, skipping GPU latency/VRAM section")

    onnx_path = args.checkpoint.with_suffix(".onnx")
    benchmarking.export_onnx(model, image_size, onnx_path)
    onnx_latency = benchmarking.benchmark_onnx_cpu_latency(
        onnx_path, image_size, num_warmup=20, num_runs=args.num_runs
    )
    results.update({
        "onnx_path": str(onnx_path),
        "onnx_cpu_latency_median_ms": onnx_latency.median_ms,
        "onnx_cpu_latency_p95_ms": onnx_latency.p95_ms,
    })

    out_path = args.out or (args.checkpoint.parent / "benchmark.json")
    out_path.write_text(json.dumps(results, indent=2))

    print(f"params: {trainable}/{total} trainable/total, {macs/1e6:.1f}M MACs")
    print(f"model file size: {results['model_file_size_mb']:.2f} MB")
    if device.type == "cuda":
        print(
            f"GPU ({results['gpu_name']}) latency: median={results['gpu_latency_median_ms']:.2f}ms "
            f"p95={results['gpu_latency_p95_ms']:.2f}ms, peak VRAM={peak_vram_mb:.1f}MB, "
            f"batch throughput={throughput:.1f} img/s @ bs={batch_size}"
        )
    print(
        f"ONNX CPU latency: median={onnx_latency.median_ms:.2f}ms p95={onnx_latency.p95_ms:.2f}ms"
    )
    print(f"results written to {out_path}")


def run_detection_benchmark(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)

    # 7 is the GraSP instrument class count -- same reasoning as
    # run_model_benchmark: only the model is needed here, not the data
    # pipeline that would otherwise supply this.
    model = build_detector(config["model"]["name"], num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    trainable, total = count_parameters(model)
    size_bytes = benchmarking.model_file_size_bytes(args.checkpoint)

    results = {
        "config_path": str(args.config),
        "checkpoint": str(args.checkpoint),
        "height": args.height,
        "width": args.width,
        "trainable_params": trainable,
        "total_params": total,
        "model_file_size_bytes": size_bytes,
        "model_file_size_mb": size_bytes / (1024 * 1024),
        "flops_note": "not measured -- thop/ONNX are unreliable for two-stage detectors, see evaluation/benchmarking.py",
    }

    if device.type == "cuda":
        gpu_latency, peak_vram_mb = benchmarking.benchmark_detection_gpu_latency(
            model, device, args.height, args.width, args.num_warmup, args.num_runs
        )
        results.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_latency_median_ms": gpu_latency.median_ms,
            "gpu_latency_p95_ms": gpu_latency.p95_ms,
            "gpu_peak_vram_mb": peak_vram_mb,
        })
        print(
            f"params: {trainable}/{total} trainable/total | "
            f"GPU ({results['gpu_name']}) latency: median={gpu_latency.median_ms:.2f}ms "
            f"p95={gpu_latency.p95_ms:.2f}ms, peak VRAM={peak_vram_mb:.1f}MB "
            f"@ {args.height}x{args.width}"
        )
    else:
        print("device is CPU, skipping GPU latency section")

    out_path = args.out or (args.checkpoint.parent / "benchmark.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"results written to {out_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "loader":
        run_loader_benchmark(args)
    elif args.mode == "model":
        run_model_benchmark(args)
    elif args.mode == "detection":
        run_detection_benchmark(args)


if __name__ == "__main__":
    main()
