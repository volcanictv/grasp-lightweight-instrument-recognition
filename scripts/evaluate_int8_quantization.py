"""Milestone 10 (efficiency, deferred until accuracy/generalizability were
in a good place -- see CLAUDE.md). Tests INT8 dynamic quantization (ONNX
Runtime, weights-only dynamic quantization -- no calibration data needed,
the safer of the two standard PTQ modes) against each Task B ensemble
member: does it actually shrink the model and speed up CPU inference
without a real accuracy cost.

Unlike scripts/benchmark.py's existing ONNX path (dummy input, latency
only), this runs the real official-test set through both the FP32 and
INT8 ONNX exports and reports actual classification accuracy/macro-F1 for
each, plus model size and CPU latency, so "no accuracy loss" is checked,
not assumed.

Usage:
    python scripts/evaluate_int8_quantization.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import f1_score

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.benchmarking import export_onnx, model_file_size_bytes
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "onnx_export")
    parser.add_argument("--num-latency-runs", type=int, default=200)
    return parser.parse_args()


def run_onnx_on_dataset(onnx_path: Path, tensors: np.ndarray) -> np.ndarray:
    # The exported graph is fixed at batch=1 (matches scripts/benchmark.py's
    # existing single-image latency convention) -- run one sample at a time.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    all_probs = []
    for i in range(len(tensors)):
        logits = session.run(None, {input_name: tensors[i : i + 1]})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        all_probs.append(probs)
    return np.concatenate(all_probs)


def benchmark_latency(onnx_path: Path, image_size: int, num_runs: int) -> tuple[float, float]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    for _ in range(20):
        session.run(None, {input_name: dummy})
    times_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times_ms.append((time.perf_counter() - start) * 1000)
    return float(np.median(times_ms)), float(np.percentile(times_ms, 95))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_config = yaml.safe_load(args.config.read_text())
    members = ensemble_config["members"]

    print(f"{'model':<16} {'metric':<22} {'fp32':>10} {'int8':>10} {'delta':>10}")
    print("-" * 72)

    for member in members:
        ds = GraspRegionDataset(
            args.data_root, args.split, transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        class_names = ds.class_names_ordered()
        y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        tensors = np.concatenate([images.numpy() for images, _labels in loader]).astype(np.float32)

        model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False)
        model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location="cpu"), strict=False)

        fp32_path = args.out_dir / f"{member['label']}_fp32.onnx"
        int8_path = args.out_dir / f"{member['label']}_int8.onnx"
        export_onnx(model, member["image_size"], fp32_path)
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

        probs_fp32 = run_onnx_on_dataset(fp32_path, tensors)
        probs_int8 = run_onnx_on_dataset(int8_path, tensors)
        acc_fp32 = (probs_fp32.argmax(axis=1) == y_true).mean()
        acc_int8 = (probs_int8.argmax(axis=1) == y_true).mean()
        f1_fp32 = f1_score(y_true, probs_fp32.argmax(axis=1), average="macro", zero_division=0)
        f1_int8 = f1_score(y_true, probs_int8.argmax(axis=1), average="macro", zero_division=0)

        lat_fp32, _ = benchmark_latency(fp32_path, member["image_size"], args.num_latency_runs)
        lat_int8, _ = benchmark_latency(int8_path, member["image_size"], args.num_latency_runs)

        size_fp32 = model_file_size_bytes(fp32_path) / 1e6
        size_int8 = model_file_size_bytes(int8_path) / 1e6

        print(f"{member['label']:<16} {'accuracy':<22} {acc_fp32:>10.4f} {acc_int8:>10.4f} {acc_int8-acc_fp32:>+10.4f}")
        print(f"{member['label']:<16} {'macro-F1':<22} {f1_fp32:>10.4f} {f1_int8:>10.4f} {f1_int8-f1_fp32:>+10.4f}")
        print(f"{member['label']:<16} {'ONNX CPU latency (ms)':<22} {lat_fp32:>10.2f} {lat_int8:>10.2f} {lat_int8-lat_fp32:>+10.2f}")
        print(f"{member['label']:<16} {'model size (MB)':<22} {size_fp32:>10.2f} {size_int8:>10.2f} {size_int8-size_fp32:>+10.2f}")
        print()


if __name__ == "__main__":
    main()
