"""Per-instance inference latency of the vote-based ensemble (4 members x 20
stochastic passes, docs/DECISIONS.md 2026-09-20), next to a single deterministic
pass of the same 4 members for reference.

Follows the project convention (CLAUDE.md): warm up first, torch.cuda.synchronize()
around each timed call, median and p95 over warm runs, one instance at a time.
Model forward only (no image decoding/cropping), random input of each member's
size, since latency does not depend on pixel values. The 20 passes of a member are
one batched forward of 20 copies of the crop with dropout kept on, so every copy
gets its own dropout mask; that is also how the votes were produced.

The earlier CPU figure used ONNX Runtime, which exports the inference graph with
dropout removed and so cannot express MC Dropout; this reports PyTorch CPU instead.

Usage:
    python scripts/benchmark_vote_ensemble_latency.py --device cuda:0 --runs 200
    CUDA_VISIBLE_DEVICES= python scripts/benchmark_vote_ensemble_latency.py --device cpu --runs 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml

from analyze_uncertainty_signals import enable_mc_dropout
from surgical_ai.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(args.ensemble_config.read_text())
    models, sizes = [], []
    for m in cfg["members"]:
        model = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        models.append(model)
        sizes.append(m["image_size"])
    inputs = [torch.randn(1, 3, s, s, device=device) for s in sizes]
    params = sum(p.numel() for model in models for p in model.parameters())

    def run(mc: bool) -> None:
        with torch.no_grad():
            for model, x in zip(models, inputs):
                if mc:
                    enable_mc_dropout(model)
                    model(x.repeat(args.mc_samples, 1, 1, 1))
                else:
                    model.eval()
                    model(x)

    def timed(mc: bool) -> dict:
        for _ in range(args.warmup):
            run(mc)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            run(mc)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)
        out = {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "runs": args.runs}
        if device.type == "cuda":
            out["peak_vram_mb"] = float(torch.cuda.max_memory_allocated(device) / 2**20)
        return out

    result = {"device": str(device), "torch_threads": torch.get_num_threads() if device.type == "cpu" else None,
              "members": [m["label"] for m in cfg["members"]], "total_params": int(params), "mc_samples": args.mc_samples,
              "single_pass_4_members": timed(False), "mc_votes_4_members_x_passes": timed(True)}
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
