"""Sampler construction from config. `data.sampling: weighted` is a
Milestone 5 ablation variable (PROJECT_SPEC.md Sec.7 "balanced sampler") —
baseline configs leave it "none".
"""

from __future__ import annotations

import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def compute_sample_weights(labels: torch.Tensor, label_counts: torch.Tensor) -> torch.Tensor:
    """Per-sample weight for multi-label data: the max inverse-class-frequency
    among a sample's active classes, so a frame containing any rare class
    (even alongside common ones) gets boosted, not averaged away.
    """
    inv_freq = 1.0 / label_counts.clamp(min=1)
    return (labels * inv_freq).max(dim=1).values


def build_sampler(mode: str, samples: list[tuple[str, torch.Tensor]]) -> Sampler | None:
    if mode == "none":
        return None
    if mode != "weighted":
        raise ValueError(f"unknown data.sampling '{mode}'. Valid: none, weighted")

    labels = torch.stack([label for _, label in samples])
    label_counts = labels.sum(dim=0)
    weights = compute_sample_weights(labels, label_counts)
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
