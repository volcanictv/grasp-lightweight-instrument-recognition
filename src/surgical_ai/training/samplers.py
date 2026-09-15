"""Sampler construction from config. `data.sampling: weighted` is a
Milestone 5 ablation variable (PROJECT_SPEC.md Sec.7 "balanced sampler") —
baseline configs leave it "none".
"""

from __future__ import annotations

import numpy as np
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


def compute_area_oversample_weights(
    area_fracs: np.ndarray, threshold: float, boost: float
) -> np.ndarray:
    """Binary weight keyed on instance mask area (not class frequency): every
    instance below `threshold` (same area-fraction definition used by the
    class-conditional abstention gate, `docs/DECISIONS.md` 2026-09-04) gets
    `boost`x the sampling weight of everything else. Distinct axis from
    `compute_sample_weights` above -- this is Task B's per-instance area, not
    Task A's per-frame class rarity -- and deliberately uniform across
    classes so it stays a one-variable change against the class-conditional
    area-gate finding rather than compounding two hypotheses in one run.
    """
    weights = np.ones_like(area_fracs, dtype=np.float64)
    weights[area_fracs < threshold] = boost
    return weights


def build_region_area_sampler(
    area_fracs: np.ndarray, threshold: float, boost: float
) -> WeightedRandomSampler:
    weights = compute_area_oversample_weights(area_fracs, threshold, boost)
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True
    )


def compute_class_conditional_area_oversample_weights(
    area_fracs: np.ndarray, target_class_mask: np.ndarray, threshold: float, boost: float
) -> np.ndarray:
    """Same area-threshold rule as `compute_area_oversample_weights`, but the
    boost is restricted to instances whose class is in `target_class_mask`.
    `docs/DECISIONS.md` 2026-09-05's uniform version helped Bipolar Forceps
    and Prograsp Forceps on the small-area subset but consistently cost
    Large Needle Driver and Monopolar Curved Scissors on the same subset --
    this restricts the boost to the classes it actually helped, leaving
    everything else (including other small-area instances) at normal
    sampling weight, as the untried follow-up flagged there.
    """
    weights = np.ones_like(area_fracs, dtype=np.float64)
    weights[(area_fracs < threshold) & target_class_mask] = boost
    return weights


def build_region_area_sampler_classcond(
    area_fracs: np.ndarray, target_class_mask: np.ndarray, threshold: float, boost: float
) -> WeightedRandomSampler:
    weights = compute_class_conditional_area_oversample_weights(
        area_fracs, target_class_mask, threshold, boost
    )
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True
    )
