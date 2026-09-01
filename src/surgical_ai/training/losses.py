"""Loss construction from config. `class_weights: true` is a Milestone 5
ablation variable for Task A (`bce`) and the default imbalance handling for
Task B (`cross_entropy`, Milestone 7) — baseline configs leave it false.
Kept here (not decided per-run in the trainer) so weighting strategy stays a
one-line config change.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_pos_weight(label_counts: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Inverse-frequency positive class weight for BCEWithLogitsLoss:
    (negatives / positives) per class, so rare classes get a bigger loss
    contribution per positive example.
    """
    positives = label_counts.clamp(min=1)
    negatives = num_samples - label_counts
    return negatives / positives


def compute_class_weights(label_counts: torch.Tensor) -> torch.Tensor:
    """Balanced per-class weight for CrossEntropyLoss (Task B):
    n_samples / (n_classes * count_c), so weights average to ~1 rather than
    blowing up the loss scale the way raw inverse frequency would.
    """
    total = label_counts.sum()
    n_classes = label_counts.numel()
    return total / (n_classes * label_counts.clamp(min=1))


def build_loss(
    loss_config: dict,
    pos_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
) -> nn.Module:
    loss_type = loss_config.get("type", "bce")
    use_weights = loss_config.get("class_weights", False)

    if loss_type == "bce":
        if use_weights and pos_weight is None:
            raise ValueError("loss.class_weights is true but no pos_weight was computed")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight if use_weights else None)

    if loss_type == "cross_entropy":
        if use_weights and class_weight is None:
            raise ValueError("loss.class_weights is true but no class_weight was computed")
        return nn.CrossEntropyLoss(weight=class_weight if use_weights else None)

    raise ValueError(f"unknown loss type '{loss_type}'")
