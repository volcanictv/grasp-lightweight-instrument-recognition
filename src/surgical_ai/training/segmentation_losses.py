"""Multi-head loss for the centroid/offset segmenter (Milestone 9).

Heatmap loss is the penalty-reduced pixelwise focal loss from CornerNet
(Law & Deng 2018) / CenterNet (Zhou et al. 2019, "Objects as Points") --
standard for this family of anchor-free detectors, not derived here.
Offset loss is masked L1 (only supervised at pixels inside some instance's
mask). Semantic loss is plain cross-entropy.

`offset_weight=0.1` follows CenterNet's convention of down-weighting the
regression head relative to the heatmap head -- our offset targets are in
raw stride-pixel units (up to ~48 at stride 4 for a 384 input) rather than
CenterNet's fractional sub-pixel offsets, so the L1 magnitude is naturally
larger and would otherwise dominate the combined loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def centroid_focal_loss(pred_logits: torch.Tensor, target_heatmap: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits).clamp(min=1e-4, max=1 - 1e-4)
    pos_mask = target_heatmap.eq(1).float()
    neg_mask = target_heatmap.lt(1).float()

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * torch.pow(1 - target_heatmap, beta) * neg_mask

    num_pos = pos_mask.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def offset_l1_loss(pred_offset: torch.Tensor, target_offset: torch.Tensor, offset_mask: torch.Tensor) -> torch.Tensor:
    mask = offset_mask.unsqueeze(1).expand_as(pred_offset).float()
    num_valid = mask.sum().clamp(min=1)
    return (torch.abs(pred_offset - target_offset) * mask).sum() / num_valid


def compute_segmentation_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    offset_weight: float = 0.1,
    semantic_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    heatmap_loss = centroid_focal_loss(predictions["heatmap"], targets["heatmap"])
    offset_loss = offset_l1_loss(predictions["offset"], targets["offset"], targets["offset_mask"])
    semantic_loss = F.cross_entropy(predictions["semantic"], targets["semantic"])

    total = heatmap_loss + offset_weight * offset_loss + semantic_weight * semantic_loss
    return {
        "total": total,
        "heatmap": heatmap_loss.detach(),
        "offset": offset_loss.detach(),
        "semantic": semantic_loss.detach(),
    }
