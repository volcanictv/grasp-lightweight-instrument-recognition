"""Centroid/offset instance segmenter (Milestone 9). See
`data/segmentation_targets.py` for the full architecture rationale and
citations -- summary: a semantic head plus a centroid-heatmap +
offset-regression head, anchor-free and NMS-free, to remove the box-NMS
suppression of overlapping instruments diagnosed as Milestone 8's occlusion
recall gap (docs/DECISIONS.md).

Reuses the same MobileNetV3-Large-FPN backbone utility
`fasterrcnn_mobilenet_v3_large_fpn` uses internally
(`models/detectors/faster_rcnn.py`) -- same lightweight-backbone family as
the rest of this project.

Two registered variants, both built from the same class, differing only in
`returned_layers` (which of MobileNetV3-Large's stages feed the FPN):

  - `centroid_offset_mobilenet_v3` (`returned_layers=[1, 2]`, the original):
    picks two *shallow* stages (stride 4 and 8, 24 and 40 channels) purely
    to get a fine output resolution cheaply. This turned out to be a real
    capacity bug, not a deliberate design choice: `mobilenet_backbone`
    prunes the network to stop at the deepest *requested* stage, so
    requesting only shallow stages means the network's entire deep half
    (up to 960 channels, where most of its actual semantic understanding
    lives) is never computed at all -- not frozen, not present. Four
    training runs on this variant plateaued at the same mIoU/occlusion
    numbers (docs/DECISIONS.md, 2026-09-01) regardless of training recipe,
    consistent with a capacity ceiling from this, not a training problem.
    Kept registered, unchanged, so those runs stay reproducible.
  - `centroid_offset_mobilenet_v3_deep` (`returned_layers=[1, 3, 5]`):
    spans shallow-through-deep (stride 4/16/32, 24/80/960 channels), the
    way FPN is actually meant to be used -- real semantic depth gets
    injected into the fine-resolution output via FPN's top-down pathway,
    instead of the fine output only ever seeing shallow, low-capacity
    features. Backbone grows from ~1.9M to ~5.0M params; full model from
    2.1M to ~5.4M -- still far lighter than the 18.9M-param detector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection.backbone_utils import mobilenet_backbone

from surgical_ai.models.segmenters.registry import register_segmenter

FPN_OUT_CHANNELS = 256


def _conv_head(in_channels: int, out_channels: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, out_channels, kernel_size=1),
    )


class CentroidOffsetSegmenter(nn.Module):
    OUTPUT_STRIDE = 4

    def __init__(self, num_classes: int, pretrained: bool = True, returned_layers: list[int] | None = None):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_backbone(
            backbone_name="mobilenet_v3_large",
            weights=weights,
            fpn=True,
            trainable_layers=6,
            returned_layers=returned_layers if returned_layers is not None else [1, 2],
        )
        # heatmap head's final layer is zero-weight-initialized (RetinaNet/
        # CenterNet convention) so the *bias* alone controls the initial
        # output -- with a random weight init instead, the final conv's
        # output at init is dominated by random feature activations, not
        # the bias, and can push sigmoid to saturation (~0 or ~1) at many
        # pixels, which blew up this focal loss to the thousands before
        # this fix (log(1-pred) exploding for saturated negative pixels).
        # With zero weights, initial logits = bias = -2.19 everywhere
        # (sigmoid ~= 0.1), a sane, non-saturated starting point given
        # >99% of pixels are background for any one class.
        self.semantic_head = _conv_head(FPN_OUT_CHANNELS, num_classes + 1)
        self.heatmap_head = _conv_head(FPN_OUT_CHANNELS, num_classes)
        nn.init.constant_(self.heatmap_head[-1].weight, 0.0)
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)  # sigmoid(-2.19) ~= 0.1
        self.offset_head = _conv_head(FPN_OUT_CHANNELS, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        finest = features[next(iter(features))]
        return {
            "semantic": self.semantic_head(finest),
            "heatmap": self.heatmap_head(finest),
            "offset": self.offset_head(finest),
        }


@register_segmenter("centroid_offset_mobilenet_v3")
def build_centroid_offset_mobilenet_v3(num_classes: int, pretrained: bool) -> nn.Module:
    return CentroidOffsetSegmenter(num_classes=num_classes, pretrained=pretrained, returned_layers=[1, 2])


@register_segmenter("centroid_offset_mobilenet_v3_deep")
def build_centroid_offset_mobilenet_v3_deep(num_classes: int, pretrained: bool) -> nn.Module:
    return CentroidOffsetSegmenter(num_classes=num_classes, pretrained=pretrained, returned_layers=[1, 3, 5])
