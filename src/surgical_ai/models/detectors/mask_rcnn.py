from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import mobilenet_backbone

from surgical_ai.models.detectors.registry import register_detector

# Matches torchvision's internal `_fasterrcnn_mobilenet_v3_large_fpn` anchor
# config exactly (3 FPN levels x 5 sizes x 3 ratios = 15 anchors/location).
# Needed so `fasterrcnn_mobilenet_v3`'s trained backbone/RPN/box-head weights
# load into this model without a shape mismatch -- verified directly
# (load_state_dict(strict=False) leaves only the 12 new mask-head tensors
# missing, zero unexpected keys).
_ANCHOR_SIZES = ((32, 64, 128, 256, 512),) * 3
_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(_ANCHOR_SIZES)


@register_detector("maskrcnn_mobilenet_v3")
def build_maskrcnn_mobilenet_v3(num_classes: int, pretrained: bool) -> nn.Module:
    """Proposal-based instance segmentation on the same MobileNetV3-Large-FPN
    backbone as `fasterrcnn_mobilenet_v3` -- deliberately built (same
    backbone construction, same custom anchor generator) so a trained
    `fasterrcnn_mobilenet_v3` checkpoint's backbone/RPN/box-head weights load
    directly into this model (`load_state_dict(..., strict=False)` in the
    training script), warm-starting box localization from an already-good
    detector (AP50_box 0.831, docs/DECISIONS.md) instead of learning it from
    scratch. Only the new mask head starts untrained. Standard torchvision
    detection architecture, no novel design.
    """
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    backbone = mobilenet_backbone(
        backbone_name="mobilenet_v3_large", weights=weights, fpn=True, trainable_layers=6,
    )
    anchor_generator = AnchorGenerator(_ANCHOR_SIZES, _ASPECT_RATIOS)
    return MaskRCNN(backbone, num_classes=num_classes + 1, rpn_anchor_generator=anchor_generator)
