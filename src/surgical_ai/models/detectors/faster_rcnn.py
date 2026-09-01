from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from surgical_ai.models.detectors.registry import register_detector


@register_detector("fasterrcnn_mobilenet_v3")
def build_fasterrcnn_mobilenet_v3(num_classes: int, pretrained: bool) -> nn.Module:
    """MobileNetV3-Large-FPN backbone -- the lightweight choice consistent
    with this project's classifiers, and a standard torchvision detector
    (no novel architecture, per CLAUDE.md).
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
