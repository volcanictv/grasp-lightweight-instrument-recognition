from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
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


@register_detector("fasterrcnn_resnet50")
def build_fasterrcnn_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    """ResNet-50-FPN backbone -- the heavy-baseline choice consistent with
    Task A's backbone sweep (Milestone 6), tested here to see whether the
    same "heavier backbone buys a real but bounded accuracy gain" pattern
    holds for detection too, per Milestone 8's follow-up plan.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
