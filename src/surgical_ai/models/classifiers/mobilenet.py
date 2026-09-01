from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

from surgical_ai.models.classifiers.common import freeze_all_except, set_head
from surgical_ai.models.registry import register_model


def _build_mobilenet_v3(
    backbone_fn, weights_enum, num_classes: int, pretrained: bool, freeze_backbone: bool
) -> nn.Module:
    weights = weights_enum.DEFAULT if pretrained else None
    model = backbone_fn(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    set_head(model, model.classifier)

    if freeze_backbone:
        freeze_all_except(model, model.head)

    return model


@register_model("mobilenet_v3_small")
def build_mobilenet_v3_small(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    return _build_mobilenet_v3(
        mobilenet_v3_small, MobileNet_V3_Small_Weights, num_classes, pretrained, freeze_backbone
    )


@register_model("mobilenet_v3_large")
def build_mobilenet_v3_large(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    return _build_mobilenet_v3(
        mobilenet_v3_large, MobileNet_V3_Large_Weights, num_classes, pretrained, freeze_backbone
    )
