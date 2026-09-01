from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from surgical_ai.models.classifiers.common import freeze_all_except, set_head
from surgical_ai.models.registry import register_model


@register_model("efficientnet_b0")
def build_efficientnet_b0(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    set_head(model, model.classifier)

    if freeze_backbone:
        freeze_all_except(model, model.head)

    return model
