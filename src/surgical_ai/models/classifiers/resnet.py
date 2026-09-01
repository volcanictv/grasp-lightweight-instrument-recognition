"""resnet18 is the Milestone 6 sweep entry. resnet50 is the deliberately
heavy baseline PROJECT_SPEC.md Sec.10 calls for -- see docs/DECISIONS.md
for why ResNet-50 specifically was picked over e.g. a ViT or TAPIS itself.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from surgical_ai.models.classifiers.common import freeze_all_except, set_head
from surgical_ai.models.registry import register_model


def _build_resnet(
    backbone_fn, weights_enum, num_classes: int, pretrained: bool, freeze_backbone: bool
) -> nn.Module:
    weights = weights_enum.DEFAULT if pretrained else None
    model = backbone_fn(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    set_head(model, model.fc)

    if freeze_backbone:
        freeze_all_except(model, model.head)

    return model


@register_model("resnet18")
def build_resnet18(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    return _build_resnet(resnet18, ResNet18_Weights, num_classes, pretrained, freeze_backbone)


@register_model("resnet50")
def build_resnet50(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    return _build_resnet(resnet50, ResNet50_Weights, num_classes, pretrained, freeze_backbone)
