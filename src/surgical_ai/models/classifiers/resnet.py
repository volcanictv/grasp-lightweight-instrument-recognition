"""resnet18 is the Milestone 6 sweep entry. resnet50 is the deliberately
heavy baseline PROJECT_SPEC.md Sec.10 calls for -- see docs/DECISIONS.md
for why ResNet-50 specifically was picked over e.g. a ViT or TAPIS itself.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet50,
    resnet101,
)

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


@register_model("resnet101")
def build_resnet101(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    return _build_resnet(resnet101, ResNet101_Weights, num_classes, pretrained, freeze_backbone)


def _build_resnet_with_dropout(
    backbone_fn, weights_enum, num_classes: int, pretrained: bool, freeze_backbone: bool, dropout_p: float = 0.2
) -> nn.Module:
    """Same as `_build_resnet`, but with a Dropout layer before the final
    Linear -- plain ResNet has none (torchvision's own definition), which
    makes MC Dropout (Gal & Ghahramani 2016) silently deterministic for
    this backbone at inference time regardless of how many stochastic
    forward passes are run (docs/DECISIONS.md, 2026-09-16 uncertainty
    signal comparison: verified directly against the model definition,
    not assumed). `dropout_p=0.2` matches MobileNetV3's own existing
    pre-classifier dropout rate, for consistency across the ensemble
    rather than a second, unrelated hyperparameter.

    Separate registered name, not a change to `_build_resnet` itself --
    existing `resnet50`/`resnet101` checkpoints have `fc.weight`/`fc.bias`
    keys; this changes `fc` to a 2-element Sequential, which would break
    `strict=True` loading of every checkpoint trained before this existed.
    """
    weights = weights_enum.DEFAULT if pretrained else None
    model = backbone_fn(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
    set_head(model, model.fc)

    if freeze_backbone:
        freeze_all_except(model, model.head)

    return model


@register_model("resnet50_mcdropout")
def build_resnet50_mcdropout(num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    return _build_resnet_with_dropout(resnet50, ResNet50_Weights, num_classes, pretrained, freeze_backbone)
