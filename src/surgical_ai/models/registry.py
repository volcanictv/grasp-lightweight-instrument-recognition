"""Model registry. Adding a backbone means adding one `@register_model(...)`
function here (or a new module imported below) — never touching data/,
training/, or evaluation/.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def build_model(name: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"unknown model '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](
        num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
    )


# Import submodules for their registration side effects.
from surgical_ai.models.classifiers import efficientnet, mobilenet, resnet  # noqa: E402,F401
