"""Detector registry, mirroring models/registry.py's pattern for
classifiers. Kept separate because detector models have a fundamentally
different call interface (dict-of-losses in train mode, list-of-predictions
in eval mode) than the classifiers -- forcing them through the same
registry/build_model signature would blur that distinction rather than
clarify it.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_detector(name: str):
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise ValueError(f"detector '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def build_detector(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """num_classes is instrument classes only; +1 for background is handled
    by each builder, not the caller, to match how the box predictor head
    actually needs to be sized.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown detector '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained)


from surgical_ai.models.detectors import faster_rcnn  # noqa: E402,F401
