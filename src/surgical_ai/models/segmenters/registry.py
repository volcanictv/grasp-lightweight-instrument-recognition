"""Segmenter registry, mirroring `models/detectors/registry.py`'s pattern.
Kept separate from the classifier/detector registries because a segmenter's
forward pass returns a dict of three per-pixel heads, a different interface
again from both.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_segmenter(name: str):
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _REGISTRY:
            raise ValueError(f"segmenter '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def build_segmenter(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"unknown segmenter '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained)


from surgical_ai.models.segmenters import centroid_offset  # noqa: E402,F401
