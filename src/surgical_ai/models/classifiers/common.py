"""Shared helpers for torchvision-based classifier backbones.

Every builder in this package sets `model.head` to the replaced
classification head submodule (`.classifier` for MobileNetV3/EfficientNet,
`.fc` for ResNet). This gives training code (optimizer param groups,
freezing) a backbone-agnostic way to ask "which params are the head"
without name-substring matching against architecture-specific attribute
names.
"""

from __future__ import annotations

import torch.nn as nn


def set_head(model: nn.Module, head: nn.Module) -> None:
    """Sets model.head as a plain attribute alias, not a second registered
    submodule. A normal `model.head = head` goes through nn.Module's
    __setattr__, which registers any nn.Module-valued attribute in
    `_modules` -- since `head` is already registered under its real name
    (`classifier` or `fc`), that silently duplicates every one of its
    parameters under a second `head.*` key in state_dict(), breaking
    strict-mode loading of any checkpoint saved before this alias existed
    (see docs/DECISIONS.md). Writing directly into __dict__ bypasses that
    registration while `model.head` still resolves to the same submodule.
    """
    model.__dict__["head"] = head


def freeze_all_except(model: nn.Module, head: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True
