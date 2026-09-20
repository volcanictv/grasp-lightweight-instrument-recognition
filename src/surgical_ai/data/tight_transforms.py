"""Training augmentation for the tight-crop experiment: the project's default
recipe plus lighting diversity and random masking (docs/reports/tight_crop/).

`augmentation: masklight` inserts two tensor-space ops after ToTensor, before
Normalize, into the unchanged default pipeline. Both keep exactly-zero pixels
at zero (crops use a black background), and stay inside CLAUDE.md's
augmentation limits: multiplicative gain, gamma, a smooth illumination
gradient and small per-channel gains (no hue shift), plus occluding patches.
Validation/test transforms are never touched.
"""

from __future__ import annotations

import math

import torch
from torchvision import transforms

from surgical_ai.data.transforms import build_transforms


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * torch.rand(1).item()


class RandomLighting:
    """Lighting diversity on a [0, 1] tensor. Every op is multiplicative or a
    power, so a black background stays black."""

    def __init__(self, gain=(0.6, 1.4), gamma=(0.7, 1.5), gradient_max=0.4, channel_gain=0.08, p=0.8):
        self.gain, self.gamma = gain, gamma
        self.gradient_max, self.channel_gain, self.p = gradient_max, channel_gain, p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            t = t * _uniform(*self.gain)
        if torch.rand(1).item() < self.p:
            t = t.clamp(0, 1) ** _uniform(*self.gamma)
        if torch.rand(1).item() < self.p:
            _, h, w = t.shape
            angle = _uniform(0, 2 * math.pi)
            ys = torch.linspace(-1, 1, h).view(h, 1)
            xs = torch.linspace(-1, 1, w).view(1, w)
            ramp = (xs * math.cos(angle) + ys * math.sin(angle)) / (abs(math.cos(angle)) + abs(math.sin(angle)))
            t = t * (1 + _uniform(0, self.gradient_max) * ramp).clamp(min=0.2)
        if torch.rand(1).item() < self.p:
            gains = 1 + (torch.rand(3, 1, 1) * 2 - 1) * self.channel_gain
            t = t * gains
        return t.clamp(0, 1)


class RandomPatchMask:
    """Random masking: blank 1-3 rectangles (2-15% of the crop each) to black,
    the same value as the crop's own background."""

    def __init__(self, p=0.5, max_patches=3, area=(0.02, 0.15), aspect=(0.5, 2.0)):
        self.p, self.max_patches, self.area, self.aspect = p, max_patches, area, aspect

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return t
        _, h, w = t.shape
        t = t.clone()
        for _ in range(torch.randint(1, self.max_patches + 1, (1,)).item()):
            frac = _uniform(*self.area)
            ratio = math.exp(_uniform(math.log(self.aspect[0]), math.log(self.aspect[1])))
            ph = min(h, max(1, int(round(math.sqrt(frac * h * w * ratio)))))
            pw = min(w, max(1, int(round(math.sqrt(frac * h * w / ratio)))))
            top = torch.randint(0, h - ph + 1, (1,)).item()
            left = torch.randint(0, w - pw + 1, (1,)).item()
            t[:, top : top + ph, left : left + pw] = 0
        return t


def build_tight_transforms(image_size: int, train: bool, augmentation: str = "default") -> transforms.Compose:
    if not train or augmentation == "default":
        return build_transforms(image_size, train=train, augmentation="default")
    if augmentation != "masklight":
        raise ValueError(f"unknown data.augmentation '{augmentation}'. Valid: default, masklight")
    ops = list(build_transforms(image_size, train=True, augmentation="default").transforms)
    at = next(i for i, op in enumerate(ops) if isinstance(op, transforms.ToTensor))
    ops[at + 1 : at + 1] = [RandomLighting(), RandomPatchMask()]
    return transforms.Compose(ops)
