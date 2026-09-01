"""Image transforms for GraSP frames.

Augmentation choices are constrained by CLAUDE.md: horizontal flip, mild
color jitter, blur, noise, and mild crop/scale are fine. No vertical flip,
no large rotation, no hue shift that turns tissue non-physiological.
"""

from __future__ import annotations

import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GaussianNoise:
    """Adds mild zero-mean Gaussian noise to a normalized tensor."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


def build_transforms(image_size: int, train: bool, augmentation: str = "default") -> transforms.Compose:
    """`augmentation="strong"` is a Milestone 5 ablation variable (PROJECT_SPEC.md
    Sec.7 "D. Augmentation") — wider crop range, stronger color jitter, and
    blur/noise applied more often, still within CLAUDE.md's plausibility
    constraints (horizontal flip only, no rotation, no hue shift).
    """
    if not train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if augmentation == "default":
        crop_scale, jitter, apply_p, noise_std = (0.8, 1.0), 0.2, 0.2, 0.02
    elif augmentation == "strong":
        crop_scale, jitter, apply_p, noise_std = (0.6, 1.0), 0.4, 0.5, 0.04
    else:
        raise ValueError(f"unknown data.augmentation '{augmentation}'. Valid: default, strong")

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=crop_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=apply_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomApply([GaussianNoise(std=noise_std)], p=apply_p),
        ]
    )
