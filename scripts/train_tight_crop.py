"""Task B training for the tight-crop experiment (docs/reports/tight_crop/).

A thin wrapper around scripts/train.py: it replaces only the region-task data
setup, so the model, optimizer, loss, epoch loop, checkpointing and manifest
are exactly the standard ones. Two config options are added under `data:`

  crop_variant: standard | tight_band   what the TRAINING crops contain
                (standard = the project's mask-only letterbox crop)
  augmentation: default | masklight     masklight = default + lighting
                diversity + random masking (training only)
  tight_band_frac / tight_r_cap         band size, see data/tight_crop.py

The validation/test split is always the standard, unaltered evaluation
(mask-only letterbox crops, default eval transform), whatever the training
crops look like, so numbers compare with every earlier Task B result.

Usage (same flags as train.py):
    python scripts/train_tight_crop.py configs/tight_crop/<config>.yaml --device cuda:0 --experiments-dir DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import train as base  # noqa: E402
from surgical_ai.data.tight_crop import TightCropDataset  # noqa: E402
from surgical_ai.data.tight_transforms import build_tight_transforms  # noqa: E402


def _setup_tight_region_task(config: dict, args, device):
    data = config["data"]
    train_split, val_split = base.splits.resolve_train_val_split(data["split"])
    image_size = data["image_size"]
    augmentation = data.get("augmentation", "default")
    variant = data.get("crop_variant", "standard")
    if data.get("sampling", "none") != "none":
        raise ValueError("train_tight_crop.py supports data.sampling: none only")

    train_transform = build_tight_transforms(image_size, train=True, augmentation=augmentation)
    if variant == "standard":
        train_ds = base.GraspRegionDataset(
            args.data_root, train_split, transform=train_transform, letterbox=data.get("letterbox_crop", False),
        )
    elif variant == "tight_band":
        train_ds = TightCropDataset(
            args.data_root, train_split, transform=train_transform,
            band_frac=data.get("tight_band_frac", 0.5), r_cap=data.get("tight_r_cap", 24),
        )
    else:
        raise ValueError(f"unknown data.crop_variant '{variant}'. Valid: standard, tight_band")

    val_ds = base.GraspRegionDataset(
        args.data_root, val_split, transform=build_tight_transforms(image_size, train=False),
        letterbox=data.get("letterbox_crop", False),
    )
    class_names = train_ds.class_names_ordered()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=args.num_workers,
    )

    label_counts = torch.zeros(len(class_names))
    for _, _, _, label_idx in train_ds.instances:
        label_counts[label_idx] += 1
    class_weight = None
    if config["loss"].get("class_weights", False):
        class_weight = base.compute_class_weights(
            label_counts, power=config["loss"].get("class_weight_power", 1.0)
        ).to(device)
    loss_fn = base.build_loss(config["loss"], class_weight=class_weight)
    return train_loader, val_loader, class_names, loss_fn, base.evaluate_region


if __name__ == "__main__":
    base._setup_region_task = _setup_tight_region_task
    base.main()
