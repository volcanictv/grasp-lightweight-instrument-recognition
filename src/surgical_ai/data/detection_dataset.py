"""Instrument detection dataset (Task: object detection, Milestone 8).

One sample per annotated frame, all instances in that frame as one target
set -- unlike Task B, which is one sample per instance. Boxes come directly
from the "bbox" field already in the short-term annotations; no new
annotation was needed, per CLAUDE.md's milestone note.

Boxes are returned in torchvision's [x1, y1, x2, y2] absolute-pixel
convention (converted from the JSON's COCO-style [x, y, w, h]). Labels are
1..num_classes (0 is reserved for background by every torchvision detection
model) using the same category ordering as Task A/B, so class names line up
across tasks.

No resize/normalize transform here: torchvision's detection models
(GeneralizedRCNNTransform, wrapped inside FasterRCNN) do their own resizing
and ImageNet normalization internally from raw-pixel-space tensors and
targets -- applying it here would double-normalize and desync the boxes
from the image.

Augmentation (`build_detection_transforms`) uses torchvision.transforms.v2,
which jointly transforms images and `tv_tensors.BoundingBoxes` so a crop or
flip moves the boxes along with the pixels -- doing this by hand (as the
plain-tensor transforms in transforms.py do for classification) risks
silently desyncing boxes from content.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

import numpy as np

from surgical_ai.data import splits, statistics
from surgical_ai.data.mask_utils import decode_instance_mask


def build_detection_transforms(train: bool, augmentation: str = "none") -> Callable:
    """`augmentation="strong"` is torchvision's own reference detection
    recipe (RandomIoUCrop + RandomZoomOut + horizontal flip + color jitter) --
    standard practice, not novel, per CLAUDE.md. Tried first (Milestone 8
    follow-up) and found to make both mAP@50 and mAP@50:95 slightly *worse*
    than no augmentation at all -- see docs/DECISIONS.md. `augmentation=
    "default"` is the lighter fallback: just horizontal flip + color jitter,
    matching what actually helped the classification tasks (transforms.py),
    without the aggressive scale/crop jitter that regressed here -- also
    found to slightly regress both mAPs on its own. `"none"` (the default,
    matching the original Milestone 8 baseline's actual behavior, which
    predates this function having an `augmentation` parameter at all --
    keep this the fallback so old configs that don't set `data.augmentation`
    stay reproducible) applies no transform beyond tensor conversion, same
    as eval. All variants stay within this project's surgical-plausibility
    constraints: no vertical flip, no rotation, no hue shift.
    SanitizeBoundingBoxes (strong only, since only RandomIoUCrop can produce
    a degenerate box) drops any box a crop reduces to zero/degenerate area,
    and its matching label.
    """
    if not train or augmentation == "none":
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    if augmentation == "default":
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
    if augmentation == "strong":
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomZoomOut(fill=0, side_range=(1.0, 3.0), p=0.5),
                v2.RandomIoUCrop(),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.SanitizeBoundingBoxes(),
            ]
        )
    raise ValueError(f"unknown augmentation '{augmentation}'. Valid: none, default, strong")


class GraspDetectionDataset(Dataset):
    def __init__(
        self, data_root: Path, split: str, transform: Callable | None = None, include_masks: bool = False,
    ):
        doc = splits.load_short_term(data_root, split)
        self.frames_root = Path(data_root) / "frames-001" / "frames"
        self.transform = transform
        self.include_masks = include_masks

        self.category_ids = sorted(c["id"] for c in doc["categories"])
        self.category_names = statistics.category_names(doc)
        self._id_to_index = statistics.category_id_to_index(doc)

        anns_by_image: dict[int, list] = defaultdict(list)
        for a in doc["annotations"]:
            anns_by_image[a["image_id"]].append(a)

        self.samples: list[tuple[str, list]] = []
        for img in doc["images"]:
            anns = anns_by_image.get(img["id"])
            if not anns:
                continue
            self.samples.append((img["file_name"], anns))

    @property
    def num_classes(self) -> int:
        return len(self.category_ids)

    def class_names_ordered(self) -> list[str]:
        return [self.category_names[cid] for cid in self.category_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        file_name, anns = self.samples[idx]
        image = Image.open(self.frames_root / file_name).convert("RGB")
        width, height = image.size

        boxes, labels, masks = [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self._id_to_index[a["category_id"]] + 1)  # 0 = background
            if self.include_masks:
                masks.append(decode_instance_mask(a["segmentation"]))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_tensor, format="XYXY", canvas_size=(height, width)
            ),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        if self.include_masks:
            masks_array = np.stack(masks).astype(np.uint8) if masks else np.zeros((0, height, width), dtype=np.uint8)
            target["masks"] = tv_tensors.Mask(torch.from_numpy(masks_array))

        transform = self.transform or build_detection_transforms(train=False)
        image, target = transform(image, target)
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        if self.include_masks:
            target["masks"] = target["masks"].as_subclass(torch.Tensor)
        return image, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[tuple, tuple]:
    return tuple(zip(*batch))
