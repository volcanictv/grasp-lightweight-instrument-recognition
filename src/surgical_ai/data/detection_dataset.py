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
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from surgical_ai.data import splits, statistics


class GraspDetectionDataset(Dataset):
    def __init__(self, data_root: Path, split: str, transform: Callable | None = None):
        doc = splits.load_short_term(data_root, split)
        self.frames_root = Path(data_root) / "frames-001" / "frames"
        self.transform = transform

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

        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self._id_to_index[a["category_id"]] + 1)  # 0 = background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision.transforms.functional import to_tensor

            image = to_tensor(image)
        return image, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[tuple, tuple]:
    return tuple(zip(*batch))
