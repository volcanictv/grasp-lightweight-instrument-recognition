"""Instance segmentation dataset (Milestone 9).

One sample per annotated frame, all instances in that frame -- same
per-frame sample granularity as `detection_dataset.py`. Masks come from the
same per-instance COCO RLE segmentations Task B and Milestone 8 already use
(`data/mask_utils.py`), no new annotation.

Follows Task A/B's convention of a fixed square resize (ignoring aspect
ratio) rather than letterboxing -- consistent with the rest of this
project, not a new choice made here.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from surgical_ai.data import splits, statistics
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.segmentation_targets import render_instance_targets
from surgical_ai.data.transforms import IMAGENET_MEAN, IMAGENET_STD


class GraspSegmentationDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        image_size: int = 384,
        output_stride: int = 4,
        transform: Callable | None = None,
    ):
        doc = splits.load_short_term(data_root, split)
        self.frames_root = Path(data_root) / "frames-001" / "frames"
        self.image_size = image_size
        self.output_stride = output_stride
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
        native_width, native_height = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        masks: list[np.ndarray] = []
        labels: list[int] = []
        for a in anns:
            native_mask = decode_instance_mask(a["segmentation"])
            mask = np.array(
                Image.fromarray(native_mask * 255).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                )
            ) > 0
            if not mask.any():
                continue
            masks.append(mask)
            labels.append(self._id_to_index[a["category_id"]])

        target = render_instance_targets(
            masks, labels, self.num_classes, self.image_size, self.image_size, self.output_stride
        )
        target["instance_masks"] = (
            torch.from_numpy(np.stack(masks).astype(np.uint8))
            if masks
            else torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)
        )
        target["instance_labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        if self.transform is not None:
            image_tensor, target = self.transform(image_tensor, target)
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, dict]:
    images = torch.stack([b[0] for b in batch])
    targets = {
        "semantic": torch.stack([b[1]["semantic"] for b in batch]),
        "heatmap": torch.stack([b[1]["heatmap"] for b in batch]),
        "offset": torch.stack([b[1]["offset"] for b in batch]),
        "offset_mask": torch.stack([b[1]["offset_mask"] for b in batch]),
        "instance_masks": [b[1]["instance_masks"] for b in batch],
        "instance_labels": [b[1]["instance_labels"] for b in batch],
        "image_id": torch.cat([b[1]["image_id"] for b in batch]),
    }
    return images, targets
