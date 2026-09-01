"""Multi-label frame classification dataset (Task A).

One sample per annotated keyframe. Label is a 7-dim multi-hot vector over
instrument categories present in that frame (a frame with Bipolar Forceps
and Suction Instrument both present gets both bits set). Lazy loading only:
images are read from frames-001/frames/ on access, nothing is copied or
cached to a duplicate directory here.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from surgical_ai.data import splits, statistics


class GraspMultiLabelDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        transform: Callable | None = None,
    ):
        doc = splits.load_short_term(data_root, split)
        self.frames_root = Path(data_root) / "frames-001" / "frames"
        self.transform = transform

        self.category_ids = sorted(c["id"] for c in doc["categories"])
        self.category_names = statistics.category_names(doc)
        self._id_to_index = statistics.category_id_to_index(doc)

        labels_by_image_id: dict[int, set[int]] = defaultdict(set)
        for a in doc["annotations"]:
            labels_by_image_id[a["image_id"]].add(a["category_id"])

        # Every keyframe in the short-term files has >=1 annotation (verified
        # in Milestone 0), so this only drops frames if that ever changes.
        self.samples: list[tuple[str, torch.Tensor]] = []
        for img in doc["images"]:
            classes = labels_by_image_id.get(img["id"])
            if not classes:
                continue
            label = torch.zeros(len(self.category_ids), dtype=torch.float32)
            for cid in classes:
                label[self._id_to_index[cid]] = 1.0
            self.samples.append((img["file_name"], label))

    @property
    def num_classes(self) -> int:
        return len(self.category_ids)

    def class_names_ordered(self) -> list[str]:
        return [self.category_names[cid] for cid in self.category_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_name, label = self.samples[idx]
        image = Image.open(self.frames_root / file_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
