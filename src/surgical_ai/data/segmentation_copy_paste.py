"""Copy-paste augmentation for the Milestone 9 segmenter -- the same
technique already shown to be the single most effective Milestone 8 lever
(`copy_paste.py`, `docs/DECISIONS.md`), ported to the segmentation
pipeline. Not yet tried here as of the previous plateau
(`segmentation_extended_patience`/`segmentation_cosine_lr`): those three
runs converged without ever increasing occlusion exposure or Clip Applier
representation during training.

Unlike detection, pasting must also update the dense targets
(`data/segmentation_targets.py`), not just a box+label list -- the pasted
instance needs its own heatmap peak, offset field contribution, and
semantic-map pixels, exactly like a real instance. Compositing happens at
native frame resolution (before the base dataset's resize-to-image_size
step) so the pasted patch is resized along with everything else, keeping
it consistent with how every other instance in the frame is processed.
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from surgical_ai.data.copy_paste import build_rare_class_bank, composite_patch, load_patch, sample_paste_location
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.segmentation_targets import render_instance_targets
from surgical_ai.data.transforms import IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from surgical_ai.data.segmentation_dataset import GraspSegmentationDataset


class SegmentationCopyPasteDataset(torch.utils.data.Dataset):
    """Wraps a *train-split* `GraspSegmentationDataset`. Do not use on
    val/test -- evaluation must stay on real, unmodified frames.
    """

    def __init__(
        self,
        base_dataset: "GraspSegmentationDataset",
        paste_prob: float = 0.5,
        max_pastes: int = 2,
        rare_classes: list[str] | None = None,
        occlusion_bias: float = 0.7,
        seed: int = 42,
    ):
        self.base = base_dataset
        self.paste_prob = paste_prob
        self.max_pastes = max_pastes
        self.occlusion_bias = occlusion_bias
        self.rng = Random(seed)

        rare_classes = rare_classes if rare_classes is not None else ["Clip Applier"]
        class_names = base_dataset.class_names_ordered()
        rare_indices = {class_names.index(name) for name in rare_classes}
        self.bank = build_rare_class_bank(base_dataset.samples, base_dataset._id_to_index, rare_indices)

        self._frame_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        file_name, anns = self.base.samples[idx]
        image = Image.open(self.base.frames_root / file_name).convert("RGB")
        native_width, native_height = image.size
        image_np = np.array(image)

        masks: list[np.ndarray] = []
        labels: list[int] = []
        boxes: list[list[float]] = []
        for a in anns:
            mask = decode_instance_mask(a["segmentation"])
            if not mask.any():
                continue
            masks.append(mask.astype(bool))
            labels.append(self.base._id_to_index[a["category_id"]])
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])

        if self.bank and self.rng.random() < self.paste_prob:
            for _ in range(self.rng.randint(1, self.max_pastes)):
                label_idx = self.rng.choice(list(self.bank.keys()))
                src_file, segmentation, (sx, sy, sw, sh) = self.rng.choice(self.bank[label_idx])
                patch_rgb, patch_mask, (pw, ph) = load_patch(
                    self._frame_cache, self.base.frames_root, src_file, segmentation, sx, sy, sw, sh
                )
                if patch_rgb is None:
                    continue
                px, py = sample_paste_location(
                    self.rng, boxes, native_width, native_height, pw, ph, self.occlusion_bias
                )
                composite_patch(image_np, patch_rgb, patch_mask, px, py)

                full_mask = np.zeros((native_height, native_width), dtype=bool)
                full_mask[py : py + ph, px : px + pw] = patch_mask.astype(bool)
                masks.append(full_mask)
                labels.append(label_idx)
                boxes.append([px, py, px + pw, py + ph])

        image_size = self.base.image_size
        image_resized = Image.fromarray(image_np).resize((image_size, image_size), Image.BILINEAR)
        masks_resized = []
        for mask in masks:
            resized = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((image_size, image_size), Image.NEAREST)
            ) > 0
            masks_resized.append(resized)

        target = render_instance_targets(
            masks_resized, labels, self.base.num_classes, image_size, image_size, self.base.output_stride
        )
        target["instance_masks"] = (
            torch.from_numpy(np.stack(masks_resized).astype(np.uint8))
            if masks_resized
            else torch.zeros((0, image_size, image_size), dtype=torch.uint8)
        )
        target["instance_labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        image_tensor = torch.from_numpy(np.array(image_resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor, target
