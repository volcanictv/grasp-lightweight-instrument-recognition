"""Copy-paste augmentation (Ghiasi et al. 2021, "Simple Copy-Paste is a
Strong Data Augmentation Method for Instance Segmentation") -- pasting real
instrument crops (cut out with GraSP's own per-instance COCO RLE masks,
already available, no new annotation) onto other training frames. Not a
GAN, no generative model, no mode-collapse risk: every pasted pixel is a
real instrument crop from this dataset.

Targets two diagnosed problems at once:
  - class imbalance: pasted instances are drawn only from `rare_classes`
    (default: Clip Applier, per CLAUDE.md's documented rare class).
  - occlusion: paste placement is biased (`occlusion_bias`) to land on top
    of a box already in the frame rather than in empty space, deliberately
    increasing exposure to the >50%-bbox-overlap regime where every
    detector variant loses ~25-30 recall points (see docs/DECISIONS.md).

Shared bank/patch/placement helpers below are used by both
`CopyPasteDetectionDataset` (Milestone 8) and
`SegmentationCopyPasteDataset` (Milestone 9, `segmentation_copy_paste.py`)
-- extracted here once there were two concrete users, not before.

Composition is a hard mask paste (no edge feathering) -- Ghiasi et al.'s own
ablation found blending the paste edge doesn't matter much, so there is no
case for the extra complexity here.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from surgical_ai.data.mask_utils import decode_instance_mask

if TYPE_CHECKING:
    from surgical_ai.data.detection_dataset import GraspDetectionDataset


def build_rare_class_bank(
    samples: list[tuple[str, list]], id_to_index: dict[int, int], rare_indices: set[int]
) -> dict[int, list[tuple[str, dict, tuple[float, float, float, float]]]]:
    bank: dict[int, list[tuple[str, dict, tuple[float, float, float, float]]]] = defaultdict(list)
    for file_name, anns in samples:
        for a in anns:
            label_idx = id_to_index[a["category_id"]]
            if label_idx not in rare_indices:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            bank[label_idx].append((file_name, a["segmentation"], (x, y, w, h)))
    return {k: v for k, v in bank.items() if v}


def load_patch(
    frame_cache: dict[str, np.ndarray],
    frames_root: Path,
    file_name: str,
    segmentation: dict,
    x: float,
    y: float,
    w: float,
    h: float,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int]]:
    if file_name not in frame_cache:
        image = Image.open(frames_root / file_name).convert("RGB")
        frame_cache[file_name] = np.array(image)
    frame = frame_cache[file_name]
    fh, fw = frame.shape[:2]

    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(fw, int(round(x + w))), min(fh, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None, None, (0, 0)

    mask = decode_instance_mask(segmentation)
    return frame[y0:y1, x0:x1], mask[y0:y1, x0:x1], (x1 - x0, y1 - y0)


def sample_paste_location(
    rng: random.Random,
    boxes: list[list[float]],
    width: int,
    height: int,
    pw: int,
    ph: int,
    occlusion_bias: float,
) -> tuple[int, int]:
    max_x, max_y = max(1, width - pw), max(1, height - ph)
    if boxes and rng.random() < occlusion_bias:
        bx0, by0, bx1, by1 = rng.choice(boxes)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        px = cx - pw / 2 + rng.uniform(-0.3 * pw, 0.3 * pw)
        py = cy - ph / 2 + rng.uniform(-0.3 * ph, 0.3 * ph)
    else:
        px = rng.uniform(0, max_x)
        py = rng.uniform(0, max_y)
    return int(min(max(0, px), max_x)), int(min(max(0, py), max_y))


def composite_patch(image_np: np.ndarray, patch_rgb: np.ndarray, patch_mask: np.ndarray, px: int, py: int) -> None:
    ph, pw = patch_mask.shape
    dst = image_np[py : py + ph, px : px + pw]
    keep = patch_mask.astype(bool)[..., None]
    dst[:] = np.where(keep, patch_rgb, dst)


class CopyPasteDetectionDataset(torch.utils.data.Dataset):
    """Wraps a *train-split* `GraspDetectionDataset`. Do not use on val/test
    -- evaluation must stay on real, unmodified frames.

    `include_masks` mirrors `GraspDetectionDataset`'s own flag -- set it to
    match the base dataset when training `maskrcnn_mobilenet_v3` (Milestone
    9.5), so pasted instances get a real per-instance mask (the patch mask
    placed at the paste location) alongside their box/label, not just a box.
    """

    def __init__(
        self,
        base_dataset: "GraspDetectionDataset",
        paste_prob: float = 0.5,
        max_pastes: int = 2,
        rare_classes: list[str] | None = None,
        occlusion_bias: float = 0.7,
        seed: int = 42,
        include_masks: bool = False,
    ):
        self.base = base_dataset
        self.paste_prob = paste_prob
        self.max_pastes = max_pastes
        self.occlusion_bias = occlusion_bias
        self.include_masks = include_masks
        self.rng = random.Random(seed)

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
        width, height = image.size

        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.base._id_to_index[a["category_id"]] + 1)
            if self.include_masks:
                masks.append(decode_instance_mask(a["segmentation"]))

        image_np = np.array(image)
        if self.bank and self.rng.random() < self.paste_prob:
            for _ in range(self.rng.randint(1, self.max_pastes)):
                label_idx = self.rng.choice(list(self.bank.keys()))
                src_file, segmentation, (sx, sy, sw, sh) = self.rng.choice(self.bank[label_idx])
                patch_rgb, patch_mask, (pw, ph) = load_patch(
                    self._frame_cache, self.base.frames_root, src_file, segmentation, sx, sy, sw, sh
                )
                if patch_rgb is None:
                    continue
                px, py = sample_paste_location(self.rng, boxes, width, height, pw, ph, self.occlusion_bias)
                composite_patch(image_np, patch_rgb, patch_mask, px, py)
                boxes.append([px, py, px + pw, py + ph])
                labels.append(label_idx + 1)
                if self.include_masks:
                    full_mask = np.zeros((height, width), dtype=np.uint8)
                    full_mask[py : py + ph, px : px + pw] = patch_mask
                    masks.append(full_mask)
                    # a paste can land on top of an earlier instance's mask (deliberately,
                    # per occlusion_bias) -- zero out whatever it covers in earlier masks so
                    # no two instances claim the same pixels, matching real GraSP masks
                    # (which are also non-overlapping per pixel).
                    paste_bool = full_mask.astype(bool)
                    for i in range(len(masks) - 1):
                        masks[i] = masks[i] & ~paste_bool

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(height, width)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        if self.include_masks:
            masks_array = np.stack(masks).astype(np.uint8) if masks else np.zeros((0, height, width), dtype=np.uint8)
            target["masks"] = tv_tensors.Mask(torch.from_numpy(masks_array))

        transform = self.base.transform or self._default_transform()
        image_out, target = transform(Image.fromarray(image_np), target)
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        if self.include_masks:
            target["masks"] = target["masks"].as_subclass(torch.Tensor)
        return image_out, target

    @staticmethod
    def _default_transform():
        from surgical_ai.data.detection_dataset import build_detection_transforms

        return build_detection_transforms(train=False)
