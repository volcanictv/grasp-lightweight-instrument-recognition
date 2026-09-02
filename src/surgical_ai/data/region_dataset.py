"""Single-label instance-crop dataset (Task B).

One sample per annotated instrument instance (one row of the short-term
JSON's "annotations", not one row per frame). Each annotation already
carries its own per-instance COCO RLE segmentation and bbox -- no need to
touch the combined per-frame segmentation PNGs used by Milestone 0's
integrity checks.

Crops are bbox-cropped *and* mask-multiplied: background pixels inside the
bbox that belong to a different, co-occurring instrument (or plain tissue)
are zeroed out. This is `docs/imbalance_notes.md` Problem 4's fix for bbox
overlap -- ~30% of co-occurring instrument pairs have overlapping bboxes, so
a raw rectangular crop would routinely leak a second instrument into a
single-label sample.

Must be pointed at the uncached data root (`GraSP`/`GraSp`, not
`GraSP_cache`): bbox and segmentation coordinates are in the frame's native
resolution, and the resized frame cache built for Task A's dataloader
throughput (README.md Milestone 1) doesn't rescale annotations to match its
smaller frames. Using the cache here would silently produce wrong or
degenerate crops -- `__getitem__` checks the frame's shape against the
segmentation's recorded size and fails loudly instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from surgical_ai.data import splits, statistics
from surgical_ai.data.mask_utils import decode_instance_mask


def _pad_to_square(crop: np.ndarray) -> np.ndarray:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - ch) // 2, (side - cw) // 2
    square[top : top + ch, left : left + cw] = crop
    return square


class GraspRegionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        transform: Callable | None = None,
        letterbox: bool = False,
        letterbox_min_aspect: float = 1.0,
        crop_mode: str = "bbox",
        tip_crop_frac: float = 0.45,
    ):
        """`letterbox=True` pads the mask-cropped instance to a square (zeros,
        matching the mask-zeroed background already used) before any resize
        happens, instead of letting the downstream fixed-size resize stretch
        the crop's native aspect ratio to square. Motivated by
        docs/error_analysis.md: several of the region classifier's worst
        confusion pairs turned out to be long, thin instrument crops (e.g. a
        730x788 bbox where the tool is a diagonal sliver) where the
        distinguishing tip occupies a small fraction of the crop and gets
        compressed by a non-uniform stretch to 224x224. Padding to square
        first keeps the crop's true proportions through every transform
        after it (Resize, RandomResizedCrop) without touching those
        transforms or their configs.

        `letterbox_min_aspect` (long side / short side) gates which crops get
        padded -- a near-square crop barely distorts under a stretch resize,
        so padding it only adds black canvas with no upside. Default 1.0
        letterboxes every crop (the first ablation's behavior, kept for
        reproducibility). `docs/DECISIONS.md` 2026-09-02: the first
        unconditional-letterbox run fixed the diagnosed aspect-ratio
        confusion pairs but measurably hurt the two smallest classes (Clip
        Applier, Laparoscopic Grasper) -- restricting padding to genuinely
        elongated crops is the follow-up meant to keep the fix's benefit
        while reducing how much of the fixed-size canvas becomes wasted
        black padding overall.
        """
        if crop_mode not in ("bbox", "tip"):
            raise ValueError(f"unknown crop_mode '{crop_mode}'. Valid: bbox, tip")

        doc = splits.load_short_term(data_root, split)
        self.frames_root = Path(data_root) / "frames-001" / "frames"
        self.transform = transform
        self.letterbox = letterbox
        self.letterbox_min_aspect = letterbox_min_aspect
        self.crop_mode = crop_mode
        self.tip_crop_frac = tip_crop_frac

        self.category_ids = sorted(c["id"] for c in doc["categories"])
        self.category_names = statistics.category_names(doc)
        self._id_to_index = statistics.category_id_to_index(doc)

        images_by_id = {img["id"]: img for img in doc["images"]}

        self.instances: list[tuple[str, dict, tuple[int, int, int, int], int]] = []
        for a in doc["annotations"]:
            x, y, w, h = a["bbox"]
            x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
            if w <= 0 or h <= 0:
                continue
            img_meta = images_by_id[a["image_id"]]
            self.instances.append(
                (
                    img_meta["file_name"],
                    a["segmentation"],
                    (x, y, w, h),
                    self._id_to_index[a["category_id"]],
                )
            )

    @property
    def num_classes(self) -> int:
        return len(self.category_ids)

    def class_names_ordered(self) -> list[str]:
        return [self.category_names[cid] for cid in self.category_ids]

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_name, segmentation, (x, y, w, h), label_idx = self.instances[idx]
        frame = np.array(Image.open(self.frames_root / file_name).convert("RGB"))
        height, width = frame.shape[:2]

        # bbox/segmentation coordinates are in the frame's native resolution
        # (recorded in segmentation["size"]). A resized frame cache (built
        # for Task A's dataloader throughput, see README.md's Milestone 1
        # section) uses smaller pixel dimensions without rescaling
        # annotations, which silently produces degenerate or wrong crops --
        # fail loudly instead of computing garbage.
        native_size = tuple(segmentation["size"])
        if (height, width) != native_size:
            raise ValueError(
                f"frame {file_name} is {(height, width)} but its annotations are in "
                f"{native_size} space -- GraspRegionDataset needs the uncached data root "
                "(bbox/segmentation crop coordinates don't match a resized frame cache)"
            )

        mask = decode_instance_mask(segmentation)

        if self.crop_mode == "tip":
            crop = self._tip_crop(frame, mask, x, y, w, h, height, width)
        else:
            # Clip defensively -- bbox rounding can push a coordinate 1px
            # past the frame edge.
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)

            if self.letterbox:
                ch, cw = crop.shape[:2]
                aspect = max(ch, cw) / max(1, min(ch, cw))
                if aspect >= self.letterbox_min_aspect:
                    crop = _pad_to_square(crop)

        image = Image.fromarray(crop)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label_idx, dtype=torch.long)

    def _tip_crop(
        self, frame: np.ndarray, mask: np.ndarray, x: int, y: int, w: int, h: int,
        height: int, width: int,
    ) -> np.ndarray:
        """Crops a square window around the instrument's working tip instead
        of the whole bbox. Motivated by `docs/error_analysis.md`: letterbox
        padding (see class docstring) stops the fixed-size resize from
        *distorting* an elongated crop's aspect ratio, but the distinguishing
        jaw/tip shape is still only a small fraction of the crop's pixels --
        padding doesn't zoom in on it. Laparoscopic instruments enter the
        frame from an off-screen port, so the end of the instrument nearest
        a frame border is the entry point (uninformative shaft) and the end
        farthest from every border is the working tip -- found via the mask's
        PCA major axis rather than assumed from bbox geometry, since the
        instrument can be held at any angle.

        Falls back to a plain bbox crop for a degenerate (near-empty) mask.
        """
        ys, xs = np.nonzero(mask)
        if len(xs) < 2:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            return (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)

        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered / len(pts)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]
        minor_axis = eigvecs[:, np.argmin(eigvals)]

        proj_major = centered @ major_axis
        extreme_a, extreme_b = pts[np.argmin(proj_major)], pts[np.argmax(proj_major)]

        def border_dist(pt: np.ndarray) -> float:
            px, py = pt
            return min(px, width - 1 - px, py, height - 1 - py)

        tip_x, tip_y = extreme_a if border_dist(extreme_a) > border_dist(extreme_b) else extreme_b

        # Sized from the mask's own major/minor axis extents, not the
        # axis-aligned bbox's w/h -- a diagonally-held tool can have a
        # near-square bbox (e.g. 730x788) while the tool itself is a long
        # thin rod at 45 degrees, which would badly oversize the crop if
        # sized from bbox dimensions instead of the mask's true geometry.
        true_length = proj_major.max() - proj_major.min()
        true_width = np.ptp(centered @ minor_axis)
        side = max(self.tip_crop_frac * true_length, 1.5 * true_width)
        half = int(round(side)) // 2
        cx, cy = int(round(tip_x)), int(round(tip_y))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(width, cx + half), min(height, cy + half)

        crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)
        # A tip near a frame border clips asymmetrically and comes out
        # non-square -- pad rather than let the downstream resize distort it,
        # same reasoning as the letterbox option above.
        return _pad_to_square(crop)
