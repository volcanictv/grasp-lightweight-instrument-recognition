"""Tight instrument crops with a bounded background buffer, plus a raw-bbox
crop for the robustness test (docs/reports/tight_crop/).

The standard Task B crop (`GraspRegionDataset`) keeps only the instrument's own
mask pixels and zeroes everything else inside the bbox. `TightCropDataset`
instead keeps the instrument plus a band of the *real* surrounding background,
so the model sees the instrument's silhouette against its actual surroundings,
while the instrument still outnumbers the band:

- the band is the set of background pixels nearest to the mask, taken until it
  holds `band_frac` x the instrument's own pixel count (`band_frac` < 1, so the
  instrument always has strictly more pixels than the band), never farther than
  `r_cap` pixels from the mask;
- pixels of any *other* annotated instrument in the same frame are excluded from
  the band, so each crop shows exactly one instrument (same guarantee as the
  standard crop);
- everything outside instrument + band is black, matching the standard crop's
  convention, and the region is cropped to its bounding box and padded to a
  square before the model-size resize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset, _pad_to_square


def tight_band_crop(
    frame: np.ndarray,
    mask: np.ndarray,
    other_mask: np.ndarray | None = None,
    band_frac: float = 0.5,
    r_cap: int = 24,
) -> tuple[np.ndarray, dict]:
    """Returns (crop, info). `info` records the band radius reached and the
    instrument/band pixel counts so the instrument > band invariant can be
    audited over a whole dataset."""
    if not 0 < band_frac < 1:
        raise ValueError(f"band_frac must be in (0, 1) so the instrument outnumbers the band, got {band_frac}")
    mask = mask.astype(bool)
    area = int(mask.sum())
    if area == 0:
        return np.zeros((8, 8, 3), dtype=np.uint8), {"radius": 0.0, "instrument_px": 0, "band_px": 0, "empty": True}

    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    wy0, wy1 = max(0, ys.min() - r_cap), min(height, ys.max() + 1 + r_cap)
    wx0, wx1 = max(0, xs.min() - r_cap), min(width, xs.max() + 1 + r_cap)
    window_mask = mask[wy0:wy1, wx0:wx1]

    dist = ndimage.distance_transform_edt(~window_mask)
    allowed = ~window_mask
    if other_mask is not None:
        allowed &= ~other_mask[wy0:wy1, wx0:wx1].astype(bool)
    allowed &= dist <= r_cap

    d_sorted = np.sort(dist[allowed])
    n_max = int(band_frac * area)
    if n_max <= 0 or len(d_sorted) == 0:
        band = np.zeros_like(window_mask)
    elif n_max >= len(d_sorted):
        band = allowed
    else:
        # strict "<" so ties at the cutoff distance cannot push the band past n_max
        band = allowed & (dist < d_sorted[n_max])

    keep = window_mask | band
    ky, kx = np.nonzero(keep)
    ky0, ky1, kx0, kx1 = ky.min(), ky.max() + 1, kx.min(), kx.max() + 1
    window_frame = frame[wy0:wy1, wx0:wx1]
    crop = (window_frame[ky0:ky1, kx0:kx1] * keep[ky0:ky1, kx0:kx1, None]).astype(np.uint8)
    info = {
        "radius": float(dist[band].max()) if band.any() else 0.0,
        "instrument_px": area,
        "band_px": int(band.sum()),
        "empty": False,
    }
    return crop, info


class _NativeFrameDataset(GraspRegionDataset):
    """Shared frame loading (with the same native-resolution guard as the
    standard dataset) for the two crop variants below."""

    def __init__(self, data_root: Path, split: str, transform: Callable | None = None):
        super().__init__(data_root, split, transform=transform)
        self._by_file: dict[str, list[int]] = {}
        for idx, (file_name, _seg, _box, _label) in enumerate(self.instances):
            self._by_file.setdefault(file_name, []).append(idx)

    def _load(self, idx: int) -> tuple[np.ndarray, dict, tuple[int, int, int, int], int, str]:
        file_name, segmentation, box, label_idx = self.instances[idx]
        frame = np.array(Image.open(self.frames_root / file_name).convert("RGB"))
        if frame.shape[:2] != tuple(segmentation["size"]):
            raise ValueError(
                f"frame {file_name} is {frame.shape[:2]} but its annotations are in "
                f"{tuple(segmentation['size'])} space -- needs the uncached data root"
            )
        return frame, segmentation, box, label_idx, file_name

    def _finish(self, crop: np.ndarray, label_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.fromarray(_pad_to_square(crop))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label_idx, dtype=torch.long)


class TightCropDataset(_NativeFrameDataset):
    """Instrument + bounded background band, single instrument per crop."""

    def __init__(self, data_root: Path, split: str, transform: Callable | None = None,
                 band_frac: float = 0.5, r_cap: int = 24):
        super().__init__(data_root, split, transform=transform)
        self.band_frac = band_frac
        self.r_cap = r_cap

    def crop_and_info(self, idx: int) -> tuple[np.ndarray, dict]:
        frame, segmentation, _box, _label, file_name = self._load(idx)
        mask = decode_instance_mask(segmentation).astype(bool)
        others = np.zeros_like(mask)
        for j in self._by_file[file_name]:
            if j != idx:
                others |= decode_instance_mask(self.instances[j][1]).astype(bool)
        return tight_band_crop(frame, mask, others, self.band_frac, self.r_cap)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        crop, _info = self.crop_and_info(idx)
        return self._finish(crop, self.instances[idx][3])


class RawBBoxCropDataset(_NativeFrameDataset):
    """Untouched original pixels inside the annotated bbox: no mask multiply,
    so other instruments and tissue stay in the crop. A robustness test, not
    the standard evaluation."""

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame, _segmentation, (x, y, w, h), label_idx, _file = self._load(idx)
        height, width = frame.shape[:2]
        crop = frame[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]
        return self._finish(crop, label_idx)
