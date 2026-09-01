"""Target generation for Milestone 9's centroid/offset instance segmenter.

Not a reproduction of one paper -- a practical combination of two ideas,
documented here so the citation is honest (CLAUDE.md: no claimed novelty):

  - A per-pixel semantic segmentation head (standard).
  - A centroid-heatmap + offset-regression head for instance separation,
    following Kurmann et al. 2021's "mask then classify" framing (group
    pixels around a predicted centroid, then read off class from the
    semantic head) and structurally close to Cheng et al. 2020's
    Panoptic-DeepLab (same heatmap+offset instance-center design).

Both heads are anchor-free and NMS-free: an overlapping instrument's pixels
are never suppressed the way Faster R-CNN's box NMS suppresses a heavily-
occluded detection. That is the specific, diagnosed failure mode this
architecture targets (docs/DECISIONS.md: heavy occlusion costs ~25-30
recall points on every Milestone 8 detector variant).

Heatmap peak-drawing uses the CornerNet/CenterNet Gaussian-radius formula
(Law & Deng 2018; Zhou et al. 2019, "Objects as Points") -- a standard,
widely-reproduced convention, not something derived here.

Targets are generated at `output_stride` (default 4) below the model's
input resolution, matching CenterNet/Panoptic-DeepLab convention. GraSP's
instruments are all large relative to COCO's small-object tail (first-
percentile mask area ~2400-2700px^2 per docs/DECISIONS.md), so stride-4
targets don't lose objects the way they might on COCO.
"""

from __future__ import annotations

import numpy as np
import torch


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CornerNet/CenterNet formula: smallest radius such that a box shifted
    by that radius from the ground-truth box still has >= min_overlap IoU
    with it, solved via three quadratics (one per corner-pair case).
    """
    a1, b1, c1 = 1, (height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1**2 - 4 * a1 * c1, 0))
    r1 = (b1 + sq1) / 2

    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2**2 - 4 * a2 * c2, 0))
    r2 = (b2 + sq2) / 2

    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(b3**2 - 4 * a3 * c3, 0))
    r3 = (b3 + sq3) / (2 * a3)

    return float(min(r1, r2, r3))


def draw_gaussian_peak(channel: np.ndarray, center: tuple[float, float], radius: float) -> None:
    """Draws one Gaussian peak into `channel` (H, W), elementwise-max'd with
    whatever is already there (CenterNet convention: overlapping peaks of
    the same class take the brighter value, never overwrite/average).
    """
    radius = max(int(round(radius)), 1)
    sigma = radius / 3.0
    cx, cy = int(round(center[0])), int(round(center[1]))
    h, w = channel.shape

    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    gauss = np.exp(-(xs[None, :] ** 2 + ys[:, None] ** 2) / (2 * sigma**2))
    np.maximum(channel[y0:y1, x0:x1], gauss, out=channel[y0:y1, x0:x1])


def downsample_mask_nearest(mask: np.ndarray, output_stride: int, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor downsample by simple striding (not interpolation --
    a binary mask has no useful in-between values to interpolate). Shared
    by target generation and evaluation (`evaluation/segmentation.py`) so
    ground-truth masks are downsampled identically in both places.
    """
    small = mask[::output_stride, ::output_stride][:out_h, :out_w]
    if small.shape == (out_h, out_w):
        return small
    padded = np.zeros((out_h, out_w), dtype=mask.dtype)
    padded[: small.shape[0], : small.shape[1]] = small
    return padded


def render_instance_targets(
    masks: list[np.ndarray],
    labels: list[int],
    num_classes: int,
    height: int,
    width: int,
    output_stride: int = 4,
) -> dict[str, torch.Tensor]:
    """`masks` are already resized to (height, width) -- the model's input
    resolution -- by the caller (GraspSegmentationDataset). Targets are
    produced at (height // output_stride, width // output_stride).

    Raster order matters where instances overlap: semantic/offset targets
    at a shared pixel take the *last* instance in the list, a simplification
    (no true multi-instance-per-pixel supervision) noted here rather than
    hidden -- exactly the kind of overlap this architecture is meant to
    handle better than box-NMS at inference, but training supervision still
    picks one instance per pixel.
    """
    out_h, out_w = height // output_stride, width // output_stride
    semantic = np.zeros((out_h, out_w), dtype=np.int64)
    heatmap = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    offset = np.zeros((2, out_h, out_w), dtype=np.float32)
    offset_mask = np.zeros((out_h, out_w), dtype=bool)

    grid_y, grid_x = np.mgrid[0:out_h, 0:out_w]

    for mask, label_idx in zip(masks, labels):
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        cx, cy = xs.mean() / output_stride, ys.mean() / output_stride

        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        radius = gaussian_radius((y1 - y0) / output_stride, (x1 - x0) / output_stride)
        draw_gaussian_peak(heatmap[label_idx], (cx, cy), radius)

        keep = downsample_mask_nearest(mask, output_stride, out_h, out_w).astype(bool)

        semantic[keep] = label_idx + 1  # 0 = background
        offset[0][keep] = cx - grid_x[keep]
        offset[1][keep] = cy - grid_y[keep]
        offset_mask[keep] = True

    return {
        "semantic": torch.from_numpy(semantic),
        "heatmap": torch.from_numpy(heatmap),
        "offset": torch.from_numpy(offset),
        "offset_mask": torch.from_numpy(offset_mask),
    }
