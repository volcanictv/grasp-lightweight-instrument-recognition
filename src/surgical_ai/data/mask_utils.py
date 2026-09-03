"""Shared COCO RLE instance-mask decoding for GraSP's per-annotation segmentations.

Used by both `region_dataset.py` (Task B crops) and `copy_paste.py`
(Milestone 8 augmentation) -- extracted here once there were two call sites.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils


def decode_instance_mask(segmentation: dict) -> np.ndarray:
    rle = {
        "size": segmentation["size"],
        "counts": segmentation["counts"].encode("utf-8")
        if isinstance(segmentation["counts"], str)
        else segmentation["counts"],
    }
    return mask_utils.decode(rle)


def is_likely_occluded(mask: np.ndarray, min_component_frac: float = 0.15) -> bool:
    """True if the mask is fragmented into two or more substantial pieces --
    the signature left by tissue/blood covering the middle of an instrument,
    not a single stray noisy pixel (docs/error_analysis.md's tissue-occlusion
    cause). A clean single-piece mask, even a heavily distorted or genuinely
    ambiguous-looking one, returns False -- this is deliberately narrow,
    only for the one failure mode temporal context was confirmed (not just
    assumed) to fix (docs/DECISIONS.md, 2026-09-02's temporal-context
    proof-of-concept: 40/41 nearby frames correct for a fragmented-mask
    instance, vs. 0/22 and ~1/3 for the two other diagnosed causes, where
    this signal does not fire). Validated directly against known examples
    of all three causes before use, not assumed to generalize.

    Fires on 2.69% of official test instances (77/2861) -- small and
    targeted by construction, not a broad trigger.
    """
    from scipy import ndimage

    labeled, n = ndimage.label(mask)
    if n < 2:
        return False
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    total = sizes.sum()
    if total == 0:
        return False
    significant = sum(1 for s in sizes if s / total >= min_component_frac)
    return significant >= 2
