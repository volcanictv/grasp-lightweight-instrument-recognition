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
