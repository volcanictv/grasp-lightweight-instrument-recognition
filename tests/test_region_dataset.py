import json

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from surgical_ai.data.region_dataset import GraspRegionDataset


def _make_dataset(tmp_path, frame_size: int, seg_size: int):
    """frame_size: actual side length of the frame written to disk.
    seg_size: side length recorded in the annotation's segmentation/bbox
    space. Equal -> a normal, consistent sample. Different -> reproduces
    pointing GraspRegionDataset at a resized frame cache.
    """
    mask = np.zeros((seg_size, seg_size), dtype=np.uint8)
    mask[: seg_size // 2, : seg_size // 2] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))

    doc = {
        "categories": [{"id": 1, "name": "TestTool"}],
        "images": [
            {
                "id": 1,
                "file_name": "CASE001/1.jpg",
                "width": seg_size,
                "height": seg_size,
                "video_name": "CASE001",
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, seg_size // 2, seg_size // 2],
                "segmentation": {
                    "size": [seg_size, seg_size],
                    "counts": rle["counts"].decode("utf-8"),
                },
            }
        ],
    }

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "grasp_short-term_train.json").write_text(json.dumps(doc))

    frames_dir = tmp_path / "frames-001" / "frames" / "CASE001"
    frames_dir.mkdir(parents=True)
    Image.new("RGB", (frame_size, frame_size), color=(100, 150, 200)).save(frames_dir / "1.jpg")

    return GraspRegionDataset(tmp_path, "train", transform=None)


def test_matching_resolution_produces_a_crop(tmp_path):
    ds = _make_dataset(tmp_path, frame_size=8, seg_size=8)
    image, label = ds[0]
    assert label.item() == 0
    assert image.size == (4, 4)  # bbox is seg_size//2 square


def test_mismatched_resolution_fails_loudly(tmp_path):
    ds = _make_dataset(tmp_path, frame_size=4, seg_size=8)
    with pytest.raises(ValueError, match="native resolution|GraspRegionDataset needs"):
        ds[0]
