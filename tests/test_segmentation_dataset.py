from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from surgical_ai.data.segmentation_dataset import GraspSegmentationDataset, collate_fn


def _rle_square(size: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> dict:
    m = np.zeros(size, dtype=np.uint8, order="F")
    m[y0:y1, x0:x1] = 1
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _write_fake_split(tmp_path: Path) -> Path:
    data_root = tmp_path / "GraSP"
    frames_dir = data_root / "frames-001" / "frames" / "CASE001"
    frames_dir.mkdir(parents=True)
    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True)

    height, width = 100, 100
    Image.fromarray(np.full((height, width, 3), 80, dtype=np.uint8)).save(frames_dir / "000000001.jpg")

    doc = {
        "images": [{"id": 1, "file_name": "CASE001/000000001.jpg", "width": width, "height": height, "video_name": "CASE001"}],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20],
                "segmentation": _rle_square((height, width), 10, 30, 10, 30),
            }
        ],
        "categories": [{"id": 1, "name": "Bipolar Forceps"}, {"id": 2, "name": "Clip Applier"}],
    }
    import json

    (ann_dir / "grasp_short-term_train.json").write_text(json.dumps(doc))
    (ann_dir / "grasp_short-term_test.json").write_text(json.dumps({**doc, "images": [], "annotations": []}))
    return data_root


def test_dataset_produces_consistent_shapes(tmp_path):
    data_root = _write_fake_split(tmp_path)
    ds = GraspSegmentationDataset(data_root, "train", image_size=64, output_stride=4)
    assert len(ds) == 1
    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    assert target["semantic"].shape == (16, 16)
    assert target["heatmap"].shape == (2, 16, 16)
    assert target["offset"].shape == (2, 16, 16)
    assert target["instance_masks"].shape == (1, 64, 64)
    assert target["instance_labels"].tolist() == [0]  # category_id 1 -> index 0


def test_collate_fn_batches_fixed_size_and_keeps_instance_lists(tmp_path):
    data_root = _write_fake_split(tmp_path)
    ds = GraspSegmentationDataset(data_root, "train", image_size=64, output_stride=4)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    assert images.shape == (1, 3, 64, 64)
    assert targets["heatmap"].shape == (1, 2, 16, 16)
    assert len(targets["instance_masks"]) == 1
