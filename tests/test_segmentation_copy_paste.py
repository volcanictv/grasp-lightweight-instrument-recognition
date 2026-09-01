from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from surgical_ai.data.segmentation_copy_paste import SegmentationCopyPasteDataset


class _FakeSegBaseDataset:
    def __init__(self, frames_root, samples, id_to_index, names, image_size=64, output_stride=4):
        self.frames_root = frames_root
        self.samples = samples
        self._id_to_index = id_to_index
        self._names = names
        self.image_size = image_size
        self.output_stride = output_stride

    @property
    def num_classes(self):
        return len(self._names)

    def class_names_ordered(self):
        return self._names

    def __len__(self):
        return len(self.samples)


def _rle_square(size, y0, y1, x0, x1):
    m = np.zeros(size, dtype=np.uint8, order="F")
    m[y0:y1, x0:x1] = 1
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _make_dataset(tmp_path: Path, paste_prob: float) -> SegmentationCopyPasteDataset:
    height, width = 100, 100
    frame_a = np.full((height, width, 3), 50, dtype=np.uint8)
    frame_b = np.full((height, width, 3), 200, dtype=np.uint8)
    Image.fromarray(frame_a).save(tmp_path / "frame_a.jpg")
    Image.fromarray(frame_b).save(tmp_path / "frame_b.jpg")

    anns_a = [
        {"bbox": [10, 10, 20, 20], "category_id": 1, "segmentation": _rle_square((height, width), 10, 30, 10, 30)}
    ]
    anns_b = [
        {"bbox": [60, 60, 12, 12], "category_id": 2, "segmentation": _rle_square((height, width), 60, 72, 60, 72)}
    ]

    base = _FakeSegBaseDataset(
        frames_root=tmp_path,
        samples=[("frame_a.jpg", anns_a), ("frame_b.jpg", anns_b)],
        id_to_index={1: 0, 2: 1},
        names=["Common", "Clip Applier"],
    )
    return SegmentationCopyPasteDataset(
        base, paste_prob=paste_prob, max_pastes=1, rare_classes=["Clip Applier"], occlusion_bias=0.7, seed=0
    )


def test_zero_paste_prob_leaves_one_real_instance(tmp_path):
    ds = _make_dataset(tmp_path, paste_prob=0.0)
    _, target = ds[0]
    assert target["instance_labels"].tolist() == [0]
    assert target["instance_masks"].shape[0] == 1


def test_paste_adds_second_instance_with_full_targets(tmp_path):
    ds = _make_dataset(tmp_path, paste_prob=1.0)
    image, target = ds[0]
    assert target["instance_labels"].tolist() == [0, 1]
    assert target["instance_masks"].shape[0] == 2

    out_hw = ds.base.image_size // ds.base.output_stride
    assert target["semantic"].shape == (out_hw, out_hw)
    assert target["heatmap"].shape == (2, out_hw, out_hw)
    # both classes should have a nonzero heatmap peak now that both instances exist
    assert target["heatmap"][0].max() > 0
    assert target["heatmap"][1].max() > 0


def test_bank_only_contains_rare_classes(tmp_path):
    ds = _make_dataset(tmp_path, paste_prob=1.0)
    assert 0 not in ds.bank
    assert 1 in ds.bank
