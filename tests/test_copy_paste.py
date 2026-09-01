from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from surgical_ai.data.copy_paste import CopyPasteDetectionDataset


class _FakeBaseDataset:
    """Minimal stand-in for GraspDetectionDataset's interface, so this test
    doesn't need a real GraSP data root -- only the fields CopyPasteDetectionDataset
    actually reads: frames_root, samples, _id_to_index, class_names_ordered(), transform.
    """

    def __init__(self, frames_root: Path, samples: list, id_to_index: dict, names: list):
        self.frames_root = frames_root
        self.samples = samples
        self._id_to_index = id_to_index
        self._names = names
        self.transform = None

    def class_names_ordered(self) -> list:
        return self._names

    def __len__(self) -> int:
        return len(self.samples)


def _rle_square(size: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> dict:
    m = np.zeros(size, dtype=np.uint8, order="F")
    m[y0:y1, x0:x1] = 1
    rle = mask_utils.encode(m)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _make_dataset(tmp_path: Path, paste_prob: float, occlusion_bias: float = 0.7) -> CopyPasteDetectionDataset:
    height, width = 40, 40
    frame_a = np.full((height, width, 3), 50, dtype=np.uint8)
    frame_b = np.full((height, width, 3), 200, dtype=np.uint8)  # distinct color: rare-class source
    Image.fromarray(frame_a).save(tmp_path / "frame_a.jpg")
    Image.fromarray(frame_b).save(tmp_path / "frame_b.jpg")

    # frame_a has one real instance (category 1 -> index 0, "Common"), a 10x10 box.
    anns_a = [
        {"bbox": [5, 5, 10, 10], "category_id": 1, "segmentation": _rle_square((height, width), 5, 15, 5, 15)}
    ]
    # frame_b (never used as a destination, only as a copy-paste source) has one
    # rare-class instance (category 2 -> index 1, "Clip Applier"), a 6x6 box.
    anns_b = [
        {"bbox": [20, 20, 6, 6], "category_id": 2, "segmentation": _rle_square((height, width), 20, 26, 20, 26)}
    ]

    base = _FakeBaseDataset(
        frames_root=tmp_path,
        samples=[("frame_a.jpg", anns_a), ("frame_b.jpg", anns_b)],
        id_to_index={1: 0, 2: 1},
        names=["Common", "Clip Applier"],
    )
    return CopyPasteDetectionDataset(
        base, paste_prob=paste_prob, max_pastes=1, rare_classes=["Clip Applier"],
        occlusion_bias=occlusion_bias, seed=0,
    )


def test_zero_paste_prob_leaves_frame_unmodified(tmp_path):
    ds = _make_dataset(tmp_path, paste_prob=0.0)
    image, target = ds[0]
    assert target["boxes"].shape[0] == 1
    assert target["labels"].tolist() == [1]  # original instance only (label 1 = index 0 + 1)


def test_paste_adds_rare_class_instance_and_changes_pixels(tmp_path):
    ds = _make_dataset(tmp_path, paste_prob=1.0)
    image, target = ds[0]
    assert target["boxes"].shape[0] == 2
    # the appended box is the pasted "Clip Applier" instance: index 1 + 1 = label 2
    assert 2 in target["labels"].tolist()
    # the pasted patch's distinct color (200) must appear somewhere in the composed frame
    image_np = np.array(image.permute(1, 2, 0) * 255) if hasattr(image, "permute") else np.array(image)
    assert image_np.max() > 150


def test_bank_only_contains_rare_classes(tmp_path):
    ds = _make_dataset(tmp_path, paste_prob=1.0)
    # category_id 1 ("Common", index 0) must never enter the paste bank
    assert 0 not in ds.bank
    assert 1 in ds.bank  # "Clip Applier" index


def test_occlusion_bias_places_paste_near_existing_box(tmp_path):
    ds = _make_dataset(tmp_path, paste_prob=1.0, occlusion_bias=1.0)
    _, target = ds[0]
    boxes = target["boxes"].tolist()
    existing = boxes[0]
    pasted = boxes[1]
    ex_cx, ex_cy = (existing[0] + existing[2]) / 2, (existing[1] + existing[3]) / 2
    p_cx, p_cy = (pasted[0] + pasted[2]) / 2, (pasted[1] + pasted[3]) / 2
    # with occlusion_bias=1.0 the paste center is jittered around the existing
    # box's center (+/- 0.3 * pasted size), so it should land close by, not
    # scattered anywhere in the 40x40 frame.
    assert abs(p_cx - ex_cx) < 15
    assert abs(p_cy - ex_cy) < 15
