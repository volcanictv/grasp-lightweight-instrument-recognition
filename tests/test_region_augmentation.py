import json
import random

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from surgical_ai.data.region_dataset import GraspRegionDataset, instance_key, load_neighbour_meta, perturb_mask


def blob(side=120):
    mask = np.zeros((side, side), dtype=bool)
    mask[30:90, 20:100] = True
    return mask


def test_instance_key_is_file_and_box():
    assert instance_key("CASE001/1.jpg", (1, 2, 3, 4)) == "CASE001/1.jpg|1,2,3,4"


def test_perturb_mask_keeps_shape_dtype_and_never_empties():
    rng = random.Random(0)
    for _ in range(60):
        out = perturb_mask(blob(), rng)
        assert out.shape == (120, 120) and out.dtype == bool and out.any()


def test_each_corruption_changes_the_area_in_its_direction():
    areas = {"dilate": [], "erode": [], "truncate": []}
    base = blob().sum()
    for seed in range(200):
        out = perturb_mask(blob(), random.Random(seed))
        areas["dilate" if out.sum() > base else "erode" if out.sum() < base else "same"] = areas.get("same", [])
        if out.sum() > base:
            assert (out | blob()).sum() == out.sum()  # a dilation contains the original
        elif out.sum() < base:
            assert (out & ~blob()).sum() == 0  # erosion and truncation only remove pixels
    assert any(perturb_mask(blob(), random.Random(s)).sum() > base for s in range(50))
    assert any(perturb_mask(blob(), random.Random(s)).sum() < base for s in range(50))


def test_tiny_masks_are_left_alone():
    tiny = np.zeros((20, 20), dtype=bool)
    tiny[5:8, 5:8] = True
    assert (perturb_mask(tiny, random.Random(1)) == tiny).all()


def test_load_neighbour_meta_merges_directories(tmp_path):
    for name, key in (("a", "k1"), ("b", "k2")):
        (tmp_path / name / "crops").mkdir(parents=True)
        (tmp_path / name / "meta.json").write_text(json.dumps({key: [{"file": "x.jpg", "offset": 1, "area_ratio": 1.0}]}))
    merged = load_neighbour_meta([tmp_path / "a", tmp_path / "b"])
    assert set(merged) == {"k1", "k2"} and merged["k1"][0][0] == tmp_path / "a" / "crops" / "x.jpg"


def make_dataset(tmp_path, **kwargs):
    size = 64
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:32, :32] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    doc = {"categories": [{"id": 1, "name": "T"}],
           "images": [{"id": 1, "file_name": "CASE001/1.jpg", "width": size, "height": size, "video_name": "CASE001"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 32, 32],
                            "segmentation": {"size": [size, size], "counts": rle["counts"].decode("utf-8")}}]}
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "grasp_short-term_train.json").write_text(json.dumps(doc))
    frames = tmp_path / "frames-001" / "frames" / "CASE001"
    frames.mkdir(parents=True)
    Image.new("RGB", (size, size), (100, 150, 200)).save(frames / "1.jpg")
    return GraspRegionDataset(tmp_path, "train", transform=None, **kwargs)


def test_neighbour_swap_returns_the_stored_crop_with_the_same_label(tmp_path):
    nb = tmp_path / "nb"
    (nb / "crops").mkdir(parents=True)
    Image.new("RGB", (10, 20), (255, 0, 0)).save(nb / "crops" / "n.jpg")
    (nb / "meta.json").write_text(json.dumps({"CASE001/1.jpg|0,0,32,32": [{"file": "n.jpg", "offset": 1, "area_ratio": 1.0}]}))
    ds = make_dataset(tmp_path, neighbour_dir=nb, neighbour_prob=1.0)
    image, label = ds[0]
    assert label.item() == 0 and image.size == (10, 20)  # the red neighbour crop, not the 32 x 32 frame crop


def test_without_neighbours_or_perturbation_the_item_is_unchanged(tmp_path):
    image, label = make_dataset(tmp_path)[0]
    assert image.size == (32, 32) and label.item() == 0


def test_mask_perturbation_still_yields_a_valid_crop(tmp_path):
    ds = make_dataset(tmp_path, mask_perturb_prob=1.0)
    for _ in range(20):
        image, label = ds[0]
        assert min(image.size) > 0 and label.item() == 0
