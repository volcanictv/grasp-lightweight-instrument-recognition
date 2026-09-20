import json

import numpy as np
import pytest
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from surgical_ai.data.tight_crop import RawBBoxCropDataset, TightCropDataset, tight_band_crop
from surgical_ai.data.tight_transforms import RandomLighting, RandomPatchMask, build_tight_transforms


def _frame(h=120, w=160):
    rng = np.random.default_rng(0)
    return rng.integers(30, 255, size=(h, w, 3), dtype=np.uint8)


def test_instrument_outnumbers_band_and_band_is_bounded():
    mask = np.zeros((120, 160), dtype=bool)
    mask[40:80, 50:110] = True
    crop, info = tight_band_crop(_frame(), mask, None, band_frac=0.5, r_cap=10)
    assert info["instrument_px"] == mask.sum()
    assert 0 < info["band_px"] < info["instrument_px"]
    assert info["band_px"] <= 0.5 * info["instrument_px"]
    assert info["radius"] <= 10
    assert crop.shape[0] >= 40 and crop.shape[1] >= 60  # at least the instrument's own extent


def test_thin_instrument_shrinks_band_instead_of_breaking_invariant():
    mask = np.zeros((200, 200), dtype=bool)
    mask[100, 20:180] = True  # a 1 px wide line: any wide band would outnumber it
    _, info = tight_band_crop(np.full((200, 200, 3), 200, np.uint8), mask, None, band_frac=0.5, r_cap=24)
    assert info["band_px"] < info["instrument_px"]


def test_other_instruments_are_excluded_from_the_band():
    frame = np.full((100, 100, 3), 255, np.uint8)
    frame[:, 60:] = (255, 0, 0)  # the "other instrument" is pure red
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:55] = True
    other = np.zeros_like(mask)
    other[:, 60:] = True
    crop, _ = tight_band_crop(frame, mask, other, band_frac=0.9, r_cap=20)
    red = (crop[..., 0] == 255) & (crop[..., 1] == 0)
    assert not red.any()


def test_empty_mask_falls_back_without_crashing():
    crop, info = tight_band_crop(_frame(), np.zeros((120, 160), bool), None)
    assert info["empty"] and crop.ndim == 3


def test_band_frac_must_leave_the_instrument_in_the_majority():
    with pytest.raises(ValueError):
        tight_band_crop(_frame(), np.ones((120, 160), bool), None, band_frac=1.0)


def _write_dataset(tmp_path):
    def rle(m):
        r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        return {"size": list(m.shape), "counts": r["counts"].decode("utf-8")}

    a = np.zeros((100, 100), bool)
    a[20:60, 10:40] = True
    b = np.zeros((100, 100), bool)
    b[20:60, 45:80] = True
    doc = {
        "categories": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        "images": [{"id": 1, "file_name": "CASE001/1.jpg", "width": 100, "height": 100, "video_name": "CASE001"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "segmentation": rle(a)},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [45, 20, 35, 40], "segmentation": rle(b)},
        ],
    }
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "grasp_short-term_train.json").write_text(json.dumps(doc))
    frames = tmp_path / "frames-001" / "frames" / "CASE001"
    frames.mkdir(parents=True)
    img = np.zeros((100, 100, 3), np.uint8)
    img[a] = (0, 255, 0)  # instrument A is green
    img[b] = (255, 0, 0)  # instrument B is red
    img[~(a | b)] = (90, 90, 90)  # background is gray
    Image.fromarray(img).save(frames / "1.jpg", quality=100)


def test_dataset_crop_contains_one_instrument_and_some_background(tmp_path):
    _write_dataset(tmp_path)
    ds = TightCropDataset(tmp_path, "train", transform=None, band_frac=0.5, r_cap=8)
    crop, info = ds.crop_and_info(0)
    px = crop.reshape(-1, 3).astype(int)
    green = ((px[:, 1] > 200) & (px[:, 0] < 60)).sum()
    red = ((px[:, 0] > 200) & (px[:, 1] < 60)).sum()
    gray = ((abs(px - 90) < 25).all(axis=1)).sum()
    assert green > 0 and red == 0  # own instrument only, neighbour removed
    assert gray > 0  # some real background kept
    assert green > gray  # instrument outnumbers background
    image, label = ds[0]
    assert image.size[0] == image.size[1] and label.item() == 0


def test_raw_bbox_crop_keeps_original_pixels(tmp_path):
    _write_dataset(tmp_path)
    image, _ = RawBBoxCropDataset(tmp_path, "train", transform=None)[0]
    assert image.size == (40, 40)  # bbox 30x40 padded to a square


def test_lighting_keeps_black_background_black():
    t = torch.zeros(3, 32, 32)
    t[:, 8:24, 8:24] = 0.5
    for _ in range(20):
        out = RandomLighting(p=1.0)(t)
        assert (out[:, :8, :] == 0).all() and out.min() >= 0 and out.max() <= 1


def test_patch_mask_blanks_pixels_and_respects_probability():
    t = torch.ones(3, 64, 64)
    assert RandomPatchMask(p=0.0)(t).equal(t)
    assert (RandomPatchMask(p=1.0)(t) == 0).any()


def test_masklight_pipeline_shapes_and_eval_is_unchanged():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (80, 60, 3), dtype=np.uint8))
    assert build_tight_transforms(224, True, "masklight")(img).shape == (3, 224, 224)
    ev_default = build_tight_transforms(224, False, "masklight")(img)
    ev_plain = build_tight_transforms(224, False, "default")(img)
    assert torch.equal(ev_default, ev_plain)
    with pytest.raises(ValueError):
        build_tight_transforms(224, True, "nonsense")
