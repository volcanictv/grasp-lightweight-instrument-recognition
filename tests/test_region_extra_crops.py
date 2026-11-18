import pickle

import numpy as np

from surgical_ai.data.region_dataset import GraspRegionDataset
from test_region_dataset import _make_dataset


def test_extra_crops_follow_the_real_instances_padded_and_labelled(tmp_path):
    _make_dataset(tmp_path, frame_size=8, seg_size=8)
    extra = {"crops": [np.full((4, 8, 3), 200, np.uint8), np.full((6, 6, 3), 50, np.uint8)], "y": [0, 0]}
    path = tmp_path / "extra.pkl"
    path.write_bytes(pickle.dumps(extra))
    ds = GraspRegionDataset(tmp_path, "train", transform=None, letterbox=True, extra_crops_path=path)
    assert len(ds) == 3 and ds.extra_labels == [0, 0] and len(ds.instances) == 1
    image, label = ds[1]
    assert label.item() == 0 and image.size == (8, 8)  # the 4 x 8 crop is padded to a square
    assert (np.array(image) == 0).any() and (np.array(image) == 200).any()
    image, _ = ds[2]
    assert image.size == (6, 6)


def test_without_extra_crops_the_dataset_is_unchanged(tmp_path):
    ds = _make_dataset(tmp_path, frame_size=8, seg_size=8)
    assert len(ds) == 1 and ds.extra_crops == [] and ds.extra_labels == []
