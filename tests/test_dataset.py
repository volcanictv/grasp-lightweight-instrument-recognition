import os
from pathlib import Path

import pytest

from surgical_ai.data.dataset import GraspMultiLabelDataset
from surgical_ai.data.transforms import build_transforms

DATA_ROOT = Path(os.environ.get("GRASP_DATA_ROOT", Path(__file__).resolve().parents[1] / "GraSp"))
pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="GraSP dataset not available")


def test_train_split_sample_count():
    ds = GraspMultiLabelDataset(DATA_ROOT, "train")
    assert len(ds) == 2324


def test_test_split_sample_count():
    ds = GraspMultiLabelDataset(DATA_ROOT, "test")
    assert len(ds) == 1125


def test_label_is_multihot_over_seven_classes():
    ds = GraspMultiLabelDataset(DATA_ROOT, "train")
    assert ds.num_classes == 7
    _, label = ds[0]
    assert label.shape == (7,)
    assert set(label.tolist()) <= {0.0, 1.0}
    assert label.sum() >= 1  # every keyframe has at least one instrument


def test_image_shape_matches_transform_size():
    tf = build_transforms(224, train=False)
    ds = GraspMultiLabelDataset(DATA_ROOT, "test", transform=tf)
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)


def test_most_frames_are_multilabel():
    # Milestone 0 found 94.7% of annotated frames have 2+ instruments across
    # train+test combined; train alone is ~86.7%.
    ds = GraspMultiLabelDataset(DATA_ROOT, "train")
    multi = sum(1 for _, label in ds.samples if label.sum() >= 2)
    assert multi / len(ds) > 0.8
