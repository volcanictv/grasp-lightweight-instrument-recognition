import os
from pathlib import Path

import pytest

from surgical_ai.data import splits

DATA_ROOT = Path(os.environ.get("GRASP_DATA_ROOT", Path(__file__).resolve().parents[1] / "GraSp"))
pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="GraSP dataset not available")


def test_official_case_level_split_no_overlap():
    train = splits.load_short_term(DATA_ROOT, "train")
    test = splits.load_short_term(DATA_ROOT, "test")
    train_cases = set(splits.case_list(train))
    test_cases = set(splits.case_list(test))
    assert train_cases.isdisjoint(test_cases)
    assert len(train_cases) == 8
    assert len(test_cases) == 5


def test_folds_partition_train_cases():
    train = splits.load_short_term(DATA_ROOT, "train")
    fold1 = splits.load_short_term(DATA_ROOT, "fold1")
    fold2 = splits.load_short_term(DATA_ROOT, "fold2")
    fold1_cases = set(splits.case_list(fold1))
    fold2_cases = set(splits.case_list(fold2))
    assert fold1_cases.isdisjoint(fold2_cases)
    assert fold1_cases | fold2_cases == set(splits.case_list(train))


def test_unknown_split_fails_loudly():
    with pytest.raises(ValueError):
        splits.load_short_term(DATA_ROOT, "not_a_real_split")
