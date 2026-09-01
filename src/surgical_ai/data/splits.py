"""Loaders for the official GraSP short-term split JSONs.

No custom splitting logic here. Splits are defined at case level by the
dataset authors; we only parse what they already produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SHORT_TERM_SPLITS = {
    "train": "grasp_short-term_train.json",
    "test": "grasp_short-term_test.json",
    "fold1": "grasp_short-term_fold1.json",
    "fold2": "grasp_short-term_fold2.json",
}


def annotations_dir(data_root: Path) -> Path:
    return data_root / "annotations"


def load_short_term(data_root: Path, split: str) -> dict[str, Any]:
    """Load one short-term annotation file (instruments + atomic actions).

    Fails loudly if the split name isn't one of the official files, per
    CLAUDE.md: never fall back to a random split.
    """
    if split not in SHORT_TERM_SPLITS:
        raise ValueError(
            f"Unknown split '{split}'. Official short-term splits are: "
            f"{sorted(SHORT_TERM_SPLITS)}"
        )
    path = annotations_dir(data_root) / SHORT_TERM_SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def case_list(short_term_doc: dict[str, Any]) -> list[str]:
    return sorted({img["video_name"] for img in short_term_doc["images"]})


# config's data.split value -> (train split name, val split name).
# "official" trains on the 8 train-set cases and validates on the 5 held-out
# test-set cases (the real number). "foldN" is cross-validation: it means
# "hold out foldN for validation, train on the other fold" — not "use
# foldN for training".
_TRAIN_VAL_SPLITS = {
    "official": ("train", "test"),
    "fold1": ("fold2", "fold1"),
    "fold2": ("fold1", "fold2"),
}


def resolve_train_val_split(config_split: str) -> tuple[str, str]:
    if config_split not in _TRAIN_VAL_SPLITS:
        raise ValueError(
            f"Unknown data.split '{config_split}'. Valid: {sorted(_TRAIN_VAL_SPLITS)}"
        )
    return _TRAIN_VAL_SPLITS[config_split]
