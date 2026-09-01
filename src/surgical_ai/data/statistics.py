"""Dataset statistics for GraSP short-term (instrument) annotations.

Pure functions over a parsed short-term JSON doc (see splits.load_short_term),
plus on-disk checks against the frame and segmentation-mask directories.
Milestone 0 consumer: scripts/inspect_dataset.py.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


def category_names(short_term_doc: dict[str, Any]) -> dict[int, str]:
    return {c["id"]: c["name"] for c in short_term_doc["categories"]}


def category_id_to_index(short_term_doc: dict[str, Any]) -> dict[int, int]:
    """Stable category_id -> 0..N-1 index, shared by Task A (multi-hot
    position) and Task B (class label) so both tasks report classes in the
    same order.
    """
    return {cid: i for i, cid in enumerate(sorted(c["id"] for c in short_term_doc["categories"]))}


def basic_counts(short_term_doc: dict[str, Any]) -> dict[str, int]:
    images = short_term_doc["images"]
    annotations = short_term_doc["annotations"]
    annotated_image_ids = {a["image_id"] for a in annotations}
    return {
        "num_cases": len({img["video_name"] for img in images}),
        "num_keyframes": len(images),
        "num_annotated_keyframes": len(annotated_image_ids),
        "num_instances": len(annotations),
    }


def resolution_distribution(short_term_doc: dict[str, Any]) -> Counter:
    return Counter((img["width"], img["height"]) for img in short_term_doc["images"])


def class_instance_histogram(short_term_doc: dict[str, Any]) -> Counter:
    """Instance counts per category (one instrument instance = one annotation)."""
    return Counter(a["category_id"] for a in short_term_doc["annotations"])


def class_frame_presence_histogram(short_term_doc: dict[str, Any]) -> Counter:
    """Frame counts per category (a class counts once per frame regardless of
    how many instances of it appear in that frame). This is the number that
    matters for multi-label classification, where a class is a positive label
    for the frame, not a count.
    """
    frame_classes: dict[int, set[int]] = defaultdict(set)
    for a in short_term_doc["annotations"]:
        frame_classes[a["image_id"]].add(a["category_id"])
    presence = Counter()
    for classes in frame_classes.values():
        presence.update(classes)
    return presence


def per_case_class_histogram(short_term_doc: dict[str, Any]) -> dict[str, Counter]:
    image_id_to_case = {img["id"]: img["video_name"] for img in short_term_doc["images"]}
    result: dict[str, Counter] = defaultdict(Counter)
    for a in short_term_doc["annotations"]:
        case = image_id_to_case[a["image_id"]]
        result[case][a["category_id"]] += 1
    return dict(result)


def cooccurrence_matrix(short_term_doc: dict[str, Any]) -> tuple[list[int], list[list[int]]]:
    """NxN matrix over category ids (sorted). Entry [i][j] = number of frames
    containing both class i and class j (diagonal = frames containing class i,
    any count).
    """
    cat_ids = sorted(c["id"] for c in short_term_doc["categories"])
    index = {cid: i for i, cid in enumerate(cat_ids)}

    frame_classes: dict[int, set[int]] = defaultdict(set)
    for a in short_term_doc["annotations"]:
        frame_classes[a["image_id"]].add(a["category_id"])

    n = len(cat_ids)
    matrix = [[0] * n for _ in range(n)]
    for classes in frame_classes.values():
        classes = list(classes)
        for i in range(len(classes)):
            ci = index[classes[i]]
            matrix[ci][ci] += 1
            for j in range(i + 1, len(classes)):
                cj = index[classes[j]]
                matrix[ci][cj] += 1
                matrix[cj][ci] += 1
    return cat_ids, matrix


def instances_per_frame_distribution(short_term_doc: dict[str, Any]) -> Counter:
    """How many instrument instances appear in a single frame. Confirms
    frames routinely hold 2-3 instruments, per CLAUDE.md task-framing note.
    """
    frame_counts: Counter = Counter()
    counts: dict[int, int] = defaultdict(int)
    for a in short_term_doc["annotations"]:
        counts[a["image_id"]] += 1
    for c in counts.values():
        frame_counts[c] += 1
    return frame_counts


@dataclass
class IntegrityReport:
    missing_frames: list[str] = field(default_factory=list)
    corrupt_frames: list[str] = field(default_factory=list)
    missing_masks: list[str] = field(default_factory=list)
    corrupt_masks: list[str] = field(default_factory=list)
    checked_frames: int = 0
    checked_masks: int = 0

    @property
    def ok(self) -> bool:
        return not (
            self.missing_frames
            or self.corrupt_frames
            or self.missing_masks
            or self.corrupt_masks
        )


def mask_path(seg_root: Path, split: str, image_name: str) -> Path:
    # Train-fold case dirs sit directly under seg_root; test-split case dirs
    # sit under seg_root/test/. png replaces jpg.
    stem = Path(image_name).with_suffix(".png")
    if split == "test":
        return seg_root / "test" / stem
    return seg_root / stem


def check_integrity(
    short_term_doc: dict[str, Any],
    frames_root: Path,
    seg_root: Path,
    split: str,
    verify_decode: bool = True,
) -> IntegrityReport:
    """Check that every annotated keyframe has a frame file and a mask file
    on disk, and that both decode cleanly.
    """
    report = IntegrityReport()
    annotated_image_ids = {a["image_id"] for a in short_term_doc["annotations"]}
    images_by_id = {img["id"]: img for img in short_term_doc["images"]}

    for image_id in annotated_image_ids:
        img_meta = images_by_id[image_id]
        image_name = img_meta["file_name"]

        frame_path = frames_root / image_name
        report.checked_frames += 1
        if not frame_path.exists():
            report.missing_frames.append(image_name)
        elif verify_decode:
            try:
                with Image.open(frame_path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError):
                report.corrupt_frames.append(image_name)

        mask_file = mask_path(seg_root, split, image_name)
        report.checked_masks += 1
        if not mask_file.exists():
            report.missing_masks.append(image_name)
        elif verify_decode:
            try:
                with Image.open(mask_file) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError):
                report.corrupt_masks.append(image_name)

    return report
