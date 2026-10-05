"""The three semantic IoUs the GraSP and EndoVis instrument-segmentation benchmarks report (mIoU, IoU, mcIoU), ported from
the evaluation code of MATIS / TAPIS (BCV-Uniandes/MATIS, matis/evaluate/compute_all_iou.py) so our numbers are computed the
way the published ones are.

Per frame, a semantic map is painted from the predicted instances (class 1..7, lower score first so a higher score wins an
overlap) and compared with the ground-truth semantic map. For every class present in the ground truth or the prediction the
frame gets one IoU. Then:

    mIoU    mean over frames of the mean IoU over the classes present in the ground truth ("challenge" IoU)
    IoU     mean over frames of the mean IoU over the classes present in ground truth or prediction, so a predicted class that
            is absent from the frame counts as a zero
    mcIoU   per class, the mean over the frames where it is present in either map; then the mean over all 7 classes
            (a class that never appears counts 0)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_CLASSES = 7


def paint(shape: tuple[int, int], instances: Sequence[tuple[np.ndarray, int, float]]) -> np.ndarray:
    """instances: (bool mask, class 1..7, score). Returns an int8 map, 0 = background; the highest score is painted last."""
    sem = np.zeros(shape, dtype=np.int8)
    for mask, cls, _score in sorted(instances, key=lambda inst: inst[2]):
        sem[mask] = cls
    return sem


def frame_class_ious(pred: np.ndarray, gt: np.ndarray) -> tuple[dict[int, float], set[int]]:
    """IoU for every class present in the ground truth or the prediction, and the set of ground-truth classes."""
    gt_classes = set(np.unique(gt).tolist()) - {0}
    pred_classes = set(np.unique(pred).tolist()) - {0}
    ious: dict[int, float] = {}
    for label in sorted(gt_classes | pred_classes):
        p, g = pred == label, gt == label
        inter = int(np.logical_and(p, g).sum())
        union = int(p.sum()) + int(g.sum()) - inter
        ious[label] = inter / union
    return ious, gt_classes


def aggregate(frames: Sequence[tuple[dict[int, float], set[int]]]) -> dict:
    ious, gt_ious = [], []
    per_class: dict[int, list[float]] = {c: [] for c in range(1, N_CLASSES + 1)}
    for class_ious, gt_classes in frames:
        if class_ious:
            ious.append(float(np.mean(list(class_ious.values()))))
        if gt_classes:
            gt_ious.append(float(np.mean([class_ious[c] for c in gt_classes])))
        for label, value in class_ious.items():
            per_class[label].append(value)
    per_class_mean = {c: (float(np.mean(v)) if v else 0.0) for c, v in per_class.items()}
    return {"mIoU": float(np.mean(gt_ious)), "IoU": float(np.mean(ious)), "mcIoU": float(np.mean(list(per_class_mean.values()))),
            "per_class_iou": per_class_mean, "n_frames": len(frames)}
