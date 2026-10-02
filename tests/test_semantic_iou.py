import numpy as np
import pytest

from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_higher_score_wins_an_overlap():
    a, b = box(10, 10, 0, 6, 0, 6), box(10, 10, 3, 9, 3, 9)
    sem = paint((10, 10), [(a, 1, 0.9), (b, 2, 0.5)])
    assert sem[4, 4] == 1 and sem[8, 8] == 2 and sem[0, 0] == 1 and sem[9, 0] == 0
    sem = paint((10, 10), [(a, 1, 0.1), (b, 2, 0.5)])
    assert sem[4, 4] == 2


def test_perfect_prediction_gives_one_everywhere():
    gt = paint((10, 10), [(box(10, 10, 0, 5, 0, 5), 1, 1.0), (box(10, 10, 5, 10, 5, 10), 2, 1.0)])
    result = aggregate([frame_class_ious(gt, gt)])
    assert result["mIoU"] == result["IoU"] == 1.0
    assert result["mcIoU"] == pytest.approx(2 / 7)  # five classes never appear and count as zero


def test_a_false_positive_class_hurts_IoU_but_not_mIoU():
    gt = paint((10, 10), [(box(10, 10, 0, 5, 0, 5), 1, 1.0)])
    pred = paint((10, 10), [(box(10, 10, 0, 5, 0, 5), 1, 1.0), (box(10, 10, 6, 9, 6, 9), 3, 0.9)])
    ious, gt_classes = frame_class_ious(pred, gt)
    assert ious == {1: 1.0, 3: 0.0} and gt_classes == {1}
    result = aggregate([(ious, gt_classes)])
    assert result["mIoU"] == 1.0 and result["IoU"] == 0.5


def test_wrong_class_is_a_miss_for_both_classes():
    gt = paint((10, 10), [(box(10, 10, 0, 5, 0, 5), 1, 1.0)])
    pred = paint((10, 10), [(box(10, 10, 0, 5, 0, 5), 2, 1.0)])
    ious, gt_classes = frame_class_ious(pred, gt)
    assert ious == {1: 0.0, 2: 0.0}
    assert aggregate([(ious, gt_classes)])["mIoU"] == 0.0


def test_mcIoU_averages_per_class_over_frames_where_the_class_appears():
    f1 = ({1: 1.0}, {1})
    f2 = ({1: 0.0, 2: 1.0}, {1, 2})
    result = aggregate([f1, f2])
    assert result["per_class_iou"][1] == 0.5 and result["per_class_iou"][2] == 1.0
    assert result["mcIoU"] == pytest.approx((0.5 + 1.0) / 7)
