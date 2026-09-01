import numpy as np
import pytest

from surgical_ai.evaluation.segmentation import (
    decode_instances,
    evaluate_instance_ap50,
    evaluate_semantic_segmentation,
    mask_iou,
)


def test_mask_iou_identical_masks_is_one():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 2:5] = True
    assert mask_iou(m, m) == 1.0


def test_mask_iou_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:2, 0:2] = True
    b[5:7, 5:7] = True
    assert mask_iou(a, b) == 0.0


def test_evaluate_semantic_segmentation_perfect_prediction():
    gt = np.zeros((10, 10), dtype=np.int64)
    gt[2:5, 2:5] = 1
    result = evaluate_semantic_segmentation([gt], [gt], ["Bipolar Forceps"])
    assert result.per_class_iou["Bipolar Forceps"] == 1.0
    assert result.miou == 1.0


def test_evaluate_semantic_segmentation_no_overlap():
    gt = np.zeros((10, 10), dtype=np.int64)
    gt[0:3, 0:3] = 1
    pred = np.zeros((10, 10), dtype=np.int64)
    pred[7:10, 7:10] = 1
    result = evaluate_semantic_segmentation([pred], [gt], ["Bipolar Forceps"])
    assert result.per_class_iou["Bipolar Forceps"] == 0.0


def test_decode_instances_finds_single_peak():
    h, w, c = 20, 20, 2
    heatmap = np.zeros((c, h, w), dtype=np.float32)
    heatmap[0, 10, 10] = 0.9
    offset = np.zeros((2, h, w), dtype=np.float32)  # zero offset -> pixel votes for itself
    semantic = np.zeros((h, w), dtype=np.int64)
    semantic[8:13, 8:13] = 1  # class 0 (semantic id 1) foreground blob around the peak

    instances = decode_instances(heatmap, offset, semantic, score_threshold=0.1)
    assert len(instances) == 1
    mask, label, score = instances[0]
    assert label == 0
    assert mask.sum() == 25  # the 5x5 foreground blob
    assert score == pytest.approx(0.9, abs=1e-5)


def test_decode_instances_no_peaks_returns_empty():
    h, w, c = 10, 10, 2
    heatmap = np.zeros((c, h, w), dtype=np.float32)
    offset = np.zeros((2, h, w), dtype=np.float32)
    semantic = np.zeros((h, w), dtype=np.int64)
    assert decode_instances(heatmap, offset, semantic, score_threshold=0.1) == []


def test_evaluate_instance_ap50_perfect_match_is_one():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    predictions = [[(mask, 0, 0.9)]]
    ground_truths = [[(mask, 0)]]
    result = evaluate_instance_ap50(predictions, ground_truths, ["Bipolar Forceps"])
    assert result["per_class_ap50"]["Bipolar Forceps"] == 1.0
    assert result["map50"] == 1.0


def test_evaluate_instance_ap50_no_predictions_is_zero():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    predictions = [[]]
    ground_truths = [[(mask, 0)]]
    result = evaluate_instance_ap50(predictions, ground_truths, ["Bipolar Forceps"])
    assert result["per_class_ap50"]["Bipolar Forceps"] == 0.0


def test_evaluate_instance_ap50_no_ground_truth_is_nan():
    mask = np.zeros((10, 10), dtype=bool)
    predictions = [[(mask, 0, 0.5)]]
    ground_truths = [[]]
    result = evaluate_instance_ap50(predictions, ground_truths, ["Bipolar Forceps"])
    assert np.isnan(result["per_class_ap50"]["Bipolar Forceps"])
