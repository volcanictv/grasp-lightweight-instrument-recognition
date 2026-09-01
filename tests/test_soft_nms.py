import numpy as np

from surgical_ai.evaluation.detection import apply_soft_nms_to_predictions, soft_nms_scores


def test_non_overlapping_boxes_scores_unchanged():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    scores = np.array([0.9, 0.8])
    adjusted = soft_nms_scores(boxes, scores, sigma=0.5)
    np.testing.assert_allclose(adjusted, scores)


def test_overlapping_lower_score_decayed_not_removed():
    # second box heavily overlaps the first (same object, hard-NMS would
    # normally drop it entirely at IoU > ~0.5); soft-NMS should shrink its
    # score but not zero it out.
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
    scores = np.array([0.9, 0.85])
    adjusted = soft_nms_scores(boxes, scores, sigma=0.5)
    assert adjusted[0] == 0.9  # highest-scored box is never decayed
    assert 0.0 < adjusted[1] < 0.85  # decayed, but still present


def test_apply_soft_nms_groups_by_image_and_category_independently():
    # Same overlapping geometry, but across two different (image, category)
    # groups -- neither should affect the other.
    predictions = [
        {"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 0, "category_id": 1, "bbox": [1, 1, 10, 10], "score": 0.85},
        {"image_id": 0, "category_id": 2, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 0, "category_id": 2, "bbox": [1, 1, 10, 10], "score": 0.85},
        {"image_id": 1, "category_id": 1, "bbox": [500, 500, 10, 10], "score": 0.6},
    ]
    result = apply_soft_nms_to_predictions(predictions, sigma=0.5, score_threshold=0.01)
    # nothing dropped (all scores stay above the low threshold used here)
    assert len(result) == 5
    # the two (image=0, category=1) boxes decayed independently of the
    # (image=0, category=2) pair, which has identical geometry
    cat1 = sorted(p["score"] for p in result if p["category_id"] == 1 and p["image_id"] == 0)
    cat2 = sorted(p["score"] for p in result if p["category_id"] == 2 and p["image_id"] == 0)
    assert cat1 == cat2  # identical inputs per group -> identical outputs
    # the isolated far-away box (image 1) is untouched
    assert any(p["image_id"] == 1 and p["score"] == 0.6 for p in result)


def test_score_threshold_drops_heavily_decayed_boxes():
    predictions = [
        {"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5},  # identical box, IoU=1
    ]
    result = apply_soft_nms_to_predictions(predictions, sigma=0.5, score_threshold=0.3)
    # IoU=1.0 -> decay factor exp(-1/0.5) ~= 0.135 -> 0.5*0.135 ~= 0.068, below threshold
    assert len(result) == 1
    assert result[0]["score"] == 0.9
