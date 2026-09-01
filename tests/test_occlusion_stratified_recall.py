from types import SimpleNamespace

from surgical_ai.evaluation.detection import (
    compute_occlusion_fractions,
    evaluate_occlusion_stratified_recall,
)


def _make_dataset():
    # frame 0: two boxes, heavily overlapping (ann 2 is 100% inside ann 1's area-normalized region)
    # frame 1: one isolated box (ann 3), no overlap
    samples = [
        (
            "f0.jpg",
            [
                {"id": 1, "category_id": 10, "bbox": [0, 0, 10, 10]},
                {"id": 2, "category_id": 20, "bbox": [5, 5, 10, 10]},  # overlaps ann 1 by 25/100=0.25 of its own area
            ],
        ),
        (
            "f1.jpg",
            [
                {"id": 3, "category_id": 10, "bbox": [100, 100, 10, 10]},
            ],
        ),
    ]
    id_to_index = {10: 0, 20: 1}
    return SimpleNamespace(samples=samples, _id_to_index=id_to_index)


def test_occlusion_fractions_isolated_and_overlapping():
    ds = _make_dataset()
    fractions = compute_occlusion_fractions(ds)
    assert fractions[3] == 0.0  # isolated instance
    assert fractions[1] > 0.0  # ann 1 overlapped by ann 2
    assert fractions[2] > 0.0  # ann 2 overlapped by ann 1


def test_stratified_recall_counts_and_matches():
    ds = _make_dataset()
    fractions = compute_occlusion_fractions(ds)

    # Predictions: correctly detect ann 1 (category 10 -> index 0 -> label 1)
    # and ann 3 (also label 1), miss ann 2 (category 20 -> label 2) entirely.
    predictions = [
        {"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10], "score": 0.9},
    ]

    result = evaluate_occlusion_stratified_recall(ds, predictions, fractions)
    assert result.n_isolated == 1
    assert result.recall_isolated == 1.0
    # ann1 and ann2 each overlap the other by 25/100 = 0.25 of their own
    # area -- both land in "light" (<=50% covered), one of the two detected.
    assert result.n_light == 2
    assert result.n_heavy == 0
    assert result.recall_light == 0.5
