import torch

from surgical_ai.training.segmentation_losses import centroid_focal_loss, compute_segmentation_loss, offset_l1_loss


def test_centroid_focal_loss_is_near_zero_for_perfect_prediction():
    target = torch.zeros(1, 2, 8, 8)
    target[0, 0, 4, 4] = 1.0
    logits = torch.full((1, 2, 8, 8), -10.0)
    logits[0, 0, 4, 4] = 10.0  # confident correct positive prediction
    loss = centroid_focal_loss(logits, target)
    assert loss.item() < 0.01


def test_centroid_focal_loss_penalizes_wrong_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    bad_logits = torch.full((1, 1, 8, 8), -10.0)  # predicts background everywhere, including the peak
    good_logits = bad_logits.clone()
    good_logits[0, 0, 4, 4] = 10.0
    assert centroid_focal_loss(bad_logits, target) > centroid_focal_loss(good_logits, target)


def test_offset_l1_loss_ignores_unmasked_pixels():
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[0, :, 0, 0] = 100.0  # huge error, but outside the mask
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert offset_l1_loss(pred, target, mask).item() == 0.0


def test_offset_l1_loss_masked_region_matches_manual_l1():
    pred = torch.zeros(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[0, :, 1, 1] = 2.0
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, 1, 1] = True
    # 2 channels both off by 2.0 at the one masked pixel -> mean abs error = 2.0
    assert offset_l1_loss(pred, target, mask).item() == 2.0


def test_compute_segmentation_loss_returns_all_components():
    predictions = {
        "heatmap": torch.randn(1, 2, 8, 8, requires_grad=True),
        "offset": torch.randn(1, 2, 8, 8, requires_grad=True),
        "semantic": torch.randn(1, 3, 8, 8, requires_grad=True),
    }
    targets = {
        "heatmap": torch.zeros(1, 2, 8, 8),
        "offset": torch.zeros(1, 2, 8, 8),
        "offset_mask": torch.zeros(1, 8, 8, dtype=torch.bool),
        "semantic": torch.zeros(1, 8, 8, dtype=torch.int64),
    }
    result = compute_segmentation_loss(predictions, targets)
    assert set(result.keys()) == {"total", "heatmap", "offset", "semantic"}
    assert result["total"].requires_grad
    assert torch.isfinite(result["total"])
