import torch
import torchvision.models.detection.roi_heads as roi_heads

from surgical_ai.models.detectors.weighted_loss import apply_class_weighted_detection_loss


def test_weighted_loss_changes_classification_loss_value(monkeypatch):
    original_loss_fn = roi_heads.fastrcnn_loss
    # Registers the current value for automatic restore at teardown -- this
    # patches a shared module-level function and must not leak into other
    # tests, even though apply_class_weighted_detection_loss below mutates
    # it directly rather than through monkeypatch.
    monkeypatch.setattr(roi_heads, "fastrcnn_loss", original_loss_fn)

    num_classes_incl_bg = 3  # background + 2 instrument classes
    class_logits = torch.randn(6, num_classes_incl_bg)
    box_regression = torch.randn(6, num_classes_incl_bg * 4)
    labels = [torch.tensor([0, 1, 1, 2, 0, 1])]
    regression_targets = [torch.randn(6, 4)]

    unweighted_cls_loss, _ = original_loss_fn(class_logits, box_regression, labels, regression_targets)

    class_weight = torch.tensor([1.0, 10.0, 0.1])  # heavily favor class 1 in the loss
    apply_class_weighted_detection_loss(class_weight)
    weighted_cls_loss, _ = roi_heads.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

    assert not torch.isclose(weighted_cls_loss, unweighted_cls_loss)


def test_weighted_loss_matches_manual_weighted_cross_entropy(monkeypatch):
    monkeypatch.setattr(roi_heads, "fastrcnn_loss", roi_heads.fastrcnn_loss)  # register for restore

    class_logits = torch.randn(4, 3)
    box_regression = torch.zeros(4, 12)
    labels = [torch.tensor([0, 1, 2, 1])]
    regression_targets = [torch.zeros(4, 4)]
    class_weight = torch.tensor([1.0, 2.0, 0.5])

    apply_class_weighted_detection_loss(class_weight)
    cls_loss, _ = roi_heads.fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

    expected = torch.nn.functional.cross_entropy(class_logits, labels[0], weight=class_weight)
    assert torch.isclose(cls_loss, expected)
