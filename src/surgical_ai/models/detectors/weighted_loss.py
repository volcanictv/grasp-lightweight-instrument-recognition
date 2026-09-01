"""Class-weighted box-classification loss for torchvision's Faster R-CNN.

torchvision's `RoIHeads` calls a module-level function, `fastrcnn_loss`,
which computes the box classifier's cross-entropy with no `weight` argument
-- every instrument class contributes equally to the loss regardless of how
rare it is. This is the exact same imbalance problem Milestone 5 fixed for
Task A (Clip Applier: 64/6170 train instances, a 25.6x gap vs. the most
common class) via a weighted BCE `pos_weight`, and Task B via a weighted
`CrossEntropyLoss` -- but it was never applied to the detector, a real gap
this project's own precedent should have caught earlier.

torchvision does not expose a `class_weight` constructor argument on
FasterRCNN/RoIHeads for this loss, so there is no clean subclassing point
short of copying the whole (large) `RoIHeads.forward` method. `RoIHeads.forward`
calls `fastrcnn_loss` as a bare name, resolved from the module's global
namespace at call time (confirmed by inspecting the source), so replacing
the module attribute before training affects every subsequent call without
needing to touch `RoIHeads` at all. This is a real fragility: it depends on
torchvision's internal implementation not changing this call structure.
Pinned torchvision version is already recorded per-run in every manifest,
so a future torchvision upgrade breaking this silently (falling back to
unweighted loss) would at least be traceable to a specific version bump.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models.detection.roi_heads as _roi_heads


def _weighted_fastrcnn_loss(class_weight: torch.Tensor):
    def loss_fn(class_logits, box_regression, labels, regression_targets):
        labels_cat = torch.cat(labels, dim=0)
        regression_targets_cat = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(
            class_logits, labels_cat, weight=class_weight.to(class_logits.device)
        )

        sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
        labels_pos = labels_cat[sampled_pos_inds_subset]
        n, _ = class_logits.shape
        box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets_cat[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels_cat.numel()
        return classification_loss, box_loss

    return loss_fn


def apply_class_weighted_detection_loss(class_weight: torch.Tensor) -> None:
    """Monkeypatches torchvision's box-classification loss to use
    `class_weight` (length num_classes + 1, index 0 = background). Call once
    before training; affects every FasterRCNN instance process-wide (the
    patch is on the shared module, not per-model), which is fine for this
    project's one-run-per-process training scripts but would need undoing
    (`_roi_heads.fastrcnn_loss = _roi_heads._original_fastrcnn_loss`, if ever
    running two differently-weighted detectors in one process.
    """
    _roi_heads.fastrcnn_loss = _weighted_fastrcnn_loss(class_weight)
