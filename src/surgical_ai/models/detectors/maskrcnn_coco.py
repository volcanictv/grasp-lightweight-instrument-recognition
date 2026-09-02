from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from surgical_ai.models.detectors.registry import register_detector


@register_detector("maskrcnn_resnet50_coco")
def build_maskrcnn_resnet50_coco(num_classes: int, pretrained: bool) -> nn.Module:
    """COCO-pretrained Mask R-CNN (ResNet-50-FPN, v2 head) -- accuracy-first
    priority (docs/DECISIONS.md, 2026-09-01), not a lightweight candidate.
    Unlike `maskrcnn_mobilenet_v3`'s warm start (box/RPN/box-head only, mask
    head untrained), every component here -- backbone, RPN, box head, AND
    mask head -- starts pretrained on real instance segmentation (COCO),
    not just ImageNet classification. This is a fundamentally different risk
    profile than the earlier from-scratch-headed ResNet-50 box detector that
    overfit GraSP's 8 training cases (docs/DECISIONS.md, Milestone 8): fine-
    tuning an already-solved detection+segmentation problem needs far less
    adaptation than learning it from scratch.

    `pretrained=False` here still loads COCO weights for the box/RPN/backbone
    (there is no meaningful "untrained ResNet-50-FPN-v2" starting point worth
    using instead) -- only the final classification and mask-predictor layers
    are replaced and re-initialized for GraSP's 7 classes.
    """
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes + 1)

    return model
