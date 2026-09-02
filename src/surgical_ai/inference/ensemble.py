"""Weighted box+mask fusion across multiple Mask R-CNN checkpoints
(Milestone 9.5 follow-up, docs/DECISIONS.md).

Motivated by having two independently-trained, mutually-clean models on
hand: `instance_segmentation_maskrcnn` (trained on fold2, i.e. never saw
fold1) and `instance_segmentation_maskrcnn_fold2` (trained on fold1,
i.e. never saw fold2). Neither has ever seen any official-test case in
training -- fold1 and fold2 partition the 8 official-train cases exactly,
disjoint from the 5 official-test cases -- so ensembling their
predictions on official test is not leakage, just two independent
models voting on genuinely unseen data.

A simplified Weighted Boxes Fusion (Solovyev et al. 2021): detections
from all models are pooled, greedily clustered by same-class box-IoU
against each cluster's current representative, then each cluster is
collapsed to a single detection via score-weighted averaging of its box
coordinates and mask probability maps (not just picking one model's
mask) -- an unmatched (single-model) detection survives as its own
cluster with a downweighted score, since it only got one model's vote.
"""

from __future__ import annotations

from surgical_ai.inference.tracking import box_iou


def weighted_fusion_merge(
    detections_per_model: list[list[dict]],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """`detections_per_model[i]` is model i's detections for one image,
    each a dict with `box` (list[float] xyxy), `label` (int), `score`
    (float), `mask` (np.ndarray float probability map, pre-threshold, same
    shape across all models/detections for one image).

    Returns a fused detection list, same dict shape. A cluster's score is
    the sum of its members' scores divided by the number of models
    ensembled -- so a detection only one model found tops out at
    `score / n_models`, reflecting partial agreement, while a detection
    every model agrees on can reach its members' average score.
    """
    n_models = len(detections_per_model)
    all_dets = [d for model_dets in detections_per_model for d in model_dets]
    all_dets.sort(key=lambda d: -d["score"])

    clusters: list[list[dict]] = []
    for d in all_dets:
        best_iou, best_cluster = 0.0, None
        for cluster in clusters:
            rep = cluster[0]
            if rep["label"] != d["label"]:
                continue
            iou = box_iou(rep["box"], d["box"])
            if iou > best_iou:
                best_iou, best_cluster = iou, cluster
        if best_iou >= iou_threshold:
            best_cluster.append(d)
        else:
            clusters.append([d])

    merged = []
    for cluster in clusters:
        total_score = sum(d["score"] for d in cluster)
        box = [
            sum(d["box"][i] * d["score"] for d in cluster) / total_score
            for i in range(4)
        ]
        mask = sum(d["mask"] * d["score"] for d in cluster) / total_score
        merged.append(
            {
                "box": box,
                "label": cluster[0]["label"],
                "score": total_score / n_models,
                "mask": mask,
            }
        )
    return merged
