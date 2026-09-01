"""Instance segmentation evaluation (Milestone 9).

Covers three metric families, all deliberately kept because they answer
different questions and are each already promised somewhere in this repo:

  - IoU / Dice / mIoU / per-class IoU: PROJECT_SPEC's required segmentation
    metrics. Semantic-level (per-pixel class agreement), no instance
    separation needed.
  - AP50_segm / mcIoU: the metric family every third-party GraSP paper
    (TAPIS, LACOSTE) actually reports (docs/findings.md) -- needed for a
    literature-comparable number. mcIoU (LACOSTE's term) is the same
    computation as mIoU, reported once under both names rather than
    duplicated.
  - Occlusion-stratified recall: this project's own diagnostic (Milestone
    8), adapted from box-IoU to mask-IoU matching, to check whether this
    architecture actually closes the occlusion gap it was built for.

Instance decoding ("mask then classify", Kurmann et al. 2021) groups
foreground pixels by nearest *predicted* centroid (offset head), not by
which heatmap channel found that centroid -- classification is a separate,
later step (majority vote from the semantic head over each group's
pixels). That order is the whole point of "mask then classify": grouping
must not depend on a possibly-wrong per-channel class guess.

All predicted/ground-truth masks compared here are at the model's output
stride (e.g. 96x96 for a 384 input at stride 4), not upsampled to input
resolution -- ground-truth masks are downsampled the same way training
targets are (`data/segmentation_targets.py::downsample_mask_nearest`), so
comparisons are apples-to-apples. This is a coarser number than a
full-resolution mask IoU would give and should be noted as such wherever
it's reported; upsampling predictions for a final, reportable number is
follow-up work, not done here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.segmentation_targets import downsample_mask_nearest
from surgical_ai.evaluation.detection import (
    _HEAVY_OCCLUSION_THRESHOLD,
    _LIGHT_OCCLUSION_THRESHOLD,
    OcclusionStratifiedRecall,
)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


@dataclass
class SemanticSegmentationMetrics:
    per_class_iou: dict[str, float]
    per_class_dice: dict[str, float]
    miou: float
    mean_dice: float

    def to_markdown(self) -> str:
        lines = ["| class | IoU | Dice |", "|---|---|---|"]
        for name in self.per_class_iou:
            lines.append(f"| {name} | {self.per_class_iou[name]:.3f} | {self.per_class_dice[name]:.3f} |")
        lines.append(f"| **mIoU / mean Dice (== mcIoU)** | {self.miou:.3f} | {self.mean_dice:.3f} |")
        return "\n".join(lines)


def evaluate_semantic_segmentation(
    pred_semantic: list[np.ndarray], gt_semantic: list[np.ndarray], class_names: list[str]
) -> SemanticSegmentationMetrics:
    """Both lists are per-image (H, W) integer maps: 0 = background,
    1..num_classes = the same category order as `class_names`. Confusion
    counts accumulate across the whole dataset before dividing, not
    averaged per-image, so a class absent from one frame doesn't distort
    its score.
    """
    num_classes = len(class_names)
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    pred_area = np.zeros(num_classes)
    gt_area = np.zeros(num_classes)

    for pred, gt in zip(pred_semantic, gt_semantic):
        for c in range(num_classes):
            p, g = pred == (c + 1), gt == (c + 1)
            intersection[c] += np.logical_and(p, g).sum()
            union[c] += np.logical_or(p, g).sum()
            pred_area[c] += p.sum()
            gt_area[c] += g.sum()

    per_class_iou, per_class_dice = {}, {}
    for c, name in enumerate(class_names):
        per_class_iou[name] = float(intersection[c] / union[c]) if union[c] > 0 else float("nan")
        denom = pred_area[c] + gt_area[c]
        per_class_dice[name] = float(2 * intersection[c] / denom) if denom > 0 else float("nan")

    valid_iou = [v for v in per_class_iou.values() if not np.isnan(v)]
    valid_dice = [v for v in per_class_dice.values() if not np.isnan(v)]
    return SemanticSegmentationMetrics(
        per_class_iou=per_class_iou,
        per_class_dice=per_class_dice,
        miou=float(np.mean(valid_iou)) if valid_iou else float("nan"),
        mean_dice=float(np.mean(valid_dice)) if valid_dice else float("nan"),
    )


def decode_instances(
    heatmap: np.ndarray,
    offset: np.ndarray,
    semantic: np.ndarray,
    score_threshold: float = 0.3,
    nms_kernel: int = 7,
) -> list[tuple[np.ndarray, int, float]]:
    """`heatmap`: (C, H, W) sigmoid probabilities (already sigmoid'd, not
    logits). `offset`: (2, H, W) raw pixel offsets, channel 0 = x, channel
    1 = y (matching data/segmentation_targets.py's convention). `semantic`:
    (H, W) integer argmax of the semantic head, 0 = background.

    Defaults (`score_threshold=0.3`, `nms_kernel=7`) are tuned for this
    project's actual output resolution (96x96 for a 384 input, much coarser
    than COCO-scale CenterNet's typical 128x128+), not copied from
    CenterNet's own defaults (`score_threshold~0.1`-ish, `nms_kernel=3`) --
    those produced 10-20x too many spurious local-maxima "peaks" per real
    instance on this project's first trained checkpoint (docs/DECISIONS.md),
    because a small kernel doesn't suppress a heatmap that hasn't yet
    trained down to a single sharp unimodal peak per instance. Tuned
    empirically against real validation predictions, not assumed.

    Returns a list of (mask: bool (H, W), label: int 0-indexed, score: float).
    """
    pooled = heatmap.max(axis=0)
    local_max = ndimage.maximum_filter(pooled, size=nms_kernel) == pooled
    candidate = local_max & (pooled >= score_threshold)
    peak_ys, peak_xs = np.nonzero(candidate)

    fg_mask = semantic > 0
    if len(peak_ys) == 0 or not fg_mask.any():
        return []

    fg_ys, fg_xs = np.nonzero(fg_mask)
    pred_cy = fg_ys + offset[1][fg_ys, fg_xs]
    pred_cx = fg_xs + offset[0][fg_ys, fg_xs]
    pred_centers = np.stack([pred_cy, pred_cx], axis=1)
    peak_coords = np.stack([peak_ys.astype(float), peak_xs.astype(float)], axis=1)

    dists = np.linalg.norm(pred_centers[:, None, :] - peak_coords[None, :, :], axis=2)
    assignment = dists.argmin(axis=1)

    instances = []
    for k in range(len(peak_ys)):
        member = assignment == k
        if not member.any():
            continue
        member_ys, member_xs = fg_ys[member], fg_xs[member]
        mask = np.zeros(heatmap.shape[1:], dtype=bool)
        mask[member_ys, member_xs] = True
        member_classes = semantic[member_ys, member_xs] - 1  # back to 0-indexed
        label = int(np.bincount(member_classes).argmax())
        score = float(pooled[peak_ys[k], peak_xs[k]])
        instances.append((mask, label, score))
    return instances


def evaluate_instance_ap50(
    predictions: list[list[tuple[np.ndarray, int, float]]],
    ground_truths: list[list[tuple[np.ndarray, int]]],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Per-class AP at a fixed mask-IoU threshold, greedy COCO-style
    matching (predictions ranked by score; each claims its best-IoU
    unmatched ground truth). Hand-rolled rather than routed through
    pycocotools -- our masks are dense booleans, not RLEs, and re-encoding
    for one metric isn't worth the indirection.
    """
    ap_per_class: dict[str, float] = {}

    for c, name in enumerate(class_names):
        scored_preds = [
            (image_idx, mask, score)
            for image_idx, preds in enumerate(predictions)
            for mask, label, score in preds
            if label == c
        ]
        gt_count = sum(1 for gts in ground_truths for _mask, label in gts if label == c)

        if gt_count == 0:
            ap_per_class[name] = float("nan")
            continue
        if not scored_preds:
            ap_per_class[name] = 0.0
            continue

        scored_preds.sort(key=lambda t: -t[2])
        used_gt: dict[int, set[int]] = {}
        tp = np.zeros(len(scored_preds))
        fp = np.zeros(len(scored_preds))

        for i, (image_idx, mask, _score) in enumerate(scored_preds):
            candidates = [(gi, m) for gi, (m, label) in enumerate(ground_truths[image_idx]) if label == c]
            used = used_gt.setdefault(image_idx, set())
            best_iou, best_gi = 0.0, -1
            for gi, gmask in candidates:
                if gi in used:
                    continue
                iou = mask_iou(mask, gmask)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold:
                tp[i] = 1
                used.add(best_gi)
            else:
                fp[i] = 1

        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cum / gt_count
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        recall = np.concatenate([[0.0], recall, [1.0]])
        precision = np.concatenate([[1.0], precision, [0.0]])
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        idx = np.where(recall[1:] != recall[:-1])[0]
        ap_per_class[name] = float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))

    valid = [v for v in ap_per_class.values() if not np.isnan(v)]
    return {"per_class_ap50": ap_per_class, "map50": float(np.mean(valid)) if valid else float("nan")}


def evaluate_occlusion_stratified_recall_segm(
    dataset,
    predictions: list[list[tuple[np.ndarray, int, float]]],
    occlusion_fractions: dict[int, float],
    output_stride: int,
    score_threshold: float = 0.1,
    iou_threshold: float = 0.5,
) -> OcclusionStratifiedRecall:
    """Mask-IoU analog of Milestone 8's box-based occlusion-stratified
    recall (evaluation/detection.py) -- same bucket thresholds, same
    question ("does the model still find an instrument occluded by
    another one"), matched on mask IoU instead of box IoU.
    """
    out_h, out_w = dataset.image_size // output_stride, dataset.image_size // output_stride
    counts = {"isolated": 0, "light": 0, "heavy": 0}
    hits = {"isolated": 0, "light": 0, "heavy": 0}

    for image_idx, (_file_name, anns) in enumerate(dataset.samples):
        image_preds = [(m, label, s) for m, label, s in predictions[image_idx] if s >= score_threshold]
        for a in anns:
            native_mask = decode_instance_mask(a["segmentation"])
            resized = np.array(
                Image.fromarray(native_mask * 255).resize(
                    (dataset.image_size, dataset.image_size), Image.NEAREST
                )
            ) > 0
            gt_mask = downsample_mask_nearest(resized, output_stride, out_h, out_w)
            if not gt_mask.any():
                continue
            label = dataset._id_to_index[a["category_id"]]
            frac = occlusion_fractions.get(a["id"], 0.0)
            bucket = (
                "isolated"
                if frac <= _LIGHT_OCCLUSION_THRESHOLD
                else ("heavy" if frac > _HEAVY_OCCLUSION_THRESHOLD else "light")
            )
            counts[bucket] += 1
            matched = any(
                label == p_label and mask_iou(gt_mask, p_mask) >= iou_threshold
                for p_mask, p_label, _s in image_preds
            )
            if matched:
                hits[bucket] += 1

    def recall(bucket: str) -> float:
        return hits[bucket] / counts[bucket] if counts[bucket] else float("nan")

    return OcclusionStratifiedRecall(
        n_isolated=counts["isolated"], n_light=counts["light"], n_heavy=counts["heavy"],
        recall_isolated=recall("isolated"), recall_light=recall("light"), recall_heavy=recall("heavy"),
    )
