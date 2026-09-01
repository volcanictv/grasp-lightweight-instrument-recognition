"""Object detection metrics via pycocotools COCOeval -- mAP@50, mAP@50:95,
per-class AP@50. Same metric family the published GraSP literature reports
(TAPIS, LACOSTE's AP50_box -- see docs/findings.md's literature check), so
this project's detection number will sit in the same table as theirs
instead of needing a translation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Occlusion buckets: fraction of an instance's own bbox area covered by any
# other co-occurring instrument's bbox in the same frame. Thresholds match
# the two figures already documented in docs/imbalance_notes.md Problem 4
# ("~30% of co-occurring pairs overlap", "~10% have the smaller box >50%
# covered"), so this analysis is directly comparable to that prior finding.
_LIGHT_OCCLUSION_THRESHOLD = 0.0
_HEAVY_OCCLUSION_THRESHOLD = 0.5


def dataset_to_coco_gt(dataset) -> COCO:
    """Builds an in-memory COCO ground-truth object from a
    GraspDetectionDataset, following torchvision's own reference
    conversion pattern (coco_utils.convert_to_coco_api).
    """
    images, annotations, categories = [], [], []
    ann_id = 1
    for idx in range(len(dataset)):
        file_name, anns = dataset.samples[idx]
        images.append({"id": idx, "file_name": file_name})
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            label = dataset._id_to_index[a["category_id"]] + 1  # 0 = background
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": label,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    for i, name in enumerate(dataset.class_names_ordered()):
        categories.append({"id": i + 1, "name": name})

    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco.createIndex()
    return coco


@dataclass
class DetectionMetrics:
    class_names: list[str]
    map50_95: float
    map50: float
    per_class_ap50: dict[str, float]

    def to_markdown(self) -> str:
        lines = ["| class | AP@50 |", "|---|---|"]
        for name in self.class_names:
            lines.append(f"| {name} | {self.per_class_ap50[name]:.3f} |")
        lines.append(f"| **mAP@50 / mAP@50:95** | {self.map50:.3f} / {self.map50_95:.3f} |")
        return "\n".join(lines)


def evaluate_detection(
    coco_gt: COCO, predictions: list[dict], class_names: list[str]
) -> DetectionMetrics:
    """predictions: list of {"image_id", "category_id", "bbox": [x,y,w,h], "score"}."""
    if not predictions:
        return DetectionMetrics(
            class_names=class_names, map50_95=0.0, map50=0.0,
            per_class_ap50={n: 0.0 for n in class_names},
        )

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50_95 = float(coco_eval.stats[0])
    map50 = float(coco_eval.stats[1])

    # precision shape [T, R, K, A, M]: IoU thresholds x recall thresholds x
    # categories x area ranges x max-dets. IoU=0.5 -> index 0, area='all'
    # -> index 0, maxDets=100 -> last index.
    precision = coco_eval.eval["precision"]
    per_class_ap50 = {}
    for k, name in enumerate(class_names):
        p = precision[0, :, k, 0, -1]
        p = p[p > -1]
        per_class_ap50[name] = float(np.mean(p)) if p.size else float("nan")

    return DetectionMetrics(
        class_names=class_names, map50_95=map50_95, map50=map50, per_class_ap50=per_class_ap50
    )


def compute_occlusion_fractions(dataset) -> dict[int, float]:
    """ann_id -> max fraction of this instance's own bbox area covered by
    any other co-occurring instrument's bbox in the same frame. 0.0 means
    no overlap with anything else in its frame. This is an occlusion proxy,
    not a measurement of true pixel-level occlusion (two boxes can overlap
    without either instrument actually hiding the other), but it is cheap,
    matches how docs/imbalance_notes.md Problem 4 already characterizes
    bbox overlap in this dataset, and needs no new annotation.
    """
    fractions: dict[int, float] = {}
    for _file_name, anns in dataset.samples:
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append((a["id"], x, y, x + w, y + h, w * h))
        for i, (ann_id, x1, y1, x2, y2, area) in enumerate(boxes):
            if area <= 0:
                fractions[ann_id] = 0.0
                continue
            max_frac = 0.0
            for j, (_other_id, ox1, oy1, ox2, oy2, _oarea) in enumerate(boxes):
                if i == j:
                    continue
                iw = max(0.0, min(x2, ox2) - max(x1, ox1))
                ih = max(0.0, min(y2, oy2) - max(y1, oy1))
                max_frac = max(max_frac, (iw * ih) / area)
            fractions[ann_id] = max_frac
    return fractions


def _box_iou(a: list[float], b: list[float]) -> float:
    """a, b: [x, y, w, h]."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class OcclusionStratifiedRecall:
    n_isolated: int
    n_light: int
    n_heavy: int
    recall_isolated: float
    recall_light: float
    recall_heavy: float

    def to_markdown(self) -> str:
        return (
            "| occlusion bucket | n instances | recall |\n"
            "|---|---|---|\n"
            f"| isolated (no overlap) | {self.n_isolated} | {self.recall_isolated:.3f} |\n"
            f"| light overlap (<=50% covered) | {self.n_light} | {self.recall_light:.3f} |\n"
            f"| heavy overlap (>50% covered) | {self.n_heavy} | {self.recall_heavy:.3f} |"
        )


def evaluate_occlusion_stratified_recall(
    dataset,
    predictions: list[dict],
    occlusion_fractions: dict[int, float],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> OcclusionStratifiedRecall:
    """Recall at a fixed score/IoU threshold, split by how occluded each
    ground-truth instance is. Answers "does the model still find an
    instrument when another one is covering part of it" directly, unlike
    COCO's small/medium/large-area AP breakdown, which this dataset has
    almost no instances in the 'small' bucket for (see docs/DECISIONS.md)
    and so cannot answer that question.
    """
    preds_by_image: dict[int, list[dict]] = {}
    for p in predictions:
        if p["score"] >= score_threshold:
            preds_by_image.setdefault(p["image_id"], []).append(p)

    counts = {"isolated": 0, "light": 0, "heavy": 0}
    hits = {"isolated": 0, "light": 0, "heavy": 0}

    for image_idx, (_file_name, anns) in enumerate(dataset.samples):
        image_preds = preds_by_image.get(image_idx, [])
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            category_id = dataset._id_to_index[a["category_id"]] + 1
            frac = occlusion_fractions.get(a["id"], 0.0)
            bucket = (
                "isolated"
                if frac <= _LIGHT_OCCLUSION_THRESHOLD
                else ("heavy" if frac > _HEAVY_OCCLUSION_THRESHOLD else "light")
            )
            counts[bucket] += 1
            matched = any(
                p["category_id"] == category_id and _box_iou([x, y, w, h], p["bbox"]) >= iou_threshold
                for p in image_preds
            )
            if matched:
                hits[bucket] += 1

    def recall(bucket: str) -> float:
        return hits[bucket] / counts[bucket] if counts[bucket] else float("nan")

    return OcclusionStratifiedRecall(
        n_isolated=counts["isolated"],
        n_light=counts["light"],
        n_heavy=counts["heavy"],
        recall_isolated=recall("isolated"),
        recall_light=recall("light"),
        recall_heavy=recall("heavy"),
    )


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """box: [4], boxes: [N, 4], both [x1, y1, x2, y2]. Returns IoU, shape [N]."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def soft_nms_scores(boxes_xyxy: np.ndarray, scores: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Gaussian soft-NMS (Bodla et al., ICCV 2017, arXiv:1704.04503) --
    decays a box's score by exp(-iou^2/sigma) against every higher-scored
    box it overlaps, instead of hard-suppressing (removing) it outright.
    This is the standard fix for two real, overlapping objects where hard
    NMS incorrectly treats the lower-scored one as a duplicate detection of
    the same object -- exactly the failure mode this project's occlusion-
    stratified recall (above) measured (see docs/detection_literature_notes.md).
    Not validated for surgical instrument detection specifically in the
    literature reviewed for this project -- being tried here because it
    directly targets the diagnosed mechanism and costs nothing to test
    (pure post-processing on an already-trained checkpoint's raw, un-
    suppressed candidate boxes; see docs/DECISIONS.md).

    Returns adjusted scores in the input order; does not remove any box --
    the caller filters by a score threshold afterward. Assumes `boxes_xyxy`
    only contains boxes from a single image and a single class (NMS is
    always class-wise; mixing classes here would incorrectly decay one
    instrument's score because a different instrument overlaps it).
    """
    scores = scores.copy()
    order = list(np.argsort(-scores))
    processed: list[int] = []
    while order:
        i = order.pop(0)
        processed.append(i)
        if not order:
            break
        remaining = np.array(order)
        ious = _iou_xyxy(boxes_xyxy[i], boxes_xyxy[remaining])
        scores[remaining] *= np.exp(-(ious**2) / sigma)
        order = list(remaining[np.argsort(-scores[remaining])])
    return scores


def apply_soft_nms_to_predictions(
    predictions: list[dict], sigma: float = 0.5, score_threshold: float = 0.05
) -> list[dict]:
    """Re-scores raw COCO-format detections (as produced by
    training.trainer.collect_detections) with soft_nms_scores instead of
    torchvision's default hard NMS, grouped by (image, category) -- matching
    how class-wise hard NMS already partitions the problem -- then drops
    anything below `score_threshold`. Expects `predictions` to come from a
    model built with `box_nms_thresh` close to 1.0 (hard NMS effectively
    disabled) and a generous `box_detections_per_img`, so overlapping
    same-class candidates weren't already removed before this runs.
    """
    groups: dict[tuple[int, int], list[dict]] = {}
    for p in predictions:
        groups.setdefault((p["image_id"], p["category_id"]), []).append(p)

    output = []
    for group in groups.values():
        boxes = np.array(
            [[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]] for p in group]
        )
        scores = np.array([p["score"] for p in group])
        adjusted = soft_nms_scores(boxes, scores, sigma=sigma)
        for p, s in zip(group, adjusted):
            if s >= score_threshold:
                q = dict(p)
                q["score"] = float(s)
                output.append(q)
    return output
