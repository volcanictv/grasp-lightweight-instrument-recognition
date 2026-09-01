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
