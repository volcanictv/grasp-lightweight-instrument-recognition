"""Evaluate N Mask R-CNN checkpoints, individually and ensembled, against
the official test set. Each `--model checkpoint_path:registry_name` adds
one model (2 or more). Started as a same-architecture fold1/fold2 tool
(legitimate to ensemble on official test since neither fold's training
data overlaps it), generalized for cross-architecture ensembling (e.g.
MobileNetV3 + COCO-pretrained ResNet-50, docs/DECISIONS.md 2026-09-02,
+0.0349 AP50_segm over the best single model) and now N-way ensembling.
See src/surgical_ai/inference/ensemble.py and docs/DECISIONS.md.

Usage (three-way example):
    python scripts/evaluate_maskrcnn_ensemble.py \\
        --model experiments/instance_segmentation_maskrcnn_official_20260901-180758/best.pt:maskrcnn_mobilenet_v3 \\
        --model experiments/instance_segmentation_maskrcnn_resnet50_coco_20260901-233358/best.pt:maskrcnn_resnet50_coco \\
        --model experiments/instance_segmentation_maskrcnn_official_scratch_20260901-211934/best.pt:maskrcnn_mobilenet_v3 \\
        --data-root ./GraSP --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data import splits  # noqa: E402
from surgical_ai.data.detection_dataset import GraspDetectionDataset, build_detection_transforms, collate_fn  # noqa: E402
from surgical_ai.data.mask_utils import decode_instance_mask  # noqa: E402
from surgical_ai.evaluation.detection import compute_occlusion_fractions  # noqa: E402
from surgical_ai.evaluation.segmentation import evaluate_instance_ap50, mask_iou  # noqa: E402
from surgical_ai.inference.ensemble import weighted_fusion_merge  # noqa: E402
from surgical_ai.models.detectors.registry import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", required=True, dest="models",
        help="checkpoint_path:registry_name, repeatable. Need at least 2.",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Candidate threshold before fusion.")
    parser.add_argument("--fusion-iou-threshold", type=float, default=0.5)
    parser.add_argument("--recall-score-threshold", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("need at least 2 --model entries to ensemble")
    return args


@torch.no_grad()
def collect_model_detections(model: torch.nn.Module, image_tensor: torch.Tensor, score_threshold: float) -> list[dict]:
    output = model([image_tensor])[0]
    boxes = output["boxes"].cpu().tolist()
    labels = output["labels"].cpu().tolist()
    scores = output["scores"].cpu().tolist()
    masks = output["masks"].cpu().numpy()[:, 0]  # (N, H, W) float [0,1], pre-threshold
    dets = []
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < score_threshold:
            continue
        dets.append({"box": box, "label": label - 1, "score": score, "mask": mask})  # back to 0-indexed
    return dets


def evaluate_predictions(
    predictions: list[list[tuple]], gt_instances: list[list[tuple]], val_ds, occlusion_fractions: dict,
    class_names: list[str], recall_score_threshold: float, label: str,
) -> None:
    ap50 = evaluate_instance_ap50(predictions, gt_instances, class_names)
    counts = {"isolated": 0, "light": 0, "heavy": 0}
    hits = {"isolated": 0, "light": 0, "heavy": 0}
    for image_idx in range(len(val_ds.samples)):
        _file_name, anns = val_ds.samples[image_idx]
        preds = [(m, lbl, s) for m, lbl, s in predictions[image_idx] if s >= recall_score_threshold]
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if not gt_mask.any():
                continue
            gt_label = val_ds._id_to_index[a["category_id"]]
            frac = occlusion_fractions.get(a["id"], 0.0)
            bucket = "isolated" if frac <= 0.0 else ("heavy" if frac > 0.5 else "light")
            counts[bucket] += 1
            if any(lbl == gt_label and mask_iou(gt_mask, m) >= 0.5 for m, lbl, _s in preds):
                hits[bucket] += 1
    recall_str = ", ".join(
        f"{b}={hits[b] / counts[b] if counts[b] else float('nan'):.3f}" for b in ("isolated", "light", "heavy")
    )
    print(f"[{label}] AP50_segm={ap50['map50']:.4f}  occlusion recall: {recall_str}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    _, val_split = splits.resolve_train_val_split("official")
    val_ds = GraspDetectionDataset(args.data_root, val_split, include_masks=True, transform=build_detection_transforms(train=False))
    class_names = val_ds.class_names_ordered()
    occlusion_fractions = compute_occlusion_fractions(val_ds)

    model_specs = []
    for spec in args.models:
        checkpoint_path, registry_name = spec.rsplit(":", 1)
        model = build_detector(registry_name, num_classes=len(class_names), pretrained=False).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        model_specs.append((checkpoint_path, registry_name, model))

    n_samples = len(val_ds.samples) if args.limit is None else min(args.limit, len(val_ds.samples))
    # Holding every model's individual per-image mask predictions (in addition to
    # the ensemble's) for the full 1125-image official test set OOM'd the
    # workstation at N=3 models (28.5GB resident, killed by the kernel at
    # image ~900/1125 -- confirmed via dmesg, not the earlier network outage
    # as first suspected). Each model's solo score is already known from its
    # own dedicated run anyway, so only the ensemble's predictions are kept
    # here now -- that alone cuts peak memory by roughly (N+1)/2.
    preds_ensemble: list[list[tuple]] = []
    gt_instances: list[list[tuple]] = []

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, range(n_samples)), batch_size=1, shuffle=False, collate_fn=collate_fn,
    )
    for idx, (images, targets) in enumerate(loader):
        image_tensor = images[0].to(device)
        dets_all = [collect_model_detections(model, image_tensor, args.score_threshold) for _cp, _name, model in model_specs]

        fused = weighted_fusion_merge(dets_all, iou_threshold=args.fusion_iou_threshold)
        preds_ensemble.append([(d["mask"] >= 0.5, d["label"], d["score"]) for d in fused])

        image_idx = int(targets[0]["image_id"].item())
        _file_name, anns = val_ds.samples[image_idx]
        gts = []
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if gt_mask.any():
                gts.append((gt_mask, val_ds._id_to_index[a["category_id"]]))
        gt_instances.append(gts)

        if idx % 100 == 0:
            print(f"{idx}/{n_samples}", flush=True)

    # evaluate_predictions indexes positionally against the full val_ds.samples,
    # so pad untested indices (under --limit) with empty predictions rather than
    # truncate -- see the identical fix in evaluate_tracking_segm.py.
    for _ in range(len(val_ds.samples) - n_samples):
        preds_ensemble.append([])
        gt_instances.append([])

    # Individual per-model scores: see each model's own dedicated training run --
    # not recomputed here (see the memory note above this loop).
    evaluate_predictions(preds_ensemble, gt_instances, val_ds, occlusion_fractions, class_names, args.recall_score_threshold, f"ensemble of {len(model_specs)} (WBF)")


if __name__ == "__main__":
    main()
