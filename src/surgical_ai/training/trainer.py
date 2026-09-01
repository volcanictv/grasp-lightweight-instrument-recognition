"""Generic training loop for multi-label frame classification (Task A)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from surgical_ai.evaluation.classification import (
    MultiLabelMetrics,
    RegionClassificationMetrics,
    evaluate_multilabel,
    evaluate_region_classification,
)
from surgical_ai.evaluation.detection import DetectionMetrics, evaluate_detection
from surgical_ai.evaluation.segmentation import SemanticSegmentationMetrics, evaluate_semantic_segmentation
from surgical_ai.training.segmentation_losses import compute_segmentation_loss

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable, total). Frozen vs fine-tuned should be
    distinguishable from this alone, per PROJECT_SPEC.md §6."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_macro_f1: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, class_names: list[str],
    device: torch.device,
) -> tuple[float, MultiLabelMetrics]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_scores, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    metrics = evaluate_multilabel(y_true, y_score, class_names)
    return total_loss / n, metrics


@torch.no_grad()
def evaluate_region(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, class_names: list[str],
    device: torch.device,
) -> tuple[float, RegionClassificationMetrics]:
    """Task B counterpart to `evaluate` -- argmax predictions and
    single-label metrics instead of sigmoid scores and multi-label ones.
    """
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = evaluate_region_classification(y_true, y_pred, class_names)
    return total_loss / n, metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    evaluate_fn=evaluate,
) -> tuple[TrainHistory, MultiLabelMetrics | RegionClassificationMetrics]:
    """`evaluate_fn` defaults to Task A's `evaluate` (unchanged behavior for
    every existing call site). Pass `evaluate_region` for Task B -- the only
    requirement on its return type is a `.macro_f1` field, which both
    metrics dataclasses have, so the epoch loop and checkpoint selection
    below don't need to know which task they're running.
    """
    trainable, total = count_parameters(model)
    logger.info(
        "trainable params: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total
    )

    model.to(device)
    history = TrainHistory()
    best_f1 = -1.0
    best_metrics: MultiLabelMetrics | RegionClassificationMetrics | None = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        # Re-running eval on the train set is redundant compute (roughly 2x
        # epoch time) but is what CLAUDE.md's "train/val metric curves" asks
        # for — train_loss alone from the training pass isn't the same
        # signal as a proper metric computed in eval mode.
        _, train_metrics = evaluate_fn(model, train_loader, loss_fn, class_names, device)
        val_loss, val_metrics = evaluate_fn(model, val_loader, loss_fn, class_names, device)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_macro_f1.append(train_metrics.macro_f1)
        history.val_macro_f1.append(val_metrics.macro_f1)
        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f train_macro_f1=%.4f val_macro_f1=%.4f",
            epoch, epochs, train_loss, val_loss, train_metrics.macro_f1, val_metrics.macro_f1,
        )

        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            best_metrics = val_metrics
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return history, best_metrics


@dataclass
class DetectionHistory:
    train_loss: list[float] = field(default_factory=list)
    val_map50: list[float] = field(default_factory=list)
    val_map50_95: list[float] = field(default_factory=list)


def train_one_epoch_detection(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    """Detection models compute their own multi-task loss (RPN + ROI heads)
    internally when given targets in train mode -- no external loss_fn,
    unlike train_one_epoch above.
    """
    model.train()
    total_loss = 0.0
    n = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
        n += len(images)
    return total_loss / n


@torch.no_grad()
def collect_detections(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict]:
    """Raw COCO-format predictions, shared by run_detection_eval (aggregate
    mAP) and the occlusion-stratified recall analysis (evaluation/detection.py)
    -- both need the same per-instance predictions, just scored differently.
    """
    model.eval()
    predictions = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for t, out in zip(targets, outputs):
            image_id = int(t["image_id"].item())
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    return predictions


def run_detection_eval(
    model: nn.Module, loader: DataLoader, coco_gt, class_names: list[str], device: torch.device,
) -> DetectionMetrics:
    predictions = collect_detections(model, loader, device)
    return evaluate_detection(coco_gt, predictions, class_names)


def fit_detection(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    coco_gt_val,
    patience: int | None = None,
) -> tuple[DetectionHistory, DetectionMetrics]:
    """`epochs` is now a ceiling, not a target -- with `patience` set, training
    stops once val mAP@50 hasn't improved for that many consecutive epochs,
    since training time is no longer a constrained resource for this project
    (only inference-time efficiency is) and a flat epoch count was picked
    somewhat arbitrarily before that was clarified (see docs/DECISIONS.md).
    `patience=None` disables early stopping and always runs the full ceiling.
    """
    trainable, total = count_parameters(model)
    logger.info(
        "trainable params: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total
    )

    model.to(device)
    history = DetectionHistory()
    best_map50 = -1.0
    best_metrics: DetectionMetrics | None = None
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_detection(model, train_loader, optimizer, device)
        val_metrics = run_detection_eval(model, val_loader, coco_gt_val, class_names, device)

        history.train_loss.append(train_loss)
        history.val_map50.append(val_metrics.map50)
        history.val_map50_95.append(val_metrics.map50_95)
        logger.info(
            "epoch %d/%d train_loss=%.4f val_mAP50=%.4f val_mAP50:95=%.4f",
            epoch, epochs, train_loss, val_metrics.map50, val_metrics.map50_95,
        )

        if val_metrics.map50 > best_map50:
            best_map50 = val_metrics.map50
            best_metrics = val_metrics
            epochs_since_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_improvement += 1
            if patience is not None and epochs_since_improvement >= patience:
                logger.info(
                    "early stopping: no val mAP@50 improvement in %d epochs (best=%.4f at epoch %d)",
                    patience, best_map50, epoch - epochs_since_improvement,
                )
                break

    return history, best_metrics


@dataclass
class SegmentationHistory:
    train_loss: list[float] = field(default_factory=list)
    train_heatmap_loss: list[float] = field(default_factory=list)
    train_offset_loss: list[float] = field(default_factory=list)
    train_semantic_loss: list[float] = field(default_factory=list)
    val_miou: list[float] = field(default_factory=list)


def train_one_epoch_segmentation(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    offset_weight: float,
    semantic_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "heatmap": 0.0, "offset": 0.0, "semantic": 0.0}
    n = 0
    for images, targets in loader:
        images = images.to(device)
        targets = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in targets.items()
            if k not in ("instance_masks", "instance_labels")
        }
        optimizer.zero_grad()
        predictions = model(images)
        loss_dict = compute_segmentation_loss(
            predictions, targets, offset_weight=offset_weight, semantic_weight=semantic_weight
        )
        loss_dict["total"].backward()
        optimizer.step()

        batch_size = images.shape[0]
        for k in totals:
            totals[k] += loss_dict[k].item() * batch_size
        n += batch_size
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def run_segmentation_eval(
    model: nn.Module, loader: DataLoader, class_names: list[str], device: torch.device,
) -> SemanticSegmentationMetrics:
    """Cheap, per-epoch checkpoint-selection metric -- semantic mIoU only.
    The more expensive instance-level metrics (AP50_segm, occlusion-
    stratified recall, both needing per-image instance decoding) are
    computed once at the end on the best checkpoint, same convention
    `run_detection_training` already uses for its occlusion analysis
    (scripts/train.py).
    """
    model.eval()
    pred_semantic_list, gt_semantic_list = [], []
    for images, targets in loader:
        images = images.to(device)
        semantic_pred = model(images)["semantic"].argmax(dim=1).cpu().numpy()
        for p, g in zip(semantic_pred, targets["semantic"].numpy()):
            pred_semantic_list.append(p)
            gt_semantic_list.append(g)
    return evaluate_semantic_segmentation(pred_semantic_list, gt_semantic_list, class_names)


def fit_segmentation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    patience: int | None = None,
    offset_weight: float = 0.1,
    semantic_weight: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[SegmentationHistory, SemanticSegmentationMetrics]:
    """Mirrors `fit_detection`'s epochs-as-ceiling / patience-based early
    stopping convention (docs/DECISIONS.md: training time is not a
    constrained resource here, only inference latency is).

    `scheduler` (optional, stepped once per epoch) exists because the
    first two Milestone 9 runs (fixed `lr=0.0001`) showed periodic
    heatmap-loss spikes late in training (e.g. epoch 65, 73, 120 in
    `segmentation_extended_patience` all jump ~0.5 then recover within 1-2
    epochs) consistent with Adam occasionally destabilizing once the loss
    is already small -- a decaying LR is the standard fix, not something
    derived here.
    """
    trainable, total = count_parameters(model)
    logger.info("trainable params: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total)

    model.to(device)
    history = SegmentationHistory()
    best_miou = -1.0
    best_metrics: SemanticSegmentationMetrics | None = None
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        train_losses = train_one_epoch_segmentation(
            model, train_loader, optimizer, device, offset_weight, semantic_weight
        )
        if scheduler is not None:
            scheduler.step()
        val_metrics = run_segmentation_eval(model, val_loader, class_names, device)

        history.train_loss.append(train_losses["total"])
        history.train_heatmap_loss.append(train_losses["heatmap"])
        history.train_offset_loss.append(train_losses["offset"])
        history.train_semantic_loss.append(train_losses["semantic"])
        history.val_miou.append(val_metrics.miou)
        logger.info(
            "epoch %d/%d train_loss=%.4f (heatmap=%.4f offset=%.4f semantic=%.4f) val_mIoU=%.4f",
            epoch, epochs, train_losses["total"], train_losses["heatmap"],
            train_losses["offset"], train_losses["semantic"], val_metrics.miou,
        )

        if val_metrics.miou > best_miou:
            best_miou = val_metrics.miou
            best_metrics = val_metrics
            epochs_since_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_improvement += 1
            if patience is not None and epochs_since_improvement >= patience:
                logger.info(
                    "early stopping: no val mIoU improvement in %d epochs (best=%.4f at epoch %d)",
                    patience, best_miou, epoch - epochs_since_improvement,
                )
                break

    return history, best_metrics
