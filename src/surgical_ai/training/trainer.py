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
def run_detection_eval(
    model: nn.Module, loader: DataLoader, coco_gt, class_names: list[str], device: torch.device,
) -> DetectionMetrics:
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
) -> tuple[DetectionHistory, DetectionMetrics]:
    trainable, total = count_parameters(model)
    logger.info(
        "trainable params: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total
    )

    model.to(device)
    history = DetectionHistory()
    best_map50 = -1.0
    best_metrics: DetectionMetrics | None = None

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
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return history, best_metrics
