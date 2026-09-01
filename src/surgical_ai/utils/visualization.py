"""Headless plotting and mask-overlay helpers. No display; everything saves
to disk. Used by scripts/inspect_dataset.py for Milestone 0, and reused later
by training/evaluation visualization.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Fixed, distinct colors per instrument category id (1-7). Background (0) is
# left transparent in overlays.
CATEGORY_COLORS = {
    1: (230, 25, 75),
    2: (60, 180, 75),
    3: (255, 225, 25),
    4: (0, 130, 200),
    5: (245, 130, 48),
    6: (145, 30, 180),
    7: (70, 240, 240),
}


def plot_class_histogram(
    counts: dict[str, int], title: str, out_path: Path, ylabel: str = "count"
) -> None:
    labels = list(counts.keys())
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#4c72b0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cooccurrence_matrix(
    cat_ids: list[int], matrix: list[list[int]], names: dict[int, str], out_path: Path
) -> None:
    labels = [names[c] for c in cat_ids]
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] < arr.max() / 2 else "black",
                fontsize=7,
            )
    ax.set_title("Class co-occurrence (frames), diagonal = frames containing class")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_frames_per_case(counts: dict[str, int], title: str, out_path: Path) -> None:
    plot_class_histogram(counts, title, out_path, ylabel="frames")


def plot_confusion_matrices(
    confusion_matrices: dict[str, np.ndarray], out_path: Path, ncols: int = 4
) -> None:
    """confusion_matrices: name -> [[tn,fp],[fn,tp]], one per class (see
    evaluation/classification.py::evaluate_multilabel)."""
    names = list(confusion_matrices.keys())
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, name in zip(axes, names):
        cm = confusion_matrices[name]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred neg", "pred pos"], fontsize=7)
        ax.set_yticklabels(["true neg", "true pos"], fontsize=7)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    for ax in axes[len(names):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multiclass_confusion_matrix(
    cm: np.ndarray, class_names: list[str], out_path: Path
) -> None:
    """cm: (N, N) counts, rows=true, cols=pred (Task B, single-label)."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_prf_bars(
    per_class_precision: dict[str, float],
    per_class_recall: dict[str, float],
    per_class_f1: dict[str, float],
    out_path: Path,
) -> None:
    names = list(per_class_precision.keys())
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, [per_class_precision[n] for n in names], width, label="precision")
    ax.bar(x, [per_class_recall[n] for n in names], width, label="recall")
    ax.bar(x + width, [per_class_f1[n] for n in names], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Per-class precision / recall / F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curve(values: list[float], ylabel: str, title: str, out_path: Path) -> None:
    epochs = range(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, values, marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    train_values: list[float], val_values: list[float], ylabel: str, title: str, out_path: Path
) -> None:
    epochs = range(1, len(train_values) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_values, label="train", marker="o", markersize=3)
    ax.plot(epochs, val_values, label="val", marker="o", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_mask_overlay(
    frame_path: Path, mask_path: Path, out_path: Path, alpha: float = 0.5
) -> None:
    frame = Image.open(frame_path).convert("RGB")
    mask = Image.open(mask_path)
    mask_arr = np.array(mask)

    overlay = np.array(frame).copy()
    for cat_id, color in CATEGORY_COLORS.items():
        sel = mask_arr == cat_id
        if not sel.any():
            continue
        overlay[sel] = (
            overlay[sel] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)
