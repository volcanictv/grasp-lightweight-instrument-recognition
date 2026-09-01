"""Classification metrics for both tasks. Task A (multi-label) deliberately
omits accuracy — CLAUDE.md: accuracy is never the headline number there,
macro-F1 and per-class breakdowns are. Task B (single-label, region
classification) does report accuracy per CLAUDE.md's metrics list, but
still treats macro-F1 as primary for model selection and comparability with
Task A.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class MultiLabelMetrics:
    class_names: list[str]
    per_class_ap: dict[str, float]
    mean_ap: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_f1: float
    confusion_matrices: dict[str, np.ndarray] = field(repr=False)  # name -> [[tn,fp],[fn,tp]]

    def to_markdown(self) -> str:
        lines = [
            "| class | AP | precision | recall | F1 |",
            "|---|---|---|---|---|",
        ]
        for name in self.class_names:
            lines.append(
                f"| {name} | {self.per_class_ap[name]:.3f} | "
                f"{self.per_class_precision[name]:.3f} | "
                f"{self.per_class_recall[name]:.3f} | "
                f"{self.per_class_f1[name]:.3f} |"
            )
        lines.append(f"| **mean/macro** | {self.mean_ap:.3f} | | | {self.macro_f1:.3f} |")
        return "\n".join(lines)


def evaluate_multilabel(
    y_true: np.ndarray, y_score: np.ndarray, class_names: list[str], threshold: float = 0.5
) -> MultiLabelMetrics:
    """y_true: (N, C) binary. y_score: (N, C) sigmoid probabilities (not logits)."""
    y_pred = (y_score >= threshold).astype(int)

    per_class_ap = {}
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() == 0:
            per_class_ap[name] = float("nan")
        else:
            per_class_ap[name] = float(average_precision_score(y_true[:, i], y_score[:, i]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    cm = multilabel_confusion_matrix(y_true, y_pred)

    valid_aps = [v for v in per_class_ap.values() if not np.isnan(v)]

    return MultiLabelMetrics(
        class_names=class_names,
        per_class_ap=per_class_ap,
        mean_ap=float(np.mean(valid_aps)) if valid_aps else float("nan"),
        per_class_precision={n: float(p) for n, p in zip(class_names, precision)},
        per_class_recall={n: float(r) for n, r in zip(class_names, recall)},
        per_class_f1={n: float(f) for n, f in zip(class_names, f1)},
        macro_f1=macro_f1,
        confusion_matrices={n: cm[i] for i, n in enumerate(class_names)},
    )


@dataclass
class RegionClassificationMetrics:
    """Task B (single-label instance-crop classification) metrics per
    CLAUDE.md: accuracy, macro precision/recall/F1, per-class F1, confusion
    matrix. Accuracy is reported here (unlike Task A) because it's a
    legitimate single-label metric; it still isn't the headline number --
    macro_f1 is, for consistency with Task A and because it doesn't hide
    rare-class failure the way accuracy can.
    """

    class_names: list[str]
    accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    macro_f1: float
    confusion_matrix: np.ndarray = field(repr=False)  # rows=true, cols=pred

    def to_markdown(self) -> str:
        lines = [
            "| class | precision | recall | F1 |",
            "|---|---|---|---|",
        ]
        for name in self.class_names:
            lines.append(
                f"| {name} | {self.per_class_precision[name]:.3f} | "
                f"{self.per_class_recall[name]:.3f} | "
                f"{self.per_class_f1[name]:.3f} |"
            )
        lines.append(f"| **macro / accuracy** | | | {self.macro_f1:.3f} / {self.accuracy:.3f} |")
        return "\n".join(lines)


def evaluate_region_classification(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> RegionClassificationMetrics:
    """y_true, y_pred: (N,) integer class indices, 0..len(class_names)-1."""
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return RegionClassificationMetrics(
        class_names=class_names,
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class_precision={n: float(p) for n, p in zip(class_names, precision)},
        per_class_recall={n: float(r) for n, r in zip(class_names, recall)},
        per_class_f1={n: float(f) for n, f in zip(class_names, f1)},
        macro_f1=float(np.mean(f1)),
        confusion_matrix=cm,
    )
