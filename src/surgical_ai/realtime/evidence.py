"""Single-pass evidence, the gate score and the frame fusion, on the same math as the offline scoring
(surgical_ai.evaluation.evidential; scripts/evidential_three_member_yolo.py), so a live run and the
offline numbers cannot drift apart."""

from __future__ import annotations

import numpy as np

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores


def weighted_alpha(member_logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """member_logits (..., members, classes), weights (members,) summing to 1 -> Dirichlet alpha (..., classes)."""
    return (alpha_from_logits(member_logits) * weights[:, None]).sum(axis=-2)


def label_and_gate(member_logits: np.ndarray, weights: np.ndarray) -> tuple[int, float]:
    """Single-frame label and the epistemic score S1 the gate thresholds."""
    alpha = weighted_alpha(member_logits, weights)[None]
    return int((alpha / alpha.sum(axis=1, keepdims=True)).argmax()), float(variance_scores(alpha)["epistemic"][0])


def fuse_frames(frame_member_logits: np.ndarray, weights: np.ndarray) -> int:
    """frame_member_logits (frames, members, classes). Each frame votes with its top mu_bar as weight."""
    alpha = weighted_alpha(frame_member_logits, weights)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    return int((mu.max(axis=1)[:, None] * mu).sum(axis=0).argmax())
