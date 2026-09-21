"""Dirichlet evidential quantities and the variance-based uncertainty scores of
Duan et al. (WACV 2024, arXiv 2311.11367). No softmax is involved: evidence is
exp of the clamped logits, alpha = evidence + 1, mu_bar = alpha / alpha0.

Covariance of the class vector: Cov = Diag(mu_bar) - mu_bar mu_bar^T; its aleatoric part is
alpha0 / (alpha0 + 1) * Cov and its epistemic part 1 / (alpha0 + 1) * Cov. Sample-level scores are
the traces: (1 - sum mu_bar^2) times the scalar factor.
"""

from __future__ import annotations

import numpy as np


def alpha_from_logits(logits: np.ndarray, clamp: float = 10.0) -> np.ndarray:
    return np.exp(np.clip(logits, -clamp, clamp)) + 1.0


def variance_scores(alpha: np.ndarray) -> dict[str, np.ndarray]:
    """alpha: (N, C) Dirichlet parameters. Returns per-sample scores, higher = more uncertain."""
    alpha0 = alpha.sum(axis=1)
    mu = alpha / alpha0[:, None]
    trace_cov = 1.0 - (mu ** 2).sum(axis=1)
    return {
        "epistemic": trace_cov / (alpha0 + 1.0),
        "aleatoric": trace_cov * alpha0 / (alpha0 + 1.0),
        "total": trace_cov,
        "evidence_only": 1.0 / (alpha0 + 1.0),
    }
