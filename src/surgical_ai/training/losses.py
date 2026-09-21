"""Loss construction from config. `class_weights: true` is a Milestone 5
ablation variable for Task A (`bce`) and the default imbalance handling for
Task B (`cross_entropy`, Milestone 7) — baseline configs leave it false.
Kept here (not decided per-run in the trainer) so weighting strategy stays a
one-line config change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def compute_pos_weight(label_counts: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Inverse-frequency positive class weight for BCEWithLogitsLoss:
    (negatives / positives) per class, so rare classes get a bigger loss
    contribution per positive example.
    """
    positives = label_counts.clamp(min=1)
    negatives = num_samples - label_counts
    return negatives / positives


def compute_class_weights(label_counts: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """Balanced per-class weight for CrossEntropyLoss (Task B):
    n_samples / (n_classes * count_c), so weights average to ~1 rather than
    blowing up the loss scale the way raw inverse frequency would.

    `power` sharpens (>1) or softens (<1) the balanced weight beyond the
    standard formula -- `docs/DECISIONS.md` 2026-09-02: after the
    letterbox-crop ablation measurably hurt the two smallest classes (Clip
    Applier, Laparoscopic Grasper), tried alongside a restricted
    (aspect-ratio-gated) letterbox as a combined attempt to recover them
    without giving up the letterbox fix's confirmed gain on the confused
    classes. Default 1.0 is the original, unchanged formula.
    """
    total = label_counts.sum()
    n_classes = label_counts.numel()
    weight = total / (n_classes * label_counts.clamp(min=1))
    return weight**power if power != 1.0 else weight


class FocalBCELoss(nn.Module):
    """Multi-label focal loss (Lin et al. 2017, RetinaNet), BCE variant --
    down-weights examples the model already gets confidently right and
    concentrates gradient on borderline/hard ones. Motivated by
    docs/error_analysis.md 2026-09-02: Task A's Suction Instrument misses
    are mostly borderline (mean predicted probability 0.507 when actually
    present, only 14% scored under 0.1) rather than confidently wrong --
    exactly the failure mode focal loss targets, and a genuinely untried
    lever for this project (weighted loss/sampler/augmentation were all
    tried in Milestone 5; not this). Composes with `pos_weight` so it's a
    strict addition to the existing weighted-loss mechanism, not a
    replacement.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class EvidentialLoss(nn.Module):
    """Dirichlet evidential loss (Sensoy et al. 2018; Duan et al., WACV 2024).

    Evidence is exp of the clamped logits (no softmax anywhere), alpha = evidence + 1.
    Loss per sample = sum_c y_c (log alpha0 - log alpha_c)  +  lambda * KL(Dir(alpha_tilde) || Dir(1)),
    with alpha_tilde = y + (1 - y) * alpha. `lambda` is ramped linearly over `anneal_epochs`
    (0 = constant), driven by `set_epoch`, which the trainer calls once per epoch if present.
    Optional per-class weights scale each sample's loss by its true class's weight.
    """

    def __init__(self, reg_lambda: float, anneal_epochs: int = 0, clamp: float = 10.0,
                 class_weight: torch.Tensor | None = None):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.anneal_epochs = anneal_epochs
        self.clamp = clamp
        self.epoch = 1
        self.register_buffer("class_weight", class_weight, persistent=False)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def lambda_now(self) -> float:
        if self.anneal_epochs <= 0:
            return self.reg_lambda
        return self.reg_lambda * min(1.0, self.epoch / self.anneal_epochs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(logits.clamp(-self.clamp, self.clamp)) + 1.0
        num_classes = alpha.shape[1]
        y = nn.functional.one_hot(targets, num_classes).to(alpha.dtype)
        alpha0 = alpha.sum(dim=1, keepdim=True)
        nll = (y * (torch.log(alpha0) - torch.log(alpha))).sum(dim=1)

        alpha_t = y + (1.0 - y) * alpha
        sum_t = alpha_t.sum(dim=1, keepdim=True)
        kl = (
            torch.lgamma(sum_t).squeeze(1) - math.lgamma(num_classes) - torch.lgamma(alpha_t).sum(dim=1)
            + ((alpha_t - 1.0) * (torch.digamma(alpha_t) - torch.digamma(sum_t))).sum(dim=1)
        )
        per_sample = nll + self.lambda_now() * kl
        if self.class_weight is not None:
            w = self.class_weight[targets]
            return (per_sample * w).sum() / w.sum()  # same normalisation as nn.CrossEntropyLoss(weight=...)
        return per_sample.mean()


def build_loss(
    loss_config: dict,
    pos_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
) -> nn.Module:
    loss_type = loss_config.get("type", "bce")
    use_weights = loss_config.get("class_weights", False)

    if loss_type == "bce":
        if use_weights and pos_weight is None:
            raise ValueError("loss.class_weights is true but no pos_weight was computed")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight if use_weights else None)

    if loss_type == "focal_bce":
        if use_weights and pos_weight is None:
            raise ValueError("loss.class_weights is true but no pos_weight was computed")
        return FocalBCELoss(gamma=loss_config.get("focal_gamma", 2.0), pos_weight=pos_weight if use_weights else None)

    if loss_type == "cross_entropy":
        if use_weights and class_weight is None:
            raise ValueError("loss.class_weights is true but no class_weight was computed")
        return nn.CrossEntropyLoss(weight=class_weight if use_weights else None)

    if loss_type == "evidential":
        if use_weights and class_weight is None:
            raise ValueError("loss.class_weights is true but no class_weight was computed")
        return EvidentialLoss(
            reg_lambda=loss_config.get("edl_lambda", 1.0 / 7),
            anneal_epochs=loss_config.get("edl_anneal_epochs", 0),
            clamp=loss_config.get("edl_clamp", 10.0),
            class_weight=class_weight if use_weights else None,
        )

    raise ValueError(f"unknown loss type '{loss_type}'. Valid: bce, focal_bce, cross_entropy, evidential")
