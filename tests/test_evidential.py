import numpy as np
import torch

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.training.losses import EvidentialLoss, build_loss


def test_loss_finite_and_lower_for_correct_confident_logits():
    loss = EvidentialLoss(reg_lambda=0.1)
    targets = torch.tensor([0, 1, 2])
    good = torch.full((3, 4), -3.0)
    good[torch.arange(3), targets] = 4.0
    bad = torch.full((3, 4), -3.0)
    bad[torch.arange(3), (targets + 1) % 4] = 4.0
    assert torch.isfinite(loss(good, targets)) and torch.isfinite(loss(bad, targets))
    assert loss(good, targets) < loss(bad, targets)


def test_kl_vanishes_when_all_wrong_class_evidence_is_zero():
    # wrong-class evidence 0 (logits very negative) => alpha_tilde == 1 everywhere => KL == 0
    targets = torch.tensor([1, 3])
    logits = torch.full((2, 5), -50.0)  # clamped to -10, evidence ~4.5e-5, KL ~ 0
    with_kl = EvidentialLoss(reg_lambda=5.0)(logits, targets)
    no_kl = EvidentialLoss(reg_lambda=0.0)(logits, targets)
    assert torch.isclose(with_kl, no_kl, atol=1e-3)


def test_kl_penalises_misleading_evidence():
    targets = torch.tensor([0])
    logits = torch.tensor([[0.0, 6.0, 0.0]])
    assert EvidentialLoss(reg_lambda=1.0)(logits, targets) > EvidentialLoss(reg_lambda=0.0)(logits, targets)


def test_annealing_ramps_and_set_epoch_hook():
    loss = EvidentialLoss(reg_lambda=1.0, anneal_epochs=10)
    loss.set_epoch(1)
    assert abs(loss.lambda_now() - 0.1) < 1e-9
    loss.set_epoch(5)
    assert abs(loss.lambda_now() - 0.5) < 1e-9
    loss.set_epoch(30)
    assert loss.lambda_now() == 1.0
    assert EvidentialLoss(reg_lambda=0.3, anneal_epochs=0).lambda_now() == 0.3


def test_uniform_class_weights_match_unweighted():
    logits = torch.randn(8, 7)
    targets = torch.randint(0, 7, (8,))
    plain = EvidentialLoss(reg_lambda=0.1)(logits, targets)
    weighted = EvidentialLoss(reg_lambda=0.1, class_weight=torch.ones(7))(logits, targets)
    assert torch.isclose(plain, weighted)


def test_build_loss_evidential_branch_and_gradients():
    loss = build_loss({"type": "evidential", "class_weights": True, "edl_lambda": 0.2, "edl_anneal_epochs": 10},
                      class_weight=torch.ones(7))
    logits = torch.randn(4, 7, requires_grad=True)
    loss(logits, torch.tensor([0, 1, 2, 3])).backward()
    assert torch.isfinite(logits.grad).all()


def test_variance_scores_decompose_and_evidence_ordering():
    logits = np.array([[5.0, -5.0, -5.0], [0.5, 0.4, 0.3], [-5.0, -5.0, -5.0]])
    alpha = alpha_from_logits(logits)
    s = variance_scores(alpha)
    assert np.allclose(s["epistemic"] + s["aleatoric"], s["total"])
    assert s["evidence_only"][2] > s["evidence_only"][0]  # no evidence => most uncertain
    assert s["total"][1] > s["total"][0]  # spread-out mu_bar => larger trace
