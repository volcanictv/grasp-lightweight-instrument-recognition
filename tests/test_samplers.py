import torch

from surgical_ai.training.samplers import build_sampler, compute_sample_weights


def test_sample_with_rare_class_gets_higher_weight():
    # 3 classes; class 0 common (appears 3x), class 2 rare (appears 1x).
    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    label_counts = labels.sum(dim=0)
    weights = compute_sample_weights(labels, label_counts)
    assert weights[3] > weights[0]


def test_sample_with_rare_and_common_class_takes_max_not_average():
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    label_counts = labels.sum(dim=0)
    weights = compute_sample_weights(labels, label_counts)
    # row 2 has both a common (freq=3) and rare (freq=1) class active;
    # its weight should equal the rare class's inverse frequency, not a
    # blend, so co-occurrence with a common class never dilutes the boost.
    assert weights[2] == 1.0 / label_counts[1]


def test_build_sampler_none_returns_none():
    samples = [("a.jpg", torch.tensor([1.0, 0.0]))]
    assert build_sampler("none", samples) is None


def test_build_sampler_weighted_returns_sampler_of_correct_length():
    samples = [
        ("a.jpg", torch.tensor([1.0, 0.0])),
        ("b.jpg", torch.tensor([0.0, 1.0])),
    ]
    sampler = build_sampler("weighted", samples)
    assert len(list(sampler)) == len(samples)


def test_build_sampler_unknown_mode_fails_loudly():
    import pytest

    with pytest.raises(ValueError):
        build_sampler("oversample", [("a.jpg", torch.tensor([1.0]))])
