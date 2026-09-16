"""Phase 2 metrics: chance normalisation, per-position accuracy, K*(T), Δ(T), statistics."""

import math

import numpy as np
import pytest

from phase2.metrics import (
    bootstrap_ci,
    chance_normalised,
    compute_metrics,
    delta,
    interaction_regression,
    k_star,
    mean_ci95,
    per_position_accuracy,
)


def test_chance_normalisation_endpoints():
    assert chance_normalised(100.0 / 60, 60) == pytest.approx(0.0)
    assert chance_normalised(100.0, 60) == pytest.approx(1.0)
    assert chance_normalised(0.0, 60) < 0


def test_compute_metrics_by_hand():
    targets = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    predictions = np.array([[1, 2, 3, 4], [5, 0, 7, 0]])
    m = compute_metrics(predictions, targets, vocab_size=60)
    assert m.num_examples == 2
    assert m.exact_match == 50.0
    assert m.token_accuracy == 75.0
    assert m.chance_normalised_token_accuracy == pytest.approx((0.75 - 1 / 60) / (1 - 1 / 60))
    assert m.per_position_accuracy == [100.0, 50.0, 100.0, 50.0]


def test_per_position_rejects_bad_shapes():
    with pytest.raises(ValueError):
        per_position_accuracy(np.zeros((2, 3)), np.zeros((2, 4)))


def test_k_star():
    scores = {1: 0.2, 2: 0.85, 4: 0.93, 8: 0.99}
    assert k_star(scores, 0.9) == 4
    assert k_star(scores, 0.999) is None
    assert k_star({1: None, 2: 0.95}, 0.9) == 2


def test_delta():
    assert delta({1: 0.25, 8: 0.75}, 8, 1) == 0.5
    assert delta({1: 0.25}, 8, 1) is None


def test_mean_ci95_known_values():
    ci = mean_ci95([1.0, 2.0, 3.0])
    half = 4.303 * 1.0 / math.sqrt(3)
    assert ci["mean"] == 2.0 and ci["low"] == pytest.approx(2 - half) and ci["high"] == pytest.approx(2 + half)
    assert mean_ci95([5.0]) == {"n": 1, "mean": 5.0, "low": None, "high": None}


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    values = np.random.default_rng(0).random(500)
    first, second = bootstrap_ci(values, seed=3), bootstrap_ci(values, seed=3)
    assert first == second and first["low"] <= first["mean"] <= first["high"]


def test_interaction_regression_recovers_planted_effect():
    rows = []
    for seed in (0, 1, 2):
        for k in (1, 2, 4, 8):
            for t in (1, 2, 4, 8):
                lk, lt = math.log2(k), math.log2(t)
                rows.append({"seed": seed, "k": k, "t": t, "score": 0.1 * seed + 0.2 * lk - 0.3 * lt + 0.05 * lk * lt})
    result = interaction_regression(rows)
    assert result["log2K_x_log2T"] == pytest.approx(0.05)
    assert result["log2K"] == pytest.approx(0.2) and result["log2T"] == pytest.approx(-0.3)


def test_t_intervals_are_never_narrower_than_the_true_t_quantile():
    """Unlisted degrees of freedom must round df down, not up to the normal limit."""
    from phase2.metrics import _t_critical

    assert _t_critical(2) == 4.303           # 3 seeds, the protocol's case
    assert _t_critical(12) >= 2.179          # true t_.975(12)
    assert _t_critical(35) >= 2.030          # true t_.975(35); previously 1.96
    assert _t_critical(50) >= 2.009          # true t_.975(50); previously 1.96
    assert _t_critical(100) >= 1.984         # true t_.975(100); previously 1.96
    assert _t_critical(1000) == 1.96
    for df in range(1, 200):
        assert _t_critical(df) >= _t_critical(df + 1)
