"""The pre-registered probe decision rule, applied mechanically."""

import math

import pytest
import yaml

from phase2.probes import (
    LEARNABLE,
    NOT_LEARNABLE,
    PARTIAL,
    _inverse_normal_cdf,
    above_chance_threshold,
    classify,
    evaluate_seed,
)

PREREG = yaml.safe_load(open("configs/phase2/probe_preregistration.yaml", encoding="utf-8"))
CHANCE_ACC = PREREG["chance"]["token_accuracy_percent"]
CHANCE_LOSS = PREREG["chance"]["loss_nats"]


def record(seed, accuracy, loss=None, final_loss=None, se=0.07, evaluations=80):
    return {"seed": seed,
            "best_val": {"token_accuracy": accuracy, "loss": loss if loss is not None else CHANCE_LOSS},
            "final_val": {"token_accuracy": accuracy,
                          "loss": final_loss if final_loss is not None else CHANCE_LOSS},
            "num_evaluations": evaluations, "per_example_token_accuracy_se": se}


def test_the_committed_thresholds_are_the_ones_that_were_specified():
    """The rule is only pre-registration if these numbers cannot drift."""
    learnable = PREREG["rule"]["learnable"]
    assert learnable["accuracy_percent_at_least"] == pytest.approx(1.6666666666666667 + 5.0)
    assert learnable["loss_nats_at_most"] == pytest.approx(math.log(60) - 0.1)
    assert learnable["seeds_required"] == 2
    assert CHANCE_ACC == pytest.approx(100 / 60) and CHANCE_LOSS == pytest.approx(math.log(60))
    assert PREREG["committed_before_execution"] is True
    assert PREREG["operational"]["held_out_split"] == "val" and PREREG["operational"]["test_split_used"] is False


def test_inverse_normal_cdf_matches_known_quantiles():
    for p, expected in [(0.5, 0.0), (0.975, 1.959964), (0.99, 2.326348), (0.995, 2.575829)]:
        assert _inverse_normal_cdf(p) == pytest.approx(expected, abs=1e-5)


def test_above_chance_bar_widens_with_more_evaluations():
    """Selecting the best of M evaluations must cost something, or noise passes."""
    one = above_chance_threshold(CHANCE_ACC, 0.07, 1)
    many = above_chance_threshold(CHANCE_ACC, 0.07, 80)
    assert CHANCE_ACC < one < many


def test_learnable_needs_two_of_three_seeds():
    two = [record(0, 7.0), record(1, 7.0), record(2, 1.67)]
    assert classify(two, PREREG)["classification"] == LEARNABLE
    one = [record(0, 7.0), record(1, 1.67), record(2, 1.67)]
    assert classify(one, PREREG)["classification"] == PARTIAL


def test_loss_arm_alone_reaches_learnable_and_is_flagged_as_such():
    """The loss bar is the weaker arm; a result resting on it must be visible."""
    below = math.log(60) - 0.15
    records = [record(s, 1.70, loss=below, final_loss=below) for s in (0, 1, 2)]
    outcome = classify(records, PREREG)
    assert outcome["classification"] == LEARNABLE
    assert outcome["learnable_by_loss_arm_only"] == [0, 1, 2]


def test_at_chance_on_every_seed_is_not_learnable():
    records = [record(s, CHANCE_ACC) for s in (0, 1, 2)]
    outcome = classify(records, PREREG)
    assert outcome["classification"] == NOT_LEARNABLE
    assert outcome["seeds_above_chance"] == []


def test_a_single_seed_barely_above_chance_is_partial_not_not_learnable():
    threshold = above_chance_threshold(CHANCE_ACC, 0.07, 80)
    records = [record(0, threshold + 0.01), record(1, CHANCE_ACC), record(2, CHANCE_ACC)]
    assert classify(records, PREREG)["classification"] == PARTIAL


def test_noise_below_the_corrected_bar_does_not_count_as_above_chance():
    """Without the multiplicity correction this seed would read as signal."""
    uncorrected = CHANCE_ACC + 1.96 * 0.07
    corrected = above_chance_threshold(CHANCE_ACC, 0.07, 80)
    assert uncorrected < corrected
    records = [record(0, uncorrected), record(1, CHANCE_ACC), record(2, CHANCE_ACC)]
    assert classify(records, PREREG)["classification"] == NOT_LEARNABLE


def test_missing_metrics_do_not_silently_count_as_learnable():
    broken = {"seed": 0, "best_val": {"token_accuracy": None, "loss": None},
              "final_val": {"token_accuracy": None, "loss": None},
              "num_evaluations": 80, "per_example_token_accuracy_se": 0.07}
    seed = evaluate_seed(broken, CHANCE_ACC, CHANCE_LOSS, 6.667, 3.994)
    assert not seed["learnable"] and not seed["above_chance"]


def test_probe_b_is_recorded_as_diagnostic_only():
    """Warm-starting must never quietly become the grid's training protocol."""
    probe_b = PREREG["probes"]["probe_b"]
    assert probe_b["warm_start"] is True and probe_b["diagnostic_only"] is True
    assert "must not become the training protocol" in probe_b["note"]
