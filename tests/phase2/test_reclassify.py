"""Addendum 03: the majority-class floor and the collapse check."""

import numpy as np
import pytest

from phase2.probes import LEARNABLE, NOT_LEARNABLE, PARTIAL
from phase2.reclassify import (
    COLLAPSE_MAX_DISTINCT_TOKENS,
    DEGENERATE,
    amend_classification,
    distinct_predicted_tokens,
    is_degenerate,
    majority_class_percent,
    moved_under_arm5_criteria,
)

MAJORITY_T4 = 1.9244206112852664


def seed_row(seed, accuracy, above, learnable=False):
    return {"seed": seed, "best_val_token_accuracy": accuracy, "above_chance": above,
            "learnable": learnable, "above_chance_threshold_percent": 1.90}


def original(classification, rows, required=2):
    return {"classification": classification, "per_seed": rows,
            "seeds_above_chance": [r["seed"] for r in rows if r["above_chance"]],
            "bars": {"seeds_required": required}}


def test_majority_class_baseline_is_measured_from_the_split():
    targets = np.array([[0, 1, 1, 1], [2, 1, 3, 1]])          # token 1 is 5 of 8
    assert majority_class_percent(targets) == pytest.approx(62.5)


def test_majority_class_baseline_exceeds_uniform_chance_on_the_real_split():
    # The defect this addendum corrects: the bar sat below this figure.
    assert MAJORITY_T4 > 100 / 60


def test_collapse_threshold_catches_a_constant_predictor():
    constant = np.full((100, 17), 11)
    assert distinct_predicted_tokens(constant) == 1
    assert is_degenerate(1) and is_degenerate(COLLAPSE_MAX_DISTINCT_TOKENS)
    assert not is_degenerate(COLLAPSE_MAX_DISTINCT_TOKENS + 1)
    assert not is_degenerate(None), "an unmeasured cell is not assumed degenerate"


def test_a_constant_predictor_at_the_majority_class_no_longer_counts():
    """The exact Arm 2b situation: one token everywhere, scoring the modal class."""
    rows = [seed_row(0, 1.7664, False), seed_row(1, 1.7578, False), seed_row(2, MAJORITY_T4, True)]
    amended = amend_classification(original(PARTIAL, rows), MAJORITY_T4, {0: 2, 1: 1, 2: 1})
    assert amended["classification_original"] == PARTIAL
    assert amended["classification_amended"] == DEGENERATE
    assert amended["changed"] is True
    assert amended["seeds_above_chance_amended"] == []
    assert amended["degenerate_seeds"] == [0, 1, 2]


def test_both_conditions_are_required_to_be_above_chance():
    # Beats the old bar but not the majority class, and is not degenerate.
    rows = [seed_row(0, MAJORITY_T4 - 0.01, True)]
    amended = amend_classification(original(PARTIAL, rows), MAJORITY_T4, {0: 60})
    assert amended["per_seed"][0]["beats_majority_class"] is False
    assert amended["per_seed"][0]["above_chance_amended"] is False

    # Beats both and is not degenerate: survives.
    rows = [seed_row(0, MAJORITY_T4 + 0.01, True)]
    amended = amend_classification(original(PARTIAL, rows), MAJORITY_T4, {0: 60})
    assert amended["per_seed"][0]["above_chance_amended"] is True
    assert amended["classification_amended"] == PARTIAL


def test_a_genuine_crossing_survives_the_amendment():
    """Probe B K=8 seed 1: all 60 tokens used, beats the T=8 majority baseline."""
    majority_t8 = 1.8354
    rows = [seed_row(0, 1.7923, False), seed_row(1, 1.9273, True), seed_row(2, 1.8181, False)]
    amended = amend_classification(original(PARTIAL, rows), majority_t8, {0: 60, 1: 60, 2: 58})
    assert amended["classification_amended"] == PARTIAL
    assert amended["changed"] is False
    assert amended["seeds_above_chance_amended"] == [1]


def test_degeneracy_overrides_accuracy_even_at_the_learnable_bar():
    """"DEGENERATE regardless of accuracy" — a collapsed cell cannot be learnable."""
    rows = [seed_row(0, 90.0, True, learnable=True), seed_row(1, 90.0, True, learnable=True),
            seed_row(2, 90.0, True, learnable=True)]
    amended = amend_classification(original(LEARNABLE, rows), MAJORITY_T4, {0: 1, 1: 1, 2: 1})
    assert amended["classification_amended"] == DEGENERATE
    assert amended["seeds_meeting_learnable_bar_amended"] == []


def test_a_healthy_learnable_arm_is_unaffected():
    rows = [seed_row(s, 40.0, True, learnable=True) for s in (0, 1, 2)]
    amended = amend_classification(original(LEARNABLE, rows), MAJORITY_T4, {0: 60, 1: 59, 2: 60})
    assert amended["classification_amended"] == LEARNABLE
    assert amended["changed"] is False


RANKING = {DEGENERATE: 0, NOT_LEARNABLE: 0, PARTIAL: 1, LEARNABLE: 2}


def label_as_the_original_classifier_would(rows, required=2):
    """The pre-registered verdict for these rows, so fixtures stay self-consistent.

    A seed cannot be marked above-chance inside a report labelled NOT LEARNABLE;
    deriving the label removes that impossible combination from the test.
    """
    if sum(r["learnable"] for r in rows) >= required:
        return LEARNABLE
    return PARTIAL if any(r["above_chance"] for r in rows) else NOT_LEARNABLE


def test_the_amendment_can_only_tighten():
    """It must never turn a negative result into a positive one."""
    rows = [seed_row(s, 1.70, False) for s in (0, 1, 2)]
    amended = amend_classification(original(NOT_LEARNABLE, rows), MAJORITY_T4, {0: 60, 1: 60, 2: 60})
    assert amended["classification_amended"] == NOT_LEARNABLE

    cases = [
        ([seed_row(0, 1.95, True)], {0: 1}),                                   # above bar, collapsed
        ([seed_row(0, 1.95, True)], {0: 60}),                                  # above bar, healthy
        ([seed_row(0, MAJORITY_T4 - 0.01, True)], {0: 60}),                    # under the floor
        ([seed_row(s, 40.0, True, learnable=True) for s in (0, 1, 2)], {0: 1, 1: 1, 2: 1}),
        ([seed_row(s, 40.0, True, learnable=True) for s in (0, 1, 2)], {0: 60, 1: 60, 2: 60}),
        ([seed_row(s, 1.70, False) for s in (0, 1, 2)], {0: 60, 1: 60, 2: 60}),
    ]
    for rows, distinct in cases:
        label = label_as_the_original_classifier_would(rows)
        a = amend_classification(original(label, rows), MAJORITY_T4, distinct)
        assert RANKING[a["classification_amended"]] <= RANKING[label], (
            f"{label} -> {a['classification_amended']} loosened the verdict")


def test_arm5_movement_still_needs_two_seeds():
    one = {"1": {"classification_amended": PARTIAL, "seeds_above_chance_amended": [2]}}
    two = {"1": {"classification_amended": PARTIAL, "seeds_above_chance_amended": [0, 2]}}
    learn = {"1": {"classification_amended": LEARNABLE, "seeds_above_chance_amended": []}}
    degen = {"1": {"classification_amended": DEGENERATE, "seeds_above_chance_amended": []}}
    assert moved_under_arm5_criteria(one) is False
    assert moved_under_arm5_criteria(two) is True
    assert moved_under_arm5_criteria(learn) is True
    assert moved_under_arm5_criteria(degen) is False


def test_an_arm_moves_if_either_k_moves():
    mixed = {"1": {"classification_amended": NOT_LEARNABLE, "seeds_above_chance_amended": []},
             "8": {"classification_amended": PARTIAL, "seeds_above_chance_amended": [0, 1]}}
    assert moved_under_arm5_criteria(mixed) is True
