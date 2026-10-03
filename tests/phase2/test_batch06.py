"""Batch 06's configuration and outcome table.

Two things matter most here. The rung lengths must be non-uniform in depth but
identical across K — tuning them per K is the error Batch 06 exists to correct,
so making it per-K would defeat the batch. And the outcome table must carry the
convergence row Batch 05 showed was missing from every table in this project,
and that row must only ever withhold a comparison, never produce one.
"""

import pytest

from phase2.batch02 import stage_boundary_steps
from phase2.batch04 import BATCH04_CONFIG, EXACT_MATCH_FLOOR_PERCENT
from phase2.batch06 import (
    BATCH06_CONFIG,
    BATCH_FAILURE,
    FIX_NOT_ESTABLISHED,
    INHERITED_FIELDS,
    LEARNABLE_STILL_RISING,
    PLATEAU_SLOPE_THRESHOLD,
    PLATEAU_WITH_EXACT_MATCH,
    RUNG_STEPS,
    TARGET_STEPS,
    TOKEN_ONLY,
    has_plateaued,
    inherited_matches_batch04,
    outcome,
)

KS = (1, 4, 8)
BOUND_OK = {k: True for k in KS}
ALL_PLATEAU = {k: True for k in KS}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_two_stage_lengths_are_the_intended_changes():
    assert RUNG_STEPS == [5000, 5000, 15000, 5000]
    assert TARGET_STEPS == 60000
    assert BATCH06_CONFIG["steps"] == 90000 == sum(RUNG_STEPS) + TARGET_STEPS


def test_the_t3_rung_is_three_times_the_others_and_derived_from_measurement():
    """Batch 03's slowest K=1 seed needed 12,000 steps to pass the T=3 ceiling."""
    t1, t2, t3, t4 = RUNG_STEPS
    assert t3 == 15000 and t3 > 12000, "must clear Batch 03's slowest K=1 seed with margin"
    assert t1 == t2 == t4 == 5000, "the other rungs are ample at every K and stay put"


@pytest.mark.parametrize("field", INHERITED_FIELDS)
def test_every_inherited_field_matches_batch04(field):
    assert BATCH06_CONFIG[field] == BATCH04_CONFIG[field]


def test_rung_lengths_are_uniform_across_k_even_though_uneven_in_depth():
    """Per-K tuning is the error being corrected, so there is one ladder for all K."""
    assert inherited_matches_batch04() is True
    assert BATCH06_CONFIG["k_values"] == [1, 4, 8]
    assert isinstance(RUNG_STEPS, list) and all(isinstance(v, int) for v in RUNG_STEPS)
    # One ladder, not a mapping from K to a ladder.
    assert "rung_steps_by_k" not in BATCH06_CONFIG


def test_boundaries_are_the_preregistered_ones():
    assert stage_boundary_steps([1, 2, 3, 4], RUNG_STEPS, TARGET_STEPS) == [5000, 10000, 25000, 30000, 90000]


# ---------------------------------------------------------------------------
# The plateau test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slope,expected", [
    (-0.5, True), (0.0, True), (0.09, True),
    (0.1, False), (1.0, False), (17.694, False),
])
def test_plateau_uses_the_preregistered_threshold(slope, expected):
    assert PLATEAU_SLOPE_THRESHOLD == 0.1
    assert has_plateaued(slope) is expected


def test_an_absent_slope_cannot_establish_convergence():
    assert has_plateaued(None) is None


# ---------------------------------------------------------------------------
# The outcome table
# ---------------------------------------------------------------------------


def test_a_plateau_with_correct_rings_proposes_the_grid_resume_draft():
    result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.0, 4: 60.0, 8: 40.0}, ALL_PLATEAU, BOUND_OK)
    assert result["outcome"] == PLATEAU_WITH_EXACT_MATCH
    assert result["supports_depth_claim"] is True
    assert result["plateaued_k"] == [4, 8]


def test_correct_rings_but_still_rising_is_not_an_achieved_level():
    """Batch 04's defect: a number quoted as a level when the budget ran out."""
    result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.0, 4: 60.0, 8: 40.0},
                     {1: True, 4: False, 8: False}, BOUND_OK)
    assert result["outcome"] == LEARNABLE_STILL_RISING
    assert result["supports_depth_claim"] is False
    assert result["plateaued_k"] == []


def test_learnable_without_correct_rings_is_token_accuracy_only():
    result = outcome({1: "LEARNABLE", 4: "LEARNABLE", 8: "NOT_LEARNABLE"},
                     {1: 0.1, 4: 0.5, 8: 0.0}, ALL_PLATEAU, BOUND_OK)
    assert result["outcome"] == TOKEN_ONLY
    assert result["supports_depth_claim"] is False


def test_nothing_learnable_contradicts_batch04_and_is_a_batch_failure():
    result = outcome({k: "NOT_LEARNABLE" for k in KS}, {k: 0.0 for k in KS}, ALL_PLATEAU, BOUND_OK)
    assert result["outcome"] == BATCH_FAILURE
    assert result["comparison_withheld"] is True


@pytest.mark.parametrize("unbounded", [1, 4, 8])
def test_a_broken_bound_gates_everything(unbounded):
    bounds = {**BOUND_OK, unbounded: False}
    result = outcome({k: "LEARNABLE" for k in KS}, {k: 99.0 for k in KS}, ALL_PLATEAU, bounds)
    assert result["outcome"] == FIX_NOT_ESTABLISHED
    assert result["comparison_withheld"] is True


# ---------------------------------------------------------------------------
# The row Batch 05 showed was missing
# ---------------------------------------------------------------------------


def test_mixed_convergence_withholds_the_comparison_without_changing_the_verdict():
    """Batch 05 §3: comparing a plateaued arm against a still-rising one measures
    the budget as much as the arm. The row may only withhold, never produce."""
    result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.0, 4: 60.0, 8: 40.0},
                     {1: True, 4: True, 8: False}, BOUND_OK)
    # The per-K verdict still stands — it is per-K, not comparative.
    assert result["outcome"] == PLATEAU_WITH_EXACT_MATCH
    assert result["supports_depth_claim"] is True
    # But the cross-K comparison is withheld.
    assert result["comparison_withheld"] is True
    assert result["still_rising_k"] == [8]


def test_uniform_convergence_does_not_withhold_the_comparison():
    result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.0, 4: 60.0, 8: 40.0}, ALL_PLATEAU, BOUND_OK)
    assert result["comparison_withheld"] is False


def test_an_unrecorded_slope_withholds_the_comparison():
    result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.0, 4: 60.0, 8: 40.0},
                     {1: True, 4: True, 8: None}, BOUND_OK)
    assert result["comparison_withheld"] is True


def test_convergence_is_only_assessed_over_k_values_that_learned_something():
    """A K at chance has no trajectory, so its slope must not withhold anything."""
    result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.0, 4: 60.0, 8: 40.0},
                     {1: None, 4: True, 8: True}, BOUND_OK)
    assert result["comparison_withheld"] is False


def test_no_outcome_supports_the_depth_claim_without_a_plateau_and_rings():
    for em in (0.0, 0.5, 60.0):
        for plateau in (True, False, None):
            result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "NOT_LEARNABLE"},
                             {1: 0.0, 4: em, 8: 0.0}, {1: None, 4: plateau, 8: None}, BOUND_OK)
            if result["supports_depth_claim"]:
                assert em >= EXACT_MATCH_FLOOR_PERCENT and plateau is True
