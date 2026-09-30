"""Batch 04's configuration, exact-match reading and outcome table.

Batch 04 moves to T = 4, a primary depth that passes G2, so two things that were
load-bearing in Batch 03 must change and two must not.

Must change: no shortcut attribution at the target depth, because there is no
shallow candidate to attribute to; and exact match becomes a reported co-primary,
because at T = 3 a K = 1 cell cleared the token-accuracy bar at 64 % while
producing 0.15 % exact match.

Must not change: the classification rule, which predates every batch, and every
configuration field Batch 04 is supposed to inherit from Batch 03 — drift in one
of those would confound the comparison between the two depths.
"""

import pytest

from phase2.batch02 import SHORTCUT_CANDIDATE
from phase2.batch03 import BATCH03_CONFIG
from phase2.batch04 import (
    BATCH04_CONFIG,
    DIAGNOSTIC_T_VALUES,
    EXACT_MATCH_FLOOR_PERCENT,
    FIX_NOT_ESTABLISHED,
    GRID_RESUME_PROPOSED,
    INHERITED_FIELDS,
    LEARNABLE_WITH_EXACT_MATCH,
    NEEDS_INVESTIGATION,
    SHORTCUT_T_VALUES,
    T3_IS_THE_BOUNDARY,
    TOKEN_ACCURACY_ONLY,
    TOKEN_ONLY_NOT_A_DEPTH_RESULT,
    exact_match_reading,
    inherited_matches_batch03,
    outcome,
)

BOUND_OK = {1: True, 4: True, 8: True}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_depth_ladder_and_budget_are_the_intended_changes():
    assert BATCH04_CONFIG["target_t"] == 4
    assert BATCH04_CONFIG["curriculum_stages"] == [1, 2, 3, 4]
    assert BATCH04_CONFIG["steps"] == 40000
    assert BATCH04_CONFIG["k_values"] == [1, 4, 8]
    # Four 5,000-step rungs plus a 20,000-step target stage.
    assert 4 * BATCH04_CONFIG["steps_per_stage"] + BATCH04_CONFIG["target_steps"] == 40000


@pytest.mark.parametrize("field", INHERITED_FIELDS)
def test_every_inherited_field_matches_batch03(field):
    assert BATCH04_CONFIG[field] == BATCH03_CONFIG[field]


def test_the_residual_fix_is_still_on_and_non_affine():
    assert BATCH04_CONFIG["fix"] == "per_step"
    assert BATCH04_CONFIG["per_step_norm_elementwise_affine"] is False
    assert inherited_matches_batch03() is True


def test_the_inherited_list_covers_everything_but_the_intended_changes():
    intended = {"curriculum_stages", "target_t", "steps", "k_values"}
    covered = set(INHERITED_FIELDS) | intended
    assert set(BATCH04_CONFIG) <= covered, set(BATCH04_CONFIG) - covered


def test_diagnostic_depths_cover_the_whole_ladder_and_the_target():
    assert DIAGNOSTIC_T_VALUES == (1, 2, 3, 4)
    assert set(BATCH04_CONFIG["curriculum_stages"]) <= set(DIAGNOSTIC_T_VALUES)


def test_no_shortcut_candidate_at_the_target_depth_but_the_ladder_has_two():
    """T=4 passes G2; T=2 and T=3 fail it at 26.67%."""
    assert 4 not in SHORTCUT_CANDIDATE
    assert SHORTCUT_T_VALUES == (2, 3)


# ---------------------------------------------------------------------------
# Exact match as a reported co-primary
# ---------------------------------------------------------------------------


def test_exact_match_never_changes_a_classification():
    """It constrains interpretation, not the verdict."""
    for classification in ("NOT_LEARNABLE", "PARTIAL", "DEGENERATE"):
        reading = exact_match_reading(classification, 99.0)
        assert reading["reading"] is None
        assert reading["supports_depth_claim"] is False


def test_a_learnable_verdict_below_the_floor_is_token_accuracy_only():
    """The T=3 case: 64% token accuracy, 0.15% exact match."""
    reading = exact_match_reading("LEARNABLE", 0.1465)
    assert reading["reading"] == TOKEN_ACCURACY_ONLY
    assert reading["supports_depth_claim"] is False
    assert "no correct rings" in reading["reason"]


def test_a_learnable_verdict_at_or_above_the_floor_supports_the_depth_claim():
    reading = exact_match_reading("LEARNABLE", 100.0)
    assert reading["reading"] == LEARNABLE_WITH_EXACT_MATCH
    assert reading["supports_depth_claim"] is True


@pytest.mark.parametrize("value,supports", [
    (0.0, False), (0.9999, False), (1.0, True), (1.1, True), (100.0, True),
])
def test_the_floor_is_the_preregistered_one_percent(value, supports):
    assert EXACT_MATCH_FLOOR_PERCENT == 1.0
    assert exact_match_reading("LEARNABLE", value)["supports_depth_claim"] is supports


def test_an_absent_exact_match_supports_nothing():
    reading = exact_match_reading("LEARNABLE", None)
    assert reading["supports_depth_claim"] is False
    assert "absent measurement" in reading["reason"]


# ---------------------------------------------------------------------------
# The §8 outcome table
# ---------------------------------------------------------------------------


def test_learnable_with_correct_rings_proposes_resuming_the_grid():
    result = outcome({1: "NOT_LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.0, 4: 99.0, 8: 95.0}, BOUND_OK)
    assert result["outcome"] == GRID_RESUME_PROPOSED
    assert result["backed_by_exact_match_k"] == [4, 8]
    assert result["supports_depth_claim"] is True
    assert "withdrawn" in result["reason"]


def test_learnable_everywhere_but_no_correct_rings_is_not_a_depth_result():
    result = outcome({1: "LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 0.1, 4: 0.5, 8: 0.0}, BOUND_OK)
    assert result["outcome"] == TOKEN_ONLY_NOT_A_DEPTH_RESULT
    assert result["backed_by_exact_match_k"] == []
    assert result["supports_depth_claim"] is False


def test_nothing_learnable_makes_t3_the_boundary():
    result = outcome({1: "NOT_LEARNABLE", 4: "NOT_LEARNABLE", 8: "DEGENERATE"},
                     {1: 0.0, 4: 0.0, 8: 0.0}, BOUND_OK)
    assert result["outcome"] == T3_IS_THE_BOUNDARY
    assert result["supports_depth_claim"] is False


@pytest.mark.parametrize("unbounded_k", [1, 4, 8])
def test_a_broken_bound_at_any_k_gates_every_other_branch(unbounded_k):
    bounds = {**BOUND_OK, unbounded_k: False}
    result = outcome({1: "LEARNABLE", 4: "LEARNABLE", 8: "LEARNABLE"},
                     {1: 99.0, 4: 99.0, 8: 99.0}, bounds)
    assert result["outcome"] == FIX_NOT_ESTABLISHED
    assert result["unbounded_k"] == [unbounded_k]
    assert result["supports_depth_claim"] is False


def test_k1_succeeding_alone_inverts_batch03_and_is_flagged_not_reported():
    result = outcome({1: "LEARNABLE", 4: "NOT_LEARNABLE", 8: "NOT_LEARNABLE"},
                     {1: 99.0, 4: 0.0, 8: 0.0}, BOUND_OK)
    assert result["outcome"] == NEEDS_INVESTIGATION
    assert result["supports_depth_claim"] is False


def test_k1_succeeding_alongside_a_higher_k_is_a_normal_positive_result():
    result = outcome({1: "LEARNABLE", 4: "LEARNABLE", 8: "NOT_LEARNABLE"},
                     {1: 99.0, 4: 99.0, 8: 0.0}, BOUND_OK)
    assert result["outcome"] == GRID_RESUME_PROPOSED
    assert result["backed_by_exact_match_k"] == [1, 4]


def test_no_outcome_claims_a_depth_result_without_the_bound():
    for classification in ("LEARNABLE", "NOT_LEARNABLE", "DEGENERATE", "PARTIAL"):
        result = outcome({k: classification for k in (1, 4, 8)},
                         {k: 100.0 for k in (1, 4, 8)},
                         {k: False for k in (1, 4, 8)})
        assert result["supports_depth_claim"] is False
