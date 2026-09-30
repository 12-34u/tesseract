"""Batch 03's configuration, trend reading and outcome table.

Batch 03's whole purpose is to make the Batch 02 K comparison interpretable, and
that purpose fails in two specific ways these tests are here to prevent.

If any field other than the fix and the K range differs from Batch 02, the
comparison between the two batches is confounded by the drift instead of being
controlled for the confound — so the inherited fields are compared one by one.

If the trend across K were read after seeing the numbers, "is there a trend"
would be decidable either way on three seeds with a 31-point spread. So the
reading is a pure function fixed before any cell ran, keyed on learnability and
never on mean accuracy, and these tests pin every branch of it.
"""

import pytest

from phase2.batch02 import PROBE_CONFIG as BATCH02_CONFIG
from phase2.batch03 import (
    BATCH02_K1_SEED_SPREAD_POINTS,
    BATCH03_CONFIG,
    BATCH_FAILED,
    BATCH_FAILURE,
    CONFOUND_ATTRIBUTED,
    FIX_NOT_ESTABLISHED,
    GROWTH_LIMIT,
    INHERITED_FIELDS,
    K_INDEPENDENT,
    MONOTONE_DECLINE,
    NEGATIVE_K_STANDS,
    NON_MONOTONE,
    NOT_A_COMPOSITION_RESULT,
    accuracy_gap_is_meaningful,
    bound_held,
    inherited_matches_batch02,
    outcome,
    trend_reading,
)

# ---------------------------------------------------------------------------
# One change against Batch 02, and the K range
# ---------------------------------------------------------------------------


def test_the_fix_is_on_and_non_affine():
    """Affine would bound the state only at initialisation, and cost parameters."""
    assert BATCH03_CONFIG["fix"] == "per_step"
    assert BATCH03_CONFIG["per_step_norm_elementwise_affine"] is False
    assert BATCH02_CONFIG["fix"] == "none", "Batch 02 ran fix=none; that is why Batch 03 exists"


def test_k_range_is_widened_so_a_trend_can_be_read():
    assert BATCH03_CONFIG["k_values"] == [1, 2, 4, 8]
    assert BATCH02_CONFIG["k_values"] == [1, 8]


@pytest.mark.parametrize("field", INHERITED_FIELDS)
def test_every_inherited_field_matches_batch02(field):
    assert BATCH03_CONFIG[field] == BATCH02_CONFIG[field]


def test_inherited_check_is_what_the_runner_can_gate_on():
    assert inherited_matches_batch02() is True


def test_the_inherited_field_list_covers_everything_but_the_intended_changes():
    """A field absent from both the list and the change set would drift unnoticed."""
    intended_changes = {"fix", "per_step_norm_elementwise_affine", "k_values"}
    covered = set(INHERITED_FIELDS) | intended_changes
    assert set(BATCH03_CONFIG) <= covered, set(BATCH03_CONFIG) - covered
    assert set(BATCH02_CONFIG) <= covered, set(BATCH02_CONFIG) - covered


def test_budget_and_curriculum_are_batch02s_exactly():
    assert BATCH03_CONFIG["steps"] == 35000
    assert BATCH03_CONFIG["curriculum_stages"] == [1, 2, 3]
    assert BATCH03_CONFIG["steps_per_stage"] == 5000
    assert BATCH03_CONFIG["target_steps"] == 20000
    assert BATCH03_CONFIG["target_t"] == 3


# ---------------------------------------------------------------------------
# The post-training bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio,expected", [
    (1.0, True), (1.0000, True), (1.049, True),
    (1.05, False), (1.06, False), (17.7, False),
    (None, False),
])
def test_bound_held_uses_arm2c_limit_and_treats_absent_as_not_held(ratio, expected):
    """An absent measurement shows nothing, so it cannot license a conclusion."""
    assert bound_held(ratio) is expected
    assert GROWTH_LIMIT == 1.05


# ---------------------------------------------------------------------------
# The trend reading
# ---------------------------------------------------------------------------


def test_every_k_learnable_is_read_as_k_independent_not_as_a_boundary():
    reading = trend_reading([1, 2, 4, 8])
    assert reading["reading"] == K_INDEPENDENT
    assert reading["threshold_between"] is None


@pytest.mark.parametrize("learnable,bracket", [
    ([1], [1, 2]),
    ([1, 2], [2, 4]),
    ([1, 2, 4], [4, 8]),
])
def test_a_prefix_is_monotone_decline_and_brackets_the_threshold(learnable, bracket):
    reading = trend_reading(learnable)
    assert reading["reading"] == MONOTONE_DECLINE
    assert reading["threshold_between"] == bracket


@pytest.mark.parametrize("learnable", [[1, 4], [2, 4], [1, 2, 8], [8]])
def test_a_non_prefix_pattern_supports_no_threshold_claim(learnable):
    reading = trend_reading(learnable)
    assert reading["reading"] == NON_MONOTONE
    assert reading["threshold_between"] is None


def test_no_k_learnable_is_a_batch_failure_not_a_finding_about_k():
    """Batch 02's K=1 was LEARNABLE with the fix OFF, so K=1 failing here is a fault."""
    reading = trend_reading([])
    assert reading["reading"] == BATCH_FAILURE
    assert "Batch 02" in reading["reason"]


def test_trend_reading_ignores_order_and_duplicates_in_its_input():
    assert trend_reading([2, 1, 1])["reading"] == trend_reading([1, 2])["reading"]
    assert trend_reading([2, 1, 1])["threshold_between"] == [2, 4]


def test_accuracy_gaps_below_batch02s_seed_spread_are_not_trends():
    """Three seeds at one K already varied by 31.2 points in Batch 02."""
    assert BATCH02_K1_SEED_SPREAD_POINTS == pytest.approx(31.2213, abs=1e-3)
    assert accuracy_gap_is_meaningful(10.0) is False
    assert accuracy_gap_is_meaningful(31.0) is False
    assert accuracy_gap_is_meaningful(40.0) is True
    assert accuracy_gap_is_meaningful(None) is False


# ---------------------------------------------------------------------------
# The §8 outcome table
# ---------------------------------------------------------------------------


def test_k8_learnable_by_composition_attributes_the_batch02_failure_to_the_confound():
    result = outcome("LEARNABLE", "COMPOSITION", True, "LEARNABLE")
    assert result["outcome"] == CONFOUND_ATTRIBUTED
    assert result["answers_the_confound_question"] is True


@pytest.mark.parametrize("classification", ["NOT_LEARNABLE", "DEGENERATE", "PARTIAL"])
def test_k8_still_failing_with_the_bound_held_excludes_the_confound(classification):
    result = outcome(classification, None, True, "LEARNABLE")
    assert result["outcome"] == NEGATIVE_K_STANDS
    assert result["answers_the_confound_question"] is True


@pytest.mark.parametrize("attribution", ["SHORTCUT", "UNRESOLVED", None])
def test_k8_above_the_bar_without_composition_is_not_a_confound_attribution(attribution):
    result = outcome("LEARNABLE", attribution, True, "LEARNABLE")
    assert result["outcome"] == NOT_A_COMPOSITION_RESULT
    assert result["answers_the_confound_question"] is False


@pytest.mark.parametrize("classification", ["LEARNABLE", "NOT_LEARNABLE", "DEGENERATE"])
def test_a_broken_bound_gates_every_other_branch(classification):
    """A cell whose stream was not bounded is not a fix-on cell, whatever it scored."""
    result = outcome(classification, "COMPOSITION", False, "LEARNABLE")
    assert result["outcome"] == FIX_NOT_ESTABLISHED
    assert result["answers_the_confound_question"] is False


@pytest.mark.parametrize("k1", ["NOT_LEARNABLE", "DEGENERATE", "PARTIAL", None])
def test_k1_not_reproducing_batch02_is_a_batch_failure(k1):
    result = outcome("NOT_LEARNABLE", None, True, k1)
    assert result["outcome"] == BATCH_FAILED
    assert result["answers_the_confound_question"] is False


def test_the_bound_check_precedes_the_k1_check():
    """Order is pre-registered: an unbounded K=8 answers nothing even if K=1 also failed."""
    assert outcome("NOT_LEARNABLE", None, False, "NOT_LEARNABLE")["outcome"] == FIX_NOT_ESTABLISHED


def test_no_outcome_claims_to_answer_the_confound_question_without_the_bound():
    for classification in ("LEARNABLE", "NOT_LEARNABLE", "DEGENERATE", "PARTIAL"):
        for attribution in ("COMPOSITION", "SHORTCUT", "UNRESOLVED", None):
            result = outcome(classification, attribution, False, "LEARNABLE")
            assert result["answers_the_confound_question"] is False
