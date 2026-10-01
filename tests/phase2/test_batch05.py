"""Batch 05's configuration and comparison rule.

The comparison is the point of the batch, and it is the thing most easily bent
after seeing numbers. So the threshold, the escalation band, the exclusion order
and the row ordering are all pinned here, and each test says what going wrong
would cost.
"""

import pytest

from phase2.batch02 import PROBE_CONFIG as BATCH02_CONFIG
from phase2.batch05 import (
    ARM_A_REPLICATION_FAILED,
    ARM_A_REPLICATION_FLOOR_PERCENT,
    ARMS,
    BATCH05_CONFIG,
    COMPUTE_NOT_DEPTH,
    DEPTH_NEEDS_UNSHARED_PARAMETERS,
    DEPTH_NOT_COMPUTE,
    DIFFERENCE_THRESHOLD_POINTS,
    INHERITED_FIELDS,
    UNRESOLVED,
    UNRESOLVED_BAND,
    WIDTH_BEATS_DEPTH,
    arm_is_excluded,
    arms_differ,
    compare,
    inherited_matches_batch03,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_three_arms_with_the_intended_families():
    assert [a[0] for a in ARMS] == ["A", "B", "C"]
    assert [a[1] for a in ARMS] == ["tesseract", "width_scaled", "unrolled"]
    assert all(a[2] == 4 for a in ARMS), "all arms are matched to K=4's compute"


def test_the_fix_is_on_and_non_affine_for_every_arm():
    assert BATCH05_CONFIG["fix"] == "per_step"
    assert BATCH05_CONFIG["per_step_norm_elementwise_affine"] is False


@pytest.mark.parametrize("field", INHERITED_FIELDS)
def test_curriculum_and_budget_are_batch03s_exactly(field):
    """Arm A must replicate Batch 03's K=4 cells, so nothing may drift."""
    assert BATCH05_CONFIG[field] == BATCH02_CONFIG[field]


def test_inherited_check_is_what_the_runner_gates_on():
    assert inherited_matches_batch03() is True
    assert BATCH05_CONFIG["steps"] == 35000
    assert BATCH05_CONFIG["target_t"] == 3


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------


def test_the_threshold_is_the_argued_five_points():
    """50x the 0.10-point within-arm spread Batch 03's K=4 cells showed."""
    assert DIFFERENCE_THRESHOLD_POINTS == 5.0


@pytest.mark.parametrize("a,b,expected", [
    (100.0, 100.0, False), (100.0, 99.9, False), (100.0, 95.0, False),
    (100.0, 94.9, True), (100.0, 0.0, True), (0.0, 100.0, True),
])
def test_arms_differ_is_symmetric_and_uses_a_strict_comparison(a, b, expected):
    assert arms_differ(a, b) is expected
    assert arms_differ(b, a) is expected


def test_a_missing_value_cannot_establish_a_difference_or_its_absence():
    assert arms_differ(None, 100.0) is None
    assert arms_differ(100.0, None) is None


# ---------------------------------------------------------------------------
# The outcome rows
# ---------------------------------------------------------------------------


def test_depth_not_compute_requires_a_beating_b_and_matching_c():
    result = compare({"A": 100.0, "B": 2.0, "C": 99.9})
    assert result["reading"] == DEPTH_NOT_COMPUTE
    assert result["supports_depth_claim"] is True


def test_all_arms_agreeing_is_compute_not_depth():
    result = compare({"A": 100.0, "B": 99.0, "C": 99.5})
    assert result["reading"] == COMPUTE_NOT_DEPTH
    assert result["supports_depth_claim"] is False


def test_c_ahead_of_a_means_depth_needs_unshared_parameters():
    result = compare({"A": 80.0, "B": 2.0, "C": 99.0})
    assert result["reading"] == DEPTH_NEEDS_UNSHARED_PARAMETERS
    assert result["supports_depth_claim"] is False


def test_b_ahead_of_a_is_reported_as_a_negative_result():
    result = compare({"A": 60.0, "B": 99.0, "C": 61.0})
    assert result["reading"] == WIDTH_BEATS_DEPTH
    assert result["supports_depth_claim"] is False


def test_a_negative_row_is_checked_before_the_positive_one():
    """A must not be declared the winner while another arm is actually ahead."""
    # A beats B by >5, but C beats A by >5. The C row must win.
    result = compare({"A": 70.0, "B": 2.0, "C": 99.0})
    assert result["reading"] == DEPTH_NEEDS_UNSHARED_PARAMETERS


def test_everything_in_the_middle_band_escalates_rather_than_being_adjudicated():
    low, high = UNRESOLVED_BAND
    assert (low, high) == (5.0, 95.0)
    result = compare({"A": 60.0, "B": 20.0, "C": 55.0})
    assert result["reading"] == UNRESOLVED
    assert result["supports_depth_claim"] is False
    assert "three seeds cannot support" in result["reason"]


def test_arm_a_failing_to_replicate_batch03_is_a_batch_failure():
    for value in (0.0, 10.0, 49.9):
        result = compare({"A": value, "B": 0.0, "C": 0.0})
        assert result["reading"] == ARM_A_REPLICATION_FAILED
        assert result["supports_depth_claim"] is False
    assert ARM_A_REPLICATION_FLOOR_PERCENT == 50.0


def test_arm_a_absent_is_also_a_batch_failure():
    result = compare({"B": 100.0, "C": 100.0})
    assert result["reading"] == ARM_A_REPLICATION_FAILED


def test_an_excluded_arm_takes_no_part_but_is_reported():
    result = compare({"A": 100.0, "B": 2.0, "C": 99.0}, excluded=["C"])
    assert result["reading"] == DEPTH_NOT_COMPUTE
    assert result["excluded"] == ["C"]


# ---------------------------------------------------------------------------
# Exclusion, in the pre-registered order
# ---------------------------------------------------------------------------


def test_a_broken_bound_excludes_an_arm_before_anything_else_is_considered():
    assert arm_is_excluded("LEARNABLE", "COMPOSITION", 100.0, False) == "post_training_bound_broken"
    # Even a degenerate, shortcut-attributed arm reports the bound first.
    assert arm_is_excluded("DEGENERATE", "SHORTCUT", None, False) == "post_training_bound_broken"


def test_degenerate_and_shortcut_arms_are_excluded():
    assert arm_is_excluded("DEGENERATE", "COMPOSITION", 0.0, True) == "degenerate"
    assert arm_is_excluded("LEARNABLE", "SHORTCUT", 30.0, True) == "attributed_to_shortcut"


def test_a_missing_exact_match_excludes_an_arm():
    assert arm_is_excluded("LEARNABLE", "COMPOSITION", None, True) == "no_exact_match_recorded"


def test_a_healthy_arm_is_not_excluded():
    assert arm_is_excluded("LEARNABLE", "COMPOSITION", 100.0, True) is None


def test_no_reading_claims_the_depth_result_without_arm_a_leading():
    """Sweep the space: supports_depth_claim only where A beats B and matches C."""
    for a in (0.0, 60.0, 100.0):
        for b in (0.0, 60.0, 100.0):
            for c in (0.0, 60.0, 100.0):
                result = compare({"A": a, "B": b, "C": c})
                if result["supports_depth_claim"]:
                    assert a - b > DIFFERENCE_THRESHOLD_POINTS
                    assert abs(a - c) <= DIFFERENCE_THRESHOLD_POINTS
