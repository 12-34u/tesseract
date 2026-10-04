"""Batch 07's corrected plateau test, outcome table and conditional follow-up.

The plateau test is the piece with the most history: Batch 06's version
misclassified ceiling noise as a trend and the report had to say so rather than
override it. The corrected test is a conjunction, and the tests below pin that
BOTH conjuncts are load-bearing — each alone misclassifies a real cell from a real
batch, and those cells are used as the fixtures.

The outcome table's A-ahead-of-both row exists because Batch 05 lacked it and
returned UNRESOLVED when the pattern occurred. The follow-up trigger is pinned to
fire only where a rate check has something to rescue.
"""

import pytest

from phase2.batch05 import DIFFERENCE_THRESHOLD_POINTS
from phase2.batch07 import (
    ARM_A_REPRODUCTION_EXACT_MATCH,
    ARM_A_REPRODUCTION_FAILED,
    ARMS,
    A_AHEAD_OF_BOTH,
    BATCH07_CONFIG,
    COMPUTE_NOT_DEPTH,
    DEPTH_NEEDS_UNSHARED_PARAMETERS,
    DEPTH_NOT_COMPUTE,
    PHASE1_LR,
    PHASE2_LR,
    PLATEAU_SLOPE_THRESHOLD,
    PLATEAU_T_THRESHOLD,
    RUNG_STEPS,
    TARGET_STEPS,
    TOTAL_STEPS,
    UNRESOLVED,
    WIDTH_BEATS_DEPTH,
    compare,
    final_slope,
    has_plateaued,
    phase2_reading,
    phase2_triggered,
    slope_with_se,
)

#: Real cells from Batches 05 and 06, as (slope /1k, t) with the verdict the
#: trajectory warrants. These are the fixtures because a synthetic number cannot
#: show that the conjunction is needed.
REAL_CELLS = [
    # Batch 06 K=8: ceiling noise. Magnitude passes, significance does not.
    ("B06 K=8 s0", 0.322, 1.78, True),
    ("B06 K=8 s1", 0.121, 0.88, True),
    ("B06 K=8 s2", 0.148, 0.62, True),
    # Batch 06 at chance: tiny but consistent drift. Significance passes, magnitude does not.
    ("B06 K=1 s1", 0.037, 2.31, True),
    ("B06 K=4 s1", 0.036, 2.13, True),
    # Genuinely rising: both pass.
    ("B06 K=1 s2", 0.403, 12.51, False),
    ("B05 arm B s1", 1.884, 25.49, False),
    ("B05 arm C s0", 6.701, 23.28, False),
    ("B05 arm C s1", 17.694, 15.12, False),
    # Converged, negative slope.
    ("B05 arm A s0", -0.045, -0.65, True),
    ("B05 arm A s2", -0.210, -0.95, True),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_arms_and_budget_are_batch06s_ladder_at_k8():
    assert [a[0] for a in ARMS] == ["A", "B", "C"]
    assert [a[1] for a in ARMS] == ["tesseract", "width_scaled", "unrolled"]
    assert all(a[2] == 8 for a in ARMS), "every arm is matched to K=8's compute"
    assert RUNG_STEPS == [5000, 5000, 15000, 5000]
    assert TARGET_STEPS == 60000
    assert TOTAL_STEPS == 90000 == BATCH07_CONFIG["steps"]
    assert BATCH07_CONFIG["target_t"] == 4


def test_the_optimiser_configuration_is_in_the_config_block():
    """Batch 05 Note 01 §4: no batch report has ever held these, and this batch
    varies the rate across phases, so the artifact would be unreadable without."""
    for field in ("optimizer", "phase1_learning_rate", "phase2_learning_rate",
                  "weight_decay", "batch_size"):
        assert field in BATCH07_CONFIG
    assert PHASE1_LR == 1e-3 and PHASE2_LR == 3e-4


def test_the_fix_is_on_and_non_affine():
    assert BATCH07_CONFIG["fix"] == "per_step"
    assert BATCH07_CONFIG["per_step_norm_elementwise_affine"] is False


# ---------------------------------------------------------------------------
# The corrected plateau test — both conjuncts are load-bearing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,slope,t,plateaued", REAL_CELLS)
def test_the_corrected_test_classifies_every_real_cell_correctly(label, slope, t, plateaued):
    stats = {"slope_points_per_1000_steps": slope, "standard_error_per_1000_steps": abs(slope / t) if t else 1.0,
             "t_statistic": t}
    assert has_plateaued(stats) is plateaued, label


def test_the_magnitude_conjunct_is_needed():
    """Without it, a 0.037/1k drift at chance with t=2.31 reads as rising."""
    drift = {"slope_points_per_1000_steps": 0.037, "standard_error_per_1000_steps": 0.016, "t_statistic": 2.31}
    assert drift["t_statistic"] > PLATEAU_T_THRESHOLD, "significance alone would call this rising"
    assert drift["slope_points_per_1000_steps"] < PLATEAU_SLOPE_THRESHOLD
    assert has_plateaued(drift) is True


def test_the_significance_conjunct_is_needed():
    """Without it, Batch 06's ceiling noise at 0.322/1k reads as rising — which is
    exactly the misclassification Batch 06 §3 had to report."""
    noise = {"slope_points_per_1000_steps": 0.322, "standard_error_per_1000_steps": 0.180, "t_statistic": 1.78}
    assert noise["slope_points_per_1000_steps"] >= PLATEAU_SLOPE_THRESHOLD, "magnitude alone would call this rising"
    assert noise["t_statistic"] < PLATEAU_T_THRESHOLD
    assert has_plateaued(noise) is True


def test_an_unfittable_slope_cannot_establish_convergence():
    assert has_plateaued(None) is None
    assert has_plateaued({"slope_points_per_1000_steps": None, "t_statistic": None}) is None
    assert slope_with_se([1, 2], [1, 2]) is None, "fewer than three points"
    assert slope_with_se([5, 5, 5], [1, 2, 3]) is None, "no spread in x"


def test_slope_statistics_are_computed_per_1000_steps():
    stats = slope_with_se([0, 1000, 2000, 3000], [0.0, 1.0, 2.0, 3.0])
    assert stats["slope_points_per_1000_steps"] == pytest.approx(1.0)
    assert stats["num_points"] == 4


def test_final_slope_uses_only_the_window():
    curve = [{"step": s, "val_token_accuracy": 50.0} for s in range(0, 90001, 250)]
    curve[-1]["val_token_accuracy"] = 50.0
    stats = final_slope(curve)
    assert stats["num_points"] == 11, "2,500 steps at an eval every 250 is 11 points"


# ---------------------------------------------------------------------------
# The outcome table
# ---------------------------------------------------------------------------


def test_a_ahead_of_both_is_its_own_row():
    """The row Batch 05 lacked, which made it return UNRESOLVED."""
    result = compare({"A": 98.78, "B": 0.0, "C": 0.0})
    assert result["reading"] == A_AHEAD_OF_BOTH
    assert result["supports_depth_claim"] is True
    assert "weight sharing helps" in result["reason"]


def test_a_matching_c_and_both_beating_b_is_depth_not_compute():
    result = compare({"A": 98.0, "B": 0.0, "C": 97.0})
    assert result["reading"] == DEPTH_NOT_COMPUTE
    assert result["supports_depth_claim"] is True


def test_all_arms_agreeing_is_compute_not_depth():
    result = compare({"A": 98.0, "B": 97.0, "C": 97.5})
    assert result["reading"] == COMPUTE_NOT_DEPTH
    assert result["supports_depth_claim"] is False


def test_negative_rows_are_checked_before_the_positive_ones():
    assert compare({"A": 60.0, "B": 0.0, "C": 99.0})["reading"] == DEPTH_NEEDS_UNSHARED_PARAMETERS
    assert compare({"A": 60.0, "B": 99.0, "C": 61.0})["reading"] == WIDTH_BEATS_DEPTH


def test_everything_in_the_middle_band_escalates():
    assert compare({"A": 60.0, "B": 20.0, "C": 55.0})["reading"] == UNRESOLVED


def test_a_missing_arm_a_is_a_reproduction_failure():
    assert compare({"B": 99.0, "C": 99.0})["reading"] == ARM_A_REPRODUCTION_FAILED


def test_no_reading_supports_the_depth_claim_unless_a_leads():
    for a in (0.0, 50.0, 99.0):
        for b in (0.0, 50.0, 99.0):
            for c in (0.0, 50.0, 99.0):
                result = compare({"A": a, "B": b, "C": c})
                if result["supports_depth_claim"]:
                    assert a - b > DIFFERENCE_THRESHOLD_POINTS
                    assert c is None or a >= c - DIFFERENCE_THRESHOLD_POINTS


# ---------------------------------------------------------------------------
# The conditional follow-up
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reading", [A_AHEAD_OF_BOTH, DEPTH_NOT_COMPUTE])
def test_the_follow_up_fires_only_where_arm_a_leads(reading):
    assert phase2_triggered(reading, True)["triggered"] is True


@pytest.mark.parametrize("reading", [
    COMPUTE_NOT_DEPTH, DEPTH_NEEDS_UNSHARED_PARAMETERS, WIDTH_BEATS_DEPTH,
    UNRESOLVED, ARM_A_REPRODUCTION_FAILED, None,
])
def test_the_follow_up_does_not_fire_where_a_rate_check_has_nothing_to_rescue(reading):
    result = phase2_triggered(reading, True)
    assert result["triggered"] is False
    assert "nothing" in result["reason"] or "does not place arm A ahead" in result["reason"]


def test_the_follow_up_does_not_fire_on_a_withheld_comparison():
    """12.6 GPU-h requires knowing what it is checking."""
    result = phase2_triggered(A_AHEAD_OF_BOTH, False)
    assert result["triggered"] is False
    assert "withheld" in result["reason"]


def test_the_follow_up_reading_detects_a_rate_confound():
    result = phase2_reading({"A": 98.0, "B": 0.0, "C": 0.0}, {"B": 96.0, "C": 1.0})
    assert result["reading"] == "RATE_CONFOUNDED"
    assert result["overturns_phase1"] is True
    assert "a proper rate search" in result["reason"].lower()


def test_the_follow_up_reading_separates_matters_from_overturns():
    result = phase2_reading({"A": 98.0, "B": 0.0, "C": 0.0}, {"B": 40.0, "C": 1.0})
    assert result["reading"] == "RATE_MATTERS_BUT_ORDERING_HOLDS"
    assert result["overturns_phase1"] is False


def test_the_follow_up_reading_can_exonerate_the_rate():
    result = phase2_reading({"A": 98.0, "B": 0.0, "C": 0.0}, {"B": 1.0, "C": 0.5})
    assert result["reading"] == "RATE_DOES_NOT_EXPLAIN_THE_GAP"
    assert result["overturns_phase1"] is False


def test_the_reproduction_target_is_batch06s_recorded_value():
    assert ARM_A_REPRODUCTION_EXACT_MATCH == 98.7793
