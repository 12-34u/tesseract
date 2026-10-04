"""Batch 07 — is K = 8's T = 4 result depth or compute? (PHASE2_BATCH_07_PREREG.md)

Batch 06 solved T = 4 at K = 8 in 3 of 3 seeds, at matched STEPS. At 90,000 steps
K = 8 performs eight times K = 1's block executions, so the result is consistent
with depth or with eight times the compute. This batch holds forward compute fixed
and varies how it is spent, at a primary benchmark depth.

    A  tesseract K=8      837,436 params   215,615,488 FLOPs   8 block calls
    B  width_scaled       6,383,992        215,624,192         1 block call
    C  unrolled L=8       6,365,756        215,615,488         8 block calls

Arm A is 7.6x smaller than either alternative — twice Batch 05's ratio at K = 4,
because the shared block's count is K-invariant while the alternatives' grows
with depth.

Three things differ from Batch 05's design, each forced by a defect found since.

**The plateau test is corrected.** Batch 06 §3 showed a bare magnitude threshold
misclassifying noise at the ceiling as a rising trend. `has_plateaued` now
requires the slope to clear the magnitude floor AND be significant at t > 2.
Pre-registered for Batch 07 onward and not retroactive.

**The outcome table has a row for A ahead of both.** Batch 05 returned
UNRESOLVED because its table lacked one, and the pattern occurred.

**The learning-rate check is conditional.** Phase 1 runs arms B and C at the
inherited 1e-3. Phase 2 runs them at 3e-4, automatically and only if arm A beats
both in phase 1 — because that is the outcome whose interpretation the rate
confound would undermine.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from phase2.batch03 import GROWTH_LIMIT, bound_held  # noqa: F401  (re-exported)
from phase2.batch05 import DIFFERENCE_THRESHOLD_POINTS, UNRESOLVED_BAND

#: One arm per (id, family, depth, label). Depth is the block-execution count for
#: the recursive and unrolled families, and the matching target for width_scaled.
ARMS = (
    ("A", "tesseract", 8, "shared recursion, K=8"),
    ("B", "width_scaled", 8, "width-scaled K=1, compute-matched to K=8"),
    ("C", "unrolled", 8, "8 unrolled blocks"),
)

#: Batch 06's ladder and budget, unchanged — which is what makes arm A reusable.
RUNG_STEPS = [5000, 5000, 15000, 5000]
TARGET_STEPS = 60000
CURRICULUM_STAGES = [1, 2, 3, 4]
TARGET_T = 4
TOTAL_STEPS = sum(RUNG_STEPS) + TARGET_STEPS

#: Phase 1 runs the inherited rate; phase 2 the alternative, conditionally.
PHASE1_LR = 1e-3
PHASE2_LR = 3e-4

BATCH07_CONFIG = {
    "arms": [list(a) for a in ARMS],
    "curriculum": True,
    "curriculum_stages": CURRICULUM_STAGES,
    "rung_steps": RUNG_STEPS,
    "target_steps": TARGET_STEPS,
    "target_t": TARGET_T,
    "steps": TOTAL_STEPS,
    "seeds": 3,
    "width": "medium",
    "fix": "per_step",
    "per_step_norm_elementwise_affine": False,
    "fixed_set": None,
    "phase1_learning_rate": PHASE1_LR,
    "phase2_learning_rate": PHASE2_LR,
    "weight_decay": 0.01,
    "batch_size": 128,
    "optimizer": "adamw",
}

#: Where arm A's reused cells come from, and the value the reproduction cell must
#: land on for the environment to count as undrifted.
BATCH06_RUN_DIR = "phase2/batch06/t4_long"
BATCH06_CONSOLE_LOG = "runs/phase2/batch06_t4_long.console.log"
ARM_A_REPRODUCTION_SEED = 0
ARM_A_REPRODUCTION_EXACT_MATCH = 98.7793
ARM_A_REPRODUCTION_TOLERANCE_POINTS = 0.01

# ---------------------------------------------------------------------------
# The corrected plateau test (prereg §5)
# ---------------------------------------------------------------------------

#: Practical significance: a drift below this is not progress whatever its
#: t-statistic. Batch 06's original threshold, retained as one conjunct.
PLATEAU_SLOPE_THRESHOLD = 0.1

#: Statistical significance: the slope must exceed twice its standard error.
#: Batch 06 §3 measured per-evaluation noise of 0.36-0.61 points near the
#: ceiling, against which an 11-point fit cannot resolve the 0.25 points the
#: magnitude threshold alone asks of it.
PLATEAU_T_THRESHOLD = 2.0

#: Window over which the slope is fitted, in steps.
PLATEAU_WINDOW_STEPS = 2500


def slope_with_se(steps: Sequence[float], values: Sequence[float]) -> Optional[Dict]:
    """Least-squares slope in points per 1,000 steps, with its standard error."""
    xs, ys = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    if len(xs) < 3 or np.ptp(xs) == 0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    residual = ys - (slope * xs + intercept)
    dof = len(xs) - 2
    variance = float((residual ** 2).sum()) / dof
    sxx = float(((xs - xs.mean()) ** 2).sum())
    se = float(np.sqrt(variance / sxx)) if sxx > 0 else None
    if se is None:
        return None
    return {"slope_points_per_1000_steps": float(slope) * 1000.0,
            "standard_error_per_1000_steps": se * 1000.0,
            "t_statistic": (float(slope) / se) if se > 0 else None,
            "num_points": int(len(xs))}


def final_slope(curve: Sequence[Dict], window: int = PLATEAU_WINDOW_STEPS) -> Optional[Dict]:
    """Slope statistics over the last ``window`` steps of a recorded curve."""
    points = [(r["step"], r["val_token_accuracy"]) for r in curve
              if r.get("val_token_accuracy") is not None]
    if not points:
        return None
    cutoff = points[-1][0] - window
    tail = [p for p in points if p[0] >= cutoff]
    return slope_with_se([p[0] for p in tail], [p[1] for p in tail])


def has_plateaued(stats: Optional[Dict]) -> Optional[bool]:
    """Whether a cell has plateaued, by the corrected test.

    Still rising requires BOTH conjuncts: a slope at or above the magnitude floor
    AND a t-statistic above the threshold. Either alone misclassifies — the
    magnitude alone called Batch 06's ceiling noise a trend, and the t-test alone
    would call a 0.036 points/1k drift at chance a trend.

    None when the slope could not be fitted: an absent measurement cannot
    establish convergence.
    """
    if stats is None:
        return None
    slope = stats.get("slope_points_per_1000_steps")
    t = stats.get("t_statistic")
    if slope is None or t is None:
        return None
    rising = slope >= PLATEAU_SLOPE_THRESHOLD and t > PLATEAU_T_THRESHOLD
    return not rising


# ---------------------------------------------------------------------------
# The phase-2 trigger
# ---------------------------------------------------------------------------

A_AHEAD_OF_BOTH = "A_AHEAD_OF_BOTH"
DEPTH_NOT_COMPUTE = "DEPTH_NOT_COMPUTE"
COMPUTE_NOT_DEPTH = "COMPUTE_NOT_DEPTH"
DEPTH_NEEDS_UNSHARED_PARAMETERS = "DEPTH_NEEDS_UNSHARED_PARAMETERS"
WIDTH_BEATS_DEPTH = "WIDTH_BEATS_DEPTH"
UNRESOLVED = "UNRESOLVED_ESCALATE"
ARM_A_REPRODUCTION_FAILED = "ARM_A_FAILED_TO_REPRODUCE_BATCH06"

#: Readings in which arm A beats both alternatives, and so in which the inherited
#: learning rate is the confound that would undermine the conclusion. Phase 2 runs
#: if and only if the phase-1 reading is one of these.
PHASE2_TRIGGER_READINGS = (A_AHEAD_OF_BOTH, DEPTH_NOT_COMPUTE)


def compare(best_exact_match: Dict[str, Optional[float]], excluded: Sequence[str] = ()) -> Dict:
    """Apply PHASE2_BATCH_07_PREREG.md §6 mechanically."""
    usable = {k: v for k, v in best_exact_match.items() if k not in excluded and v is not None}
    a, b, c = usable.get("A"), usable.get("B"), usable.get("C")

    if a is None:
        return {"reading": ARM_A_REPRODUCTION_FAILED, "excluded": list(excluded),
                "reason": "arm A has no usable result, so there is no reference",
                "supports_depth_claim": False}

    low, high = UNRESOLVED_BAND
    if usable and all(low <= v <= high for v in usable.values()):
        return {"reading": UNRESOLVED, "excluded": list(excluded),
                "reason": (f"every usable arm falls in the {low}-{high} % band, where a "
                           f"{DIFFERENCE_THRESHOLD_POINTS}-point threshold would adjudicate a difference "
                           "three seeds cannot support"),
                "supports_depth_claim": False}

    def differs(x, y):
        return None if x is None or y is None else abs(x - y) > DIFFERENCE_THRESHOLD_POINTS

    a_b, a_c = differs(a, b), differs(a, c)

    # Negative readings first, so a positive verdict cannot be reached while an
    # alternative arm is actually ahead.
    if c is not None and a_c and c > a:
        return {"reading": DEPTH_NEEDS_UNSHARED_PARAMETERS, "excluded": list(excluded),
                "reason": f"arm C ({c:.4f} %) exceeds arm A ({a:.4f} %) by more than {DIFFERENCE_THRESHOLD_POINTS} points",
                "supports_depth_claim": False}
    if b is not None and a_b and b > a:
        return {"reading": WIDTH_BEATS_DEPTH, "excluded": list(excluded),
                "reason": f"arm B ({b:.4f} %) exceeds arm A ({a:.4f} %) by more than {DIFFERENCE_THRESHOLD_POINTS} points",
                "supports_depth_claim": False}
    # The row Batch 05 lacked: A ahead of BOTH.
    if b is not None and c is not None and a_b and a_c and a > b and a > c:
        return {"reading": A_AHEAD_OF_BOTH, "excluded": list(excluded),
                "reason": (f"arm A ({a:.4f} %) exceeds both arm B ({b:.4f} %) and arm C ({c:.4f} %) by more "
                           f"than {DIFFERENCE_THRESHOLD_POINTS} points: depth, and weight sharing helps"),
                "supports_depth_claim": True}
    if b is not None and a_b and a > b and (c is None or a_c is False):
        return {"reading": DEPTH_NOT_COMPUTE, "excluded": list(excluded),
                "reason": (f"arm A ({a:.4f} %) exceeds arm B ({b:.4f} %) by more than "
                           f"{DIFFERENCE_THRESHOLD_POINTS} points"
                           + ("" if c is None else f", and matches arm C ({c:.4f} %)")),
                "supports_depth_claim": True}
    if b is not None and a_b is False and (c is None or a_c is False):
        return {"reading": COMPUTE_NOT_DEPTH, "excluded": list(excluded),
                "reason": f"all usable arms agree within {DIFFERENCE_THRESHOLD_POINTS} points",
                "supports_depth_claim": False}
    return {"reading": UNRESOLVED, "excluded": list(excluded),
            "reason": "the arm pattern matches no pre-registered row",
            "supports_depth_claim": False}


def phase2_triggered(phase1_reading: Optional[str], all_plateaued: Optional[bool]) -> Dict:
    """Whether the conditional learning-rate follow-up runs (prereg §7).

    It runs only where arm A beat both alternatives, because that is the reading
    the inherited rate would undermine. It does not run on a reading that already
    goes against arm A — there is nothing for a rate check to rescue.

    A phase-1 comparison that was withheld for non-convergence does not trigger:
    spending 12.6 GPU-h on a rate check requires knowing what it is checking.
    """
    if phase1_reading in PHASE2_TRIGGER_READINGS:
        if all_plateaued is False:
            return {"triggered": False, "reason": (
                f"phase-1 reading {phase1_reading} would trigger, but not all arms had plateaued, so the "
                "comparison it rests on is withheld")}
        return {"triggered": True, "reason": (
            f"phase-1 reading {phase1_reading}: arm A ahead, so the inherited learning rate is the "
            "confound that would undermine the conclusion")}
    return {"triggered": False, "reason": (
        f"phase-1 reading {phase1_reading} does not place arm A ahead of both; a rate check has nothing "
        "to rescue")}


def phase2_reading(phase1_best: Dict[str, Optional[float]],
                   phase2_best: Dict[str, Optional[float]]) -> Dict:
    """How the follow-up is read (prereg §7.1).

    Keyed on whether 3e-4 changes either alternative's best exact match enough to
    overturn phase 1's ordering.
    """
    a = phase1_best.get("A")
    improved = {}
    for arm in ("B", "C"):
        before, after = phase1_best.get(arm), phase2_best.get(arm)
        if before is None or after is None:
            improved[arm] = None
        else:
            improved[arm] = after - before

    if a is None:
        return {"reading": "NO_REFERENCE", "improvement": improved,
                "overturns_phase1": False, "reason": "arm A has no usable result"}

    overturned = [arm for arm in ("B", "C")
                  if phase2_best.get(arm) is not None
                  and phase2_best[arm] > a - DIFFERENCE_THRESHOLD_POINTS]
    if overturned:
        return {"reading": "RATE_CONFOUNDED", "improvement": improved, "overturns_phase1": True,
                "reason": (f"at 3e-4, arm(s) {overturned} come within {DIFFERENCE_THRESHOLD_POINTS} points of "
                           "arm A, so phase 1's ordering was a learning-rate artefact. A proper rate search, "
                           "not this two-point check, is the prerequisite for any arm comparison")}
    material = [arm for arm, delta in improved.items()
                if delta is not None and delta > DIFFERENCE_THRESHOLD_POINTS]
    if material:
        return {"reading": "RATE_MATTERS_BUT_ORDERING_HOLDS", "improvement": improved,
                "overturns_phase1": False,
                "reason": (f"3e-4 improves arm(s) {material} by more than {DIFFERENCE_THRESHOLD_POINTS} points "
                           "without reaching arm A: the rate matters and the ordering survives it")}
    return {"reading": "RATE_DOES_NOT_EXPLAIN_THE_GAP", "improvement": improved,
            "overturns_phase1": False,
            "reason": ("3e-4 changes neither alternative materially, so phase 1's ordering is not a "
                       "learning-rate artefact at this second point")}
