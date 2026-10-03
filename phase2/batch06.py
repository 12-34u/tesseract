"""Batch 06 — T = 4 with Batch 04's two limits removed (PHASE2_BATCH_06_PREREG.md).

Batch 04 found T = 4 learnable at K = 4 and recorded two limits that bound every
number in it: the 5,000-step T = 3 rung gave K = 1 only 2.14 % of T = 3 while
K = 4 got 15-33 %, and every cell that took off was still rising at step 40,000.

This batch lengthens both, with each length taken from measurement rather than
chosen: the T = 3 rung goes to 15,000 because Batch 03's slowest K = 1 seed needed
12,000 steps to pass the T = 3 shortcut ceiling, and the target stage goes to
60,000 because Batch 04's steepest cell was still climbing at +5.136 points per
1,000 steps after 20,000.

The rungs are **not** tuned per K. T = 1 and T = 2 reach 100 % within 5,000 steps
at every K in every batch measured, so they stay at 5,000; T = 3 goes to what the
slowest arm needs and the faster arms get more than they need. Tuning per K would
make the K comparison meaningless, which is the error being corrected.

What Batch 05 added to the design: an outcome row for "not all K at convergence".
A fixed budget does not deliver convergence for every arm, and comparing a
plateaued arm against a still-rising one measures the budget as much as the arm.
"""

from typing import Dict, Optional, Sequence

from phase2.batch04 import BATCH04_CONFIG, EXACT_MATCH_FLOOR_PERCENT
from phase2.batch03 import GROWTH_LIMIT, bound_held  # noqa: F401  (re-exported)

#: Per-stage rung lengths. The T=3 rung is three times the others; §1.1 of the
#: pre-registration derives the 15,000 from Batch 03's K=1 takeoff data.
RUNG_STEPS = [5000, 5000, 15000, 5000]
TARGET_STEPS = 60000

BATCH06_CONFIG = {
    **BATCH04_CONFIG,
    "rung_steps": RUNG_STEPS,
    "target_steps": TARGET_STEPS,
    "steps": sum(RUNG_STEPS) + TARGET_STEPS,
}

#: Fields inherited from Batch 04 unchanged. The two stage lengths and the total
#: are the intended changes; `steps_per_stage` is superseded by `rung_steps`.
INHERITED_FIELDS = (
    "fix", "per_step_norm_elementwise_affine", "curriculum",
    "curriculum_stages", "target_t", "k_values", "seeds", "width", "fixed_set",
)

#: A cell has plateaued when its final slope is below this, in accuracy points
#: per 1,000 steps over the last 2,500 steps. Decision 2 removed adaptive
#: stopping, so nothing in the protocol forces a plateau and this is how one is
#: recognised after the fact. 0.1 points per 1,000 steps is 2.5 points over the
#: whole 25,000-step tail, well inside measurement noise on a 2,048-example split.
PLATEAU_SLOPE_THRESHOLD = 0.1

PLATEAU_WITH_EXACT_MATCH = "T4_PLATEAU_WITH_EXACT_MATCH_PROPOSE_GRID_RESUME_DRAFT"
LEARNABLE_STILL_RISING = "T4_LEARNABLE_BUT_STILL_RISING"
TOKEN_ONLY = "T4_LEARNABLE_ON_TOKEN_ACCURACY_ONLY"
NOT_LEARNABLE_ANYWHERE = "T4_NOT_LEARNABLE_AT_ANY_K"
BATCH_FAILURE = "BATCH_FAILURE_CONTRADICTS_BATCH04"
FIX_NOT_ESTABLISHED = "FIX_NOT_ESTABLISHED_AT_SOME_K"
MIXED_CONVERGENCE = "NOT_ALL_K_AT_CONVERGENCE"


def inherited_matches_batch04() -> bool:
    return all(BATCH06_CONFIG[f] == BATCH04_CONFIG[f] for f in INHERITED_FIELDS)


def resolved_batch06_config() -> Dict:
    return dict(BATCH06_CONFIG)


def has_plateaued(final_slope: Optional[float]) -> Optional[bool]:
    """Whether a cell's final slope is below the plateau threshold.

    None when the slope was not recorded: an absent measurement cannot establish
    that a cell converged, and treating it as converged is the unsafe direction.
    """
    if final_slope is None:
        return None
    return final_slope < PLATEAU_SLOPE_THRESHOLD


def outcome(classification_by_k: Dict[int, Optional[str]],
            exact_match_by_k: Dict[int, Optional[float]],
            plateaued_by_k: Dict[int, Optional[bool]],
            bound_held_by_k: Dict[int, bool],
            k_values: Sequence[int] = (1, 4, 8)) -> Dict:
    """Apply PHASE2_BATCH_06_PREREG.md §4 mechanically.

    Order is pre-registered. The bound gates everything; then the batch-failure
    check, because a result contradicting Batch 04 is not interpretable; then the
    convergence row, which can only withhold a comparison.
    """
    unbounded = [k for k in k_values if not bound_held_by_k.get(k, False)]
    if unbounded:
        return {"outcome": FIX_NOT_ESTABLISHED, "unbounded_k": unbounded,
                "reason": f"post-training growth absent or above the limit at K={unbounded}",
                "supports_depth_claim": False, "comparison_withheld": True}

    learnable = [k for k in k_values if classification_by_k.get(k) == "LEARNABLE"]
    backed = [k for k in learnable if (exact_match_by_k.get(k) or 0.0) >= EXACT_MATCH_FLOOR_PERCENT]

    if not learnable:
        # Batch 04 found K=4 LEARNABLE on a shorter budget, so nothing learnable
        # here is a fault in this batch rather than a finding about T=4.
        return {"outcome": BATCH_FAILURE, "learnable_k": [],
                "reason": "Batch 04 found K=4 LEARNABLE at 40,000 steps; nothing learnable at 90,000 is a fault",
                "supports_depth_claim": False, "comparison_withheld": True}

    # The row Batch 05 showed was missing. Only over the K values that learned
    # anything: a K at chance has no trajectory to converge.
    states = {k: plateaued_by_k.get(k) for k in learnable}
    if any(v is None for v in states.values()):
        mixed = True
    else:
        mixed = len(set(states.values())) > 1

    if not backed:
        return {"outcome": TOKEN_ONLY, "learnable_k": learnable, "backed_by_exact_match_k": [],
                "reason": (f"LEARNABLE at K={learnable} but exact match below "
                           f"{EXACT_MATCH_FLOOR_PERCENT} % everywhere: no correct rings"),
                "supports_depth_claim": False, "comparison_withheld": mixed}

    rising = [k for k in backed if plateaued_by_k.get(k) is False]
    plateaued = [k for k in backed if plateaued_by_k.get(k) is True]

    if plateaued:
        return {"outcome": PLATEAU_WITH_EXACT_MATCH, "learnable_k": learnable,
                "backed_by_exact_match_k": backed, "plateaued_k": plateaued, "still_rising_k": rising,
                "reason": (f"T=4 reaches a plateau with correct rings at K={plateaued}; the level is "
                           "reportable rather than budget-limited"),
                "supports_depth_claim": True, "comparison_withheld": mixed}
    return {"outcome": LEARNABLE_STILL_RISING, "learnable_k": learnable,
            "backed_by_exact_match_k": backed, "plateaued_k": [], "still_rising_k": rising,
            "reason": (f"exact match at or above the floor at K={backed} but every such cell is still "
                       "rising at 90,000 steps; the accuracy is not an achieved level"),
            "supports_depth_claim": False, "comparison_withheld": mixed}
