"""Batch 03 — T = 3 across K with the residual fix on (PHASE2_BATCH_03_PREREG.md).

Batch 02 found T = 3 learnable at K = 1 and not at K = 8, but ran `fix=none`, so
the residual-stream confound was uncontrolled on exactly the axis that comparison
varies — 17.70× hidden-state growth at K = 8 against ~1× at K = 1. Its K
conclusion was withdrawn for that reason.

This batch is Batch 02's configuration with one change, Addendum 02's non-affine
per-step normalisation on for every K, and the K range widened to {1, 2, 4, 8} so
a trend can be read rather than a two-point difference.

What is new here beyond Batch 02 is the trend reading, which is fixed in
`trend_reading` below rather than decided when the numbers arrive, and the
per-cell post-training bound check: Addendum 01 §A1.3 requires the bound to hold
after training, and a cell whose bound broke during training is not a fix-on cell
and must not carry a confound-related conclusion.

The test split is never read by anything in this module.
"""

from typing import Dict, List, Optional, Sequence

from phase2.batch02 import PROBE_CONFIG as BATCH02_CONFIG
from phase2.probes import LEARNABLE, NOT_LEARNABLE, PARTIAL
from phase2.reclassify import DEGENERATE

#: Batch 02's configuration with the fix on and K widened. Everything else is
#: copied from it rather than restated, so the two cannot drift apart silently;
#: `tests/phase2/test_batch03.py` asserts the copy is faithful field by field.
BATCH03_CONFIG = {
    **BATCH02_CONFIG,
    "fix": "per_step",
    "per_step_norm_elementwise_affine": False,
    "k_values": [1, 2, 4, 8],
}

#: The fields Batch 03 must hold identical to Batch 02. The fix and the K range
#: are the intended differences; anything else differing is a bug.
INHERITED_FIELDS = (
    "curriculum", "curriculum_stages", "steps_per_stage", "target_t",
    "target_steps", "steps", "seeds", "width", "fixed_set",
)

#: Arm 2c-i's limit, reused unchanged: per-step RMS growth must stay below this
#: at every K, both at initialisation (the gate) and after training (per cell).
GROWTH_LIMIT = 1.05


def inherited_matches_batch02() -> bool:
    """Whether every field Batch 03 is supposed to inherit actually matches."""
    return all(BATCH03_CONFIG[f] == BATCH02_CONFIG[f] for f in INHERITED_FIELDS)


def resolved_batch03_config() -> Dict:
    return dict(BATCH03_CONFIG)


def bound_held(growth_ratio: Optional[float], limit: float = GROWTH_LIMIT) -> bool:
    """Whether a cell's post-training growth ratio is within the bound.

    A missing ratio counts as not held: the point of the check is to refuse a
    confound-related conclusion unless the bound is shown, and an absent
    measurement shows nothing.
    """
    return growth_ratio is not None and growth_ratio < limit


# ---------------------------------------------------------------------------
# The trend reading, fixed before any cell runs
# ---------------------------------------------------------------------------

K_INDEPENDENT = "K_INDEPENDENT"
MONOTONE_DECLINE = "MONOTONE_DECLINE"
NON_MONOTONE = "NON_MONOTONE"
BATCH_FAILURE = "BATCH_FAILURE"

#: The K=1 seed spread Batch 02 measured (49.9914 % to 81.2127 %). A difference
#: in mean accuracy smaller than this is not a trend, because three seeds at one
#: K already vary by this much.
BATCH02_K1_SEED_SPREAD_POINTS = 81.2127 - 49.9914


def trend_reading(learnable_k: Sequence[int], k_values: Sequence[int] = (1, 2, 4, 8)) -> Dict:
    """Read the learnability pattern across K, per PHASE2_BATCH_03_PREREG.md §7.

    Keyed on *learnability* and never on mean accuracy, because three seeds
    cannot establish monotonicity of an accuracy that varies by 31 points across
    seeds at a single K. Returns the reading and the threshold bracket when the
    pattern is a prefix; the threshold is not localised further than that.
    """
    ordered = sorted(k_values)
    learnable = sorted(set(learnable_k))

    if not learnable:
        # Batch 02's K=1 was LEARNABLE with the fix OFF, so K=1 failing with it
        # on is a fault in this batch rather than a finding about K.
        return {"reading": BATCH_FAILURE, "learnable_k": [],
                "reason": "K=1 was LEARNABLE in Batch 02 with the fix off; no K learnable here is a batch failure",
                "threshold_between": None}

    prefix = ordered[:len(learnable)]
    if learnable == ordered:
        return {"reading": K_INDEPENDENT, "learnable_k": learnable,
                "reason": "every K learnable; any variation in mean accuracy is quantitative, not a boundary",
                "threshold_between": None}
    if learnable == prefix:
        largest = learnable[-1]
        smallest_not = next(k for k in ordered if k not in learnable)
        return {"reading": MONOTONE_DECLINE, "learnable_k": learnable,
                "reason": f"learnable up to K={largest} and not at K={smallest_not}",
                "threshold_between": [largest, smallest_not]}
    return {"reading": NON_MONOTONE, "learnable_k": learnable,
            "reason": "learnable set is not a prefix of the K range; no threshold claim is available",
            "threshold_between": None,
            "seed_spread_reference_points": BATCH02_K1_SEED_SPREAD_POINTS}


def accuracy_gap_is_meaningful(gap_points: Optional[float]) -> bool:
    """Whether an accuracy difference across K exceeds Batch 02's seed spread.

    Below that, three seeds at one K already vary by more than the difference
    being claimed, so the difference is not evidence of anything.
    """
    return gap_points is not None and gap_points > BATCH02_K1_SEED_SPREAD_POINTS


# ---------------------------------------------------------------------------
# The §8 outcome table
# ---------------------------------------------------------------------------

CONFOUND_ATTRIBUTED = "BATCH02_K8_FAILURE_ATTRIBUTED_TO_CONFOUND"
NEGATIVE_K_STANDS = "NEGATIVE_K_RESULT_STANDS_WITH_CONFOUND_EXCLUDED"
NOT_A_COMPOSITION_RESULT = "K8_LEARNABLE_BUT_NOT_COMPOSITION"
FIX_NOT_ESTABLISHED = "FIX_NOT_ESTABLISHED_AT_K8"
BATCH_FAILED = "BATCH_FAILURE"


def outcome(k8_classification: Optional[str], k8_attribution: Optional[str],
            k8_bound_held: bool, k1_classification: Optional[str]) -> Dict:
    """Apply PHASE2_BATCH_03_PREREG.md §8 mechanically.

    Order matters and is pre-registered: the bound check gates everything,
    because a cell whose stream was not bounded is not a fix-on cell and cannot
    speak to the confound either way.
    """
    if not k8_bound_held:
        return {"outcome": FIX_NOT_ESTABLISHED,
                "reason": "K=8's post-training growth ratio is absent or above the limit, so it is not a fix-on cell",
                "answers_the_confound_question": False}
    if k1_classification not in (LEARNABLE,):
        return {"outcome": BATCH_FAILED,
                "reason": "K=1 was LEARNABLE in Batch 02 with the fix off; not reproducing that is a batch failure",
                "answers_the_confound_question": False}
    if k8_classification == LEARNABLE:
        if k8_attribution == "COMPOSITION":
            return {"outcome": CONFOUND_ATTRIBUTED,
                    "reason": "K=8 learns T=3 once the stream is bounded, so the Batch 02 failure is the confound",
                    "answers_the_confound_question": True}
        return {"outcome": NOT_A_COMPOSITION_RESULT,
                "reason": f"K=8 clears the bar but its attribution is {k8_attribution}, not composition",
                "answers_the_confound_question": False}
    if k8_classification in (NOT_LEARNABLE, DEGENERATE, PARTIAL):
        return {"outcome": NEGATIVE_K_STANDS,
                "reason": "K=8 still does not learn T=3 with the stream bounded, so the confound is excluded",
                "answers_the_confound_question": True}
    return {"outcome": BATCH_FAILED, "reason": f"unrecognised K=8 classification {k8_classification!r}",
            "answers_the_confound_question": False}
