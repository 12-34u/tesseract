"""Batch 04 — T = 4 with the residual fix and the full curriculum
(PHASE2_BATCH_04_PREREG.md).

Batch 03 solved T = 3 exactly at K >= 4 with the stream bounded, and showed that
Batch 02's K = 8 failure was the residual confound. The fix has never been run at
T = 4, and Batch 01 excluded the confound there on the strength of arms that
never combined the fix with the curriculum — the exact combination that changed
the T = 3 result.

Two things differ from Batch 03 beyond the depth.

**T = 4 passes G2.** There is no shallow candidate to attribute a T = 4 result
to, so no attribution is computed at the target depth. The ladder still passes
through T = 2 and T = 3, which both fail G2 at 26.67 %, so shortcut agreement is
recorded at those depths in the stage-boundary diagnostics.

**Exact match is a reported co-primary metric.** At T = 3, K = 1 cleared the
token-accuracy bar at 43-65 % while producing 0.0000 % exact match in two of
three cells: the target is a 17-position ring and token accuracy averages over
positions, so many independent local errors still score well. The classification
rule is deliberately unchanged — it predates every batch and changing it now
would break comparability — so exact match constrains the interpretation rather
than the verdict, through `exact_match_reading` below.

The test split is never read by anything in this module.
"""

from typing import Dict, Optional, Sequence

from phase2.batch02 import PROBE_CONFIG as BATCH02_CONFIG
from phase2.batch02 import SHORTCUT_CANDIDATE
from phase2.batch03 import BATCH03_CONFIG, GROWTH_LIMIT, bound_held  # noqa: F401  (re-exported)
from phase2.probes import LEARNABLE, NOT_LEARNABLE, PARTIAL
from phase2.reclassify import DEGENERATE

#: Batch 03's configuration with the depth moved to 4, the ladder extended, and
#: K = 2 dropped. Copied from Batch 03 rather than restated so the two cannot
#: drift; `tests/phase2/test_batch04.py` checks the copy field by field.
BATCH04_CONFIG = {
    **BATCH03_CONFIG,
    "curriculum_stages": [1, 2, 3, 4],
    "target_t": 4,
    "steps": 40000,
    "k_values": [1, 4, 8],
}

#: Fields Batch 04 inherits from Batch 03 unchanged.
INHERITED_FIELDS = (
    "fix", "per_step_norm_elementwise_affine", "curriculum",
    "steps_per_stage", "target_steps", "seeds", "width", "fixed_set",
)

#: Depths measured at every stage boundary: every ladder rung plus the target.
DIAGNOSTIC_T_VALUES = (1, 2, 3, 4)

#: Depths where G2 found a shallow candidate, so agreement is worth recording.
#: T = 4 is absent because it passes G2 — there is nothing to attribute to.
SHORTCUT_T_VALUES = tuple(t for t in DIAGNOSTIC_T_VALUES if t in SHORTCUT_CANDIDATE)

#: Below this exact match, a LEARNABLE verdict is reported as resting on token
#: accuracy alone and counts as negative on the depth question. Fixed in
#: PHASE2_BATCH_04_PREREG.md §4 before the batch ran. 1 % of the 2,048-example
#: validation split is 20 correct rings, comfortably above the 3 that K = 1
#: managed at T = 3.
EXACT_MATCH_FLOOR_PERCENT = 1.0

TOKEN_ACCURACY_ONLY = "LEARNABLE_ON_TOKEN_ACCURACY_ONLY"
LEARNABLE_WITH_EXACT_MATCH = "LEARNABLE_WITH_EXACT_MATCH"


def inherited_matches_batch03() -> bool:
    """Whether every field Batch 04 should inherit from Batch 03 actually matches."""
    return all(BATCH04_CONFIG[f] == BATCH03_CONFIG[f] for f in INHERITED_FIELDS)


def resolved_batch04_config() -> Dict:
    return dict(BATCH04_CONFIG)


def exact_match_reading(classification: Optional[str], best_exact_match: Optional[float]) -> Dict:
    """How a verdict is read alongside its exact match (prereg §4).

    Never changes the classification — it cannot promote or demote one. It
    records whether a LEARNABLE verdict is backed by correct rings or rests on
    token accuracy alone, which at T = 3 was the difference between K = 4
    solving the task and K = 1 not learning it.
    """
    if classification != LEARNABLE:
        return {"reading": None, "exact_match_percent": best_exact_match,
                "floor_percent": EXACT_MATCH_FLOOR_PERCENT,
                "supports_depth_claim": False,
                "reason": f"classification is {classification}, not LEARNABLE"}
    if best_exact_match is None:
        return {"reading": None, "exact_match_percent": None,
                "floor_percent": EXACT_MATCH_FLOOR_PERCENT,
                "supports_depth_claim": False,
                "reason": "exact match not recorded; an absent measurement supports nothing"}
    if best_exact_match < EXACT_MATCH_FLOOR_PERCENT:
        return {"reading": TOKEN_ACCURACY_ONLY, "exact_match_percent": best_exact_match,
                "floor_percent": EXACT_MATCH_FLOOR_PERCENT,
                "supports_depth_claim": False,
                "reason": (f"LEARNABLE on token accuracy with {best_exact_match:.4f} % exact match, "
                           f"below the {EXACT_MATCH_FLOOR_PERCENT} % floor: no correct rings to speak of")}
    return {"reading": LEARNABLE_WITH_EXACT_MATCH, "exact_match_percent": best_exact_match,
            "floor_percent": EXACT_MATCH_FLOOR_PERCENT,
            "supports_depth_claim": True,
            "reason": f"LEARNABLE with {best_exact_match:.4f} % exact match, at or above the floor"}


# ---------------------------------------------------------------------------
# The §8 outcome table
# ---------------------------------------------------------------------------

GRID_RESUME_PROPOSED = "T4_LEARNABLE_PROPOSE_GRID_RESUME_DRAFT"
TOKEN_ONLY_NOT_A_DEPTH_RESULT = "T4_LEARNABLE_ON_TOKEN_ACCURACY_ONLY"
T3_IS_THE_BOUNDARY = "T3_IS_THE_DEPTH_BOUNDARY_WITH_THE_FIX_ON"
FIX_NOT_ESTABLISHED = "FIX_NOT_ESTABLISHED_AT_SOME_K"
NEEDS_INVESTIGATION = "ORDERING_CONTRADICTS_BATCH03_INVESTIGATE"


def outcome(classification_by_k: Dict[int, Optional[str]],
            exact_match_by_k: Dict[int, Optional[float]],
            bound_held_by_k: Dict[int, bool],
            k_values: Sequence[int] = (1, 4, 8)) -> Dict:
    """Apply PHASE2_BATCH_04_PREREG.md §8 mechanically.

    Order is pre-registered. The bound check gates everything, because a cell
    whose stream was not bounded cannot speak to the confound either way.
    """
    unbounded = [k for k in k_values if not bound_held_by_k.get(k, False)]
    if unbounded:
        return {"outcome": FIX_NOT_ESTABLISHED,
                "unbounded_k": unbounded,
                "reason": f"post-training growth absent or above the limit at K={unbounded}",
                "supports_depth_claim": False}

    learnable = [k for k in k_values if classification_by_k.get(k) == LEARNABLE]
    backed = [k for k in learnable
              if (exact_match_by_k.get(k) or 0.0) >= EXACT_MATCH_FLOOR_PERCENT]

    # Batch 03 established K=1 < K>=4 on the strict metric at T=3. K=1 learnable
    # while no higher K is would invert that and is not reported as a finding.
    higher = [k for k in k_values if k > 1]
    if 1 in backed and not any(k in backed for k in higher):
        return {"outcome": NEEDS_INVESTIGATION, "learnable_k": learnable, "backed_by_exact_match_k": backed,
                "reason": "K=1 backed by exact match while no higher K is, which inverts Batch 03's ordering",
                "supports_depth_claim": False}

    if backed:
        return {"outcome": GRID_RESUME_PROPOSED, "learnable_k": learnable, "backed_by_exact_match_k": backed,
                "reason": ("T=4 is learnable with correct rings at K="
                           f"{backed}; Batch 01 §9's exclusion of the residual confound at T=4 is withdrawn"),
                "supports_depth_claim": True}
    if learnable:
        return {"outcome": TOKEN_ONLY_NOT_A_DEPTH_RESULT, "learnable_k": learnable, "backed_by_exact_match_k": [],
                "reason": (f"LEARNABLE at K={learnable} but exact match below "
                           f"{EXACT_MATCH_FLOOR_PERCENT} % everywhere: no correct rings"),
                "supports_depth_claim": False}
    return {"outcome": T3_IS_THE_BOUNDARY, "learnable_k": [], "backed_by_exact_match_k": [],
            "reason": "no K learns T=4 with the stream bounded, so T=3 is the depth boundary with the fix on",
            "supports_depth_claim": False}
