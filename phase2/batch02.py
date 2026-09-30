"""Batch 02 — the T = 3 learnability probe (PHASE2_BATCH_02_PREREG.md).

Batch 01 found that a curriculum solves T = 1 and T = 2 and then sits at chance
from the first T = 4 stage onward. T = 3 is the untested point inside that gap,
so this probe runs Arm 3's curriculum with the ladder's top moved from 4 to 3
and nothing else changed.

Two things here are not in Batch 01.

**Stage-boundary diagnostics.** Arm 3 was specified to record validation
accuracy at each depth *at the end of each stage* and instead measured the final
model once, so its stage-by-stage evidence survived only as console output. This
module supplies the measurement as a hook the training loop calls at each
boundary, and it lands in the report as a field.

**The shortcut check.** G2 fails at T = 3: the shallow candidate
``window_product_forward``, s[i]·s[i+1]·s[i+2]·s[i+3], agrees with F³ on
26.67 % of tokens. The LEARNABLE bar is chance + 5 points = 6.67 %, so a model
that learned only that shortcut would clear it. Accuracy alone therefore cannot
distinguish depth-3 composition from the shortcut, and the agreement between a
model's predictions and the shortcut's is recorded per cell so that it can.

The test split is never read by anything in this module.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from phase2.automaton import iterate
from phase2.groups import FiniteGroup

# ---------------------------------------------------------------------------
# The shallow shortcuts G2 found, written out per depth
# ---------------------------------------------------------------------------

#: The candidate that fails G2 at each depth, named as `phase2.gates` names it.
#: T = 1 has no applicable candidate (it is the primitive), and T = 4 and T = 8
#: pass G2 with no candidate above 5 %.
SHORTCUT_CANDIDATE = {
    2: "s[i+0]*s[i+2]",
    3: "window_product_forward",
}

#: Exact token agreement between the candidate and F^T, over uniformly random
#: states of the A5 ring. Both equal 16/60: the shortcut is right exactly when a
#: local product squares to the identity, and 16 of A5's 60 elements do.
#:
#: T = 2: F²[i] = s_i s_{i+1}² s_{i+2} equals s_i s_{i+2} iff s_{i+1}² = e.
#: T = 3: F³[i] = s_i s_{i+1}² s_{i+2} s_{i+1} s_{i+2}² s_{i+3} equals
#:        s_i s_{i+1} s_{i+2} s_{i+3} iff (s_{i+1} s_{i+2})² = e.
#:
#: Both identities were checked against `phase2.automaton.iterate` element by
#: element, not derived on paper alone; `tests/phase2/test_batch02.py` re-checks
#: them.
SHORTCUT_TOKEN_AGREEMENT_A5 = {2: 16 / 60, 3: 16 / 60}


def shortcut_prediction(group: FiniteGroup, inputs: np.ndarray, t: int) -> np.ndarray:
    """What the G2-failing shallow candidate for depth ``t`` predicts.

    Written out directly from the candidate's definition rather than imported
    from the gate, so a change to the gate's candidate set cannot silently
    change what this diagnostic compares against.
    """
    if t not in SHORTCUT_CANDIDATE:
        raise ValueError(f"no G2-failing candidate recorded at T={t}; known depths {sorted(SHORTCUT_CANDIDATE)}")
    table = group.table
    s = iterate(group, inputs, 0)  # validated copy

    def cell(a: int) -> np.ndarray:
        return np.roll(s, -a, axis=-1)

    if t == 2:
        return table[cell(0), cell(2)]
    product = cell(0)
    for j in range(1, t + 1):
        product = table[product, cell(j)]
    return product


def shortcut_agreement_percent(predictions: np.ndarray, group: FiniteGroup, inputs: np.ndarray, t: int) -> float:
    """Percentage of tokens where the model and the shortcut emit the same thing.

    This is the discriminating quantity, and it is not the accuracy. A model
    that computes F^T exactly agrees with the shortcut on 26.67 % of tokens,
    because that is where the shortcut is correct. A model that computes the
    shortcut agrees with it on 100 %. The two hypotheses are therefore 73 points
    apart on this measure while being indistinguishable on accuracy alone.
    """
    return float((np.asarray(predictions) == shortcut_prediction(group, inputs, t)).mean() * 100.0)


# ---------------------------------------------------------------------------
# Pre-registered reading of a LEARNABLE verdict at T = 3
# ---------------------------------------------------------------------------

#: At or above this agreement with the shortcut, a cell's accuracy is
#: attributed to the shortcut rather than to depth-3 composition.
SHORTCUT_ATTRIBUTABLE_AGREEMENT_PERCENT = 90.0

#: At or below this, it is not. Between the two the cell is reported as
#: UNRESOLVED and escalated rather than branched on automatically: the gap
#: between the two hypotheses is 73 points wide, so a value inside it is not
#: something a pre-registered threshold should be asked to settle.
SHORTCUT_EXCLUDED_AGREEMENT_PERCENT = 50.0

SHORTCUT = "SHORTCUT"
COMPOSITION = "COMPOSITION"
UNRESOLVED = "UNRESOLVED"


def shortcut_attribution(agreement_percent: Optional[float]) -> str:
    """SHORTCUT / COMPOSITION / UNRESOLVED from one cell's shortcut agreement."""
    if agreement_percent is None:
        return UNRESOLVED
    if agreement_percent >= SHORTCUT_ATTRIBUTABLE_AGREEMENT_PERCENT:
        return SHORTCUT
    if agreement_percent <= SHORTCUT_EXCLUDED_AGREEMENT_PERCENT:
        return COMPOSITION
    return UNRESOLVED


def attribute_classification(amended: Dict, agreement_by_seed: Dict[int, Optional[float]]) -> Dict:
    """Split one arm-and-K verdict by what its above-bar seeds are doing.

    Applies only to the seeds the amended rule already counted; it can never
    promote a seed. A verdict with no seed above the bar is unaffected, which is
    why this is a refinement of the classification and not a change to it.
    """
    counted = set(amended.get("seeds_meeting_learnable_bar_amended", []))
    counted |= set(amended.get("seeds_above_chance_amended", []))
    per_seed = []
    for row in amended.get("per_seed", []):
        seed = row.get("seed")
        agreement = agreement_by_seed.get(seed)
        per_seed.append({
            "seed": seed,
            "shortcut_agreement_percent": agreement,
            "counted_by_amended_rule": seed in counted,
            "attribution": shortcut_attribution(agreement) if seed in counted else None,
        })
    attributions = [r["attribution"] for r in per_seed if r["counted_by_amended_rule"]]
    if not attributions:
        overall = None
    elif all(a == SHORTCUT for a in attributions):
        overall = SHORTCUT
    elif any(a == COMPOSITION for a in attributions):
        overall = COMPOSITION
    else:
        overall = UNRESOLVED
    return {
        "attribution": overall,
        "thresholds": {
            "shortcut_at_or_above_percent": SHORTCUT_ATTRIBUTABLE_AGREEMENT_PERCENT,
            "composition_at_or_below_percent": SHORTCUT_EXCLUDED_AGREEMENT_PERCENT,
        },
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Probe configuration, kept as data so isolation from Batch 01 is assertable
# ---------------------------------------------------------------------------

#: What the probe resolves to. Every value except ``curriculum_stages`` and
#: ``target_t`` is Arm 3's, so the probe is Arm 3 with the ladder's top moved
#: from 4 to 3. `tests/phase2/test_batch02.py` compares this against Arm 3's
#: committed configuration and fails if anything else drifted.
PROBE_CONFIG = {
    "fix": "none",
    "curriculum": True,
    "curriculum_stages": [1, 2, 3],
    "steps_per_stage": 5000,
    "target_t": 3,
    "target_steps": 20000,
    "steps": 35000,
    "seeds": 3,
    "k_values": [1, 8],
    "width": "medium",
    "fixed_set": None,
}

#: Depths measured at every stage boundary. T = 3 is the target; T = 1 and T = 2
#: are there because Arm 3's finding was that the target stage *erases* them,
#: and that claim needs a measurement at each boundary rather than one at the end.
DIAGNOSTIC_T_VALUES = (1, 2, 3)


def resolved_probe_config() -> Dict:
    return dict(PROBE_CONFIG)


def stage_boundary_steps(stages: Sequence[int], steps_per_stage: int, target_steps: int) -> List[int]:
    """Cumulative step at which each stage ends, matching CurriculumStream.

    The target depth appears twice — once as the ladder's last rung and once as
    the target stage — which is how Arm 3's schedule was built, so the boundary
    list has one more entry than there are ladder stages.
    """
    lengths = [steps_per_stage] * len(list(stages)) + [target_steps]
    out, cumulative = [], 0
    for length in lengths:
        cumulative += length
        out.append(cumulative)
    return out
