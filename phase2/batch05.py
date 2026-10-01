"""Batch 05 — depth or compute? (PHASE2_BATCH_05_PREREG.md)

Batch 03 found K = 4 solving T = 3 exactly while K = 1 did not, at matched
*steps*. At 35,000 steps K = 4 does four times K = 1's forward FLOPs, so that
result is consistent with two claims: recursion composes something width cannot,
or four times the compute is simply four times the compute.

This batch holds forward compute fixed and varies how it is spent across three
arms that match to within 0.008 %:

    A  tesseract K=4      837,436 params   108,068,864 FLOPs   4 block calls
    B  width_scaled       3,214,678        108,077,568         1 block call
    C  unrolled L=4       3,206,716        108,068,864         4 block calls

Arm A uses 3.84x fewer parameters than either alternative, which is the whole
point: if A matches or beats them, depth is being bought without parameters.

The comparison is keyed on **exact match**, because Batch 03's Note 01 showed
token accuracy admitting cells with no correct rings at all. The "differ"
threshold and its justification are in the pre-registration §5 and encoded in
`arms_differ` and `compare` below; both were fixed before any cell ran.

The test split is never read by anything in this module.
"""

from typing import Dict, List, Optional, Sequence

from phase2.batch02 import PROBE_CONFIG as BATCH02_CONFIG
from phase2.batch03 import GROWTH_LIMIT, bound_held  # noqa: F401  (re-exported)

#: One arm per (family, depth, label). Depth is the block-execution count for the
#: recursive and unrolled families, and the matching target for width_scaled.
ARMS = (
    ("A", "tesseract", 4, "shared recursion, K=4"),
    ("B", "width_scaled", 4, "width-scaled K=1, compute-matched to K=4"),
    ("C", "unrolled", 4, "4 unrolled blocks"),
)

#: Batch 03's curriculum and budget, inherited unchanged so arm A replicates
#: Batch 03's K=4 cells. Copied rather than restated; a test checks the copy.
BATCH05_CONFIG = {
    "fix": "per_step",
    "per_step_norm_elementwise_affine": False,
    "curriculum": BATCH02_CONFIG["curriculum"],
    "curriculum_stages": BATCH02_CONFIG["curriculum_stages"],
    "steps_per_stage": BATCH02_CONFIG["steps_per_stage"],
    "target_t": BATCH02_CONFIG["target_t"],
    "target_steps": BATCH02_CONFIG["target_steps"],
    "steps": BATCH02_CONFIG["steps"],
    "seeds": BATCH02_CONFIG["seeds"],
    "width": BATCH02_CONFIG["width"],
    "fixed_set": BATCH02_CONFIG["fixed_set"],
    "arms": [list(a) for a in ARMS],
}

#: Fields inherited from Batch 03's configuration (itself Batch 02's plus the fix).
INHERITED_FIELDS = (
    "curriculum", "curriculum_stages", "steps_per_stage", "target_t",
    "target_steps", "steps", "seeds", "width", "fixed_set",
)

#: Arms differ iff their best exact match differs by more than this. Argued in
#: PHASE2_BATCH_05_PREREG.md §5: among cells that took off, the observed
#: within-arm spread was 0.10 points (Batch 03 K=4: 100.00 / 99.90 / 100.00), so
#: 5 points is 50x that spread, while being small against the 100 %-vs-0.15 %
#: gap the batch is looking for.
DIFFERENCE_THRESHOLD_POINTS = 5.0

#: If every arm lands inside this band the threshold would be adjudicating a
#: difference three seeds cannot support, so the comparison escalates instead.
UNRESOLVED_BAND = (5.0, 95.0)

#: Arm A replicates Batch 03's K=4 cells (~100 % exact match). Below this it has
#: not replicated and the batch is suspect rather than informative.
ARM_A_REPLICATION_FLOOR_PERCENT = 50.0

#: The T=3 shortcut ceiling. An arm at or below it learned nothing usable.
SHORTCUT_CEILING_PERCENT = 26.6257

DEPTH_NOT_COMPUTE = "DEPTH_NOT_COMPUTE"
COMPUTE_NOT_DEPTH = "COMPUTE_NOT_DEPTH"
DEPTH_NEEDS_UNSHARED_PARAMETERS = "DEPTH_NEEDS_UNSHARED_PARAMETERS"
WIDTH_BEATS_DEPTH = "WIDTH_BEATS_DEPTH"
UNRESOLVED = "UNRESOLVED_ESCALATE"
ARM_A_REPLICATION_FAILED = "ARM_A_FAILED_TO_REPLICATE_BATCH03"


def inherited_matches_batch03() -> bool:
    """Whether every inherited field still matches Batch 02/03's configuration."""
    return all(BATCH05_CONFIG[f] == BATCH02_CONFIG[f] for f in INHERITED_FIELDS)


def resolved_batch05_config() -> Dict:
    return dict(BATCH05_CONFIG)


def arms_differ(first: Optional[float], second: Optional[float]) -> Optional[bool]:
    """Whether two arms' best exact match differ by more than the threshold.

    None when either value is missing: an absent measurement cannot establish a
    difference or the lack of one.
    """
    if first is None or second is None:
        return None
    return abs(first - second) > DIFFERENCE_THRESHOLD_POINTS


def compare(best_exact_match: Dict[str, Optional[float]],
            excluded: Sequence[str] = ()) -> Dict:
    """Apply PHASE2_BATCH_05_PREREG.md §7 mechanically.

    ``best_exact_match`` maps arm id ("A"/"B"/"C") to that arm's best exact match
    over its seeds. ``excluded`` names arms disqualified upstream — a broken
    bound, a degenerate result, or a SHORTCUT attribution — which are reported but
    take no part in the comparison.
    """
    usable = {k: v for k, v in best_exact_match.items() if k not in excluded and v is not None}
    a, b, c = usable.get("A"), usable.get("B"), usable.get("C")

    if a is None:
        return {"reading": ARM_A_REPLICATION_FAILED, "excluded": list(excluded),
                "reason": "arm A has no usable result, so there is no reference to compare against",
                "supports_depth_claim": False}
    if a < ARM_A_REPLICATION_FLOOR_PERCENT:
        return {"reading": ARM_A_REPLICATION_FAILED, "excluded": list(excluded),
                "reason": (f"arm A reached {a:.4f} % exact match, below the {ARM_A_REPLICATION_FLOOR_PERCENT} % "
                           "replication floor for Batch 03's K=4 cells"),
                "supports_depth_claim": False}

    low, high = UNRESOLVED_BAND
    if usable and all(low <= v <= high for v in usable.values()):
        return {"reading": UNRESOLVED, "excluded": list(excluded),
                "reason": (f"every usable arm falls in the {low}-{high} % band, where a 5-point threshold would "
                           "adjudicate a difference three seeds cannot support"),
                "supports_depth_claim": False}

    a_vs_b, a_vs_c = arms_differ(a, b), arms_differ(a, c)

    # Ordered as pre-registered: the two negative readings are checked before the
    # positive one, so a positive verdict cannot be reached while an alternative
    # arm is actually ahead.
    if c is not None and a_vs_c and c > a:
        return {"reading": DEPTH_NEEDS_UNSHARED_PARAMETERS, "excluded": list(excluded),
                "reason": f"arm C ({c:.4f} %) exceeds arm A ({a:.4f} %) by more than {DIFFERENCE_THRESHOLD_POINTS} points",
                "supports_depth_claim": False}
    if b is not None and a_vs_b and b > a:
        return {"reading": WIDTH_BEATS_DEPTH, "excluded": list(excluded),
                "reason": f"arm B ({b:.4f} %) exceeds arm A ({a:.4f} %) by more than {DIFFERENCE_THRESHOLD_POINTS} points",
                "supports_depth_claim": False}
    if b is not None and a_vs_b and a > b and (c is None or a_vs_c is False):
        return {"reading": DEPTH_NOT_COMPUTE, "excluded": list(excluded),
                "reason": (f"arm A ({a:.4f} %) exceeds arm B ({b:.4f} %) by more than "
                           f"{DIFFERENCE_THRESHOLD_POINTS} points"
                           + ("" if c is None else f", and matches arm C ({c:.4f} %)")),
                "supports_depth_claim": True}
    if b is not None and a_vs_b is False and (c is None or a_vs_c is False):
        return {"reading": COMPUTE_NOT_DEPTH, "excluded": list(excluded),
                "reason": ("all usable arms agree within "
                           f"{DIFFERENCE_THRESHOLD_POINTS} points, so the Batch 03 result was the compute"),
                "supports_depth_claim": False}
    return {"reading": UNRESOLVED, "excluded": list(excluded),
            "reason": "the arm pattern matches no pre-registered row",
            "supports_depth_claim": False}


def arm_is_excluded(classification: Optional[str], attribution: Optional[str],
                    best_exact_match: Optional[float], bound_held_flag: bool) -> Optional[str]:
    """Why an arm takes no part in the comparison, or None if it does.

    Checked in the pre-registered order: the bound first, because an arm whose
    stream was not bounded cannot speak to anything here.
    """
    if not bound_held_flag:
        return "post_training_bound_broken"
    if classification == "DEGENERATE":
        return "degenerate"
    if attribution == "SHORTCUT":
        return "attributed_to_shortcut"
    if best_exact_match is None:
        return "no_exact_match_recorded"
    return None
