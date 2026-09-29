"""Amended classification for Batch 01 (PHASE2_BATCH_01_ADDENDUM_03.md).

Two defects in the original rule, found when Arm 2b reported:

1. **The above-chance bar sat below the majority-class baseline.** It was built
   from uniform chance (100/60 = 1.6667 %) and landed around 1.89-1.90 %, while
   simply emitting the most-frequent target token of the T=4 validation split
   scores 1.9244 %. A model that learned nothing cleared the bar.

2. **Nothing detected collapse.** Both Arm 2b seed-2 cells predicted a single
   token at all 34,816 positions, and that token was the modal class. The
   accuracy was real; the learning was not.

This module re-derives the classification from recorded artifacts. It does not
change the pipeline: the arms classify themselves as pre-registered while they
run, and this is applied afterwards to every arm and probe uniformly, so no arm
is measured by a different rule than any other. Both classifications are
reported side by side.
"""

import collections
from typing import Dict, List, Optional, Sequence

import numpy as np

from phase2.probes import LEARNABLE, NOT_LEARNABLE, PARTIAL, classify

#: A cell emitting at most this many distinct tokens is degenerate, whatever it
#: scores. Three allows a little more than a pure constant without admitting
#: anything that could be called a solution over a 60-token vocabulary.
COLLAPSE_MAX_DISTINCT_TOKENS = 3

DEGENERATE = "DEGENERATE"


def majority_class_percent(targets: np.ndarray) -> float:
    """Score of always emitting the most frequent target token, as a percentage.

    This is the floor any claim of "above chance" has to clear: it is reachable
    with no input dependence at all.
    """
    counts = collections.Counter(np.asarray(targets).ravel().tolist())
    _, most = counts.most_common(1)[0]
    return 100.0 * most / np.asarray(targets).size


def distinct_predicted_tokens(predictions: np.ndarray) -> int:
    return int(np.unique(np.asarray(predictions)).size)


def is_degenerate(distinct: Optional[int]) -> bool:
    return distinct is not None and distinct <= COLLAPSE_MAX_DISTINCT_TOKENS


def amend_seed(seed_row: Dict, majority_percent: float, distinct: Optional[int]) -> Dict:
    """One seed's amended verdict, keeping the original alongside it."""
    accuracy = seed_row.get("best_val_token_accuracy")
    degenerate = is_degenerate(distinct)
    beats_majority = accuracy is not None and accuracy > majority_percent
    amended_above = bool(seed_row.get("above_chance")) and beats_majority and not degenerate
    return {
        **seed_row,
        "distinct_predicted_tokens": distinct,
        "majority_class_percent": majority_percent,
        "beats_majority_class": beats_majority,
        "degenerate": degenerate,
        "above_chance_original": bool(seed_row.get("above_chance")),
        "above_chance_amended": amended_above,
        # A degenerate cell cannot count as learnable either, whatever it scored.
        "learnable_amended": bool(seed_row.get("learnable")) and not degenerate,
    }


def amend_classification(original: Dict, majority_percent: float,
                         distinct_by_seed: Dict[int, Optional[int]]) -> Dict:
    """Re-derive one arm-and-K classification under the amended rule."""
    per_seed = [amend_seed(row, majority_percent, distinct_by_seed.get(row.get("seed")))
                for row in original.get("per_seed", [])]

    learnable_seeds = [s["seed"] for s in per_seed if s["learnable_amended"]]
    above_seeds = [s["seed"] for s in per_seed if s["above_chance_amended"]]
    degenerate_seeds = [s["seed"] for s in per_seed if s["degenerate"]]
    required = original.get("bars", {}).get("seeds_required", 2)

    if len(learnable_seeds) >= required:
        amended = LEARNABLE
    elif above_seeds:
        amended = PARTIAL
    elif per_seed and len(degenerate_seeds) == len(per_seed):
        amended = DEGENERATE
    else:
        amended = NOT_LEARNABLE

    return {
        "classification_original": original.get("classification"),
        "classification_amended": amended,
        "changed": amended != original.get("classification"),
        "majority_class_percent": majority_percent,
        "collapse_threshold": COLLAPSE_MAX_DISTINCT_TOKENS,
        "seeds_above_chance_original": original.get("seeds_above_chance", []),
        "seeds_above_chance_amended": above_seeds,
        "seeds_meeting_learnable_bar_amended": learnable_seeds,
        "degenerate_seeds": degenerate_seeds,
        "per_seed": per_seed,
    }


def amend_report(classification_by_k: Dict[str, Dict], majority_percent: float,
                 distinct_by_k_seed: Dict[str, Dict[int, Optional[int]]]) -> Dict:
    """Amend every K of one arm or probe."""
    return {k: amend_classification(original, majority_percent, distinct_by_k_seed.get(k, {}))
            for k, original in classification_by_k.items()}


def moved_under_arm5_criteria(amended_by_k: Dict[str, Dict], required_seeds: int = 2) -> bool:
    """Whether an arm 'moves' per PHASE2_BATCH_01_ARM5_CRITERIA.md §1.

    LEARNABLE counts; PARTIAL counts only on at least two seeds. A single
    above-chance seed does not, which the probe report established is the modal
    outcome under the null across several cells.
    """
    for amended in amended_by_k.values():
        if amended["classification_amended"] == LEARNABLE:
            return True
        if (amended["classification_amended"] == PARTIAL
                and len(amended["seeds_above_chance_amended"]) >= required_seeds):
            return True
    return False
