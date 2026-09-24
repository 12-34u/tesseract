"""Low-order signal structure of the A5 CA task (analysis only; no training, no benchmark change).

For the main task, y[i] = F^T(s)[i] is a group word over the window s[i..i+T].
Question: how much of y[i] can any PROPER subset of that window determine?

Excluding exactly one window cell gives the upper bound for every smaller
subset (any smaller subset is contained in some all-but-one subset), so for each
excluded offset j we compute

    predictability(j) = E_rest [ max_y P(y | rest) ]

exactly over the 60 values of the excluded cell, averaged over sampled states.
The maximum over j is then the best accuracy obtainable from any proper subset.
Chance is 1/|G|.

If that maximum is at chance for a given T, gradient descent gets no partial
credit from any lower-order feature: the model must acquire the full
(T+1)-way interaction at once. That is a property of the task, independent of
model width.
"""

from typing import Dict, List, Sequence

import numpy as np

from phase2.automaton import iterate
from phase2.groups import get_group
from phase2.verification import symbolic_word


def word_occurrence_counts(t: int) -> Dict[int, int]:
    """How many times each window offset appears in the symbolic word for F^T."""
    counts: Dict[int, int] = {}
    for offset in symbolic_word(t):
        counts[offset] = counts.get(offset, 0) + 1
    return dict(sorted(counts.items()))


def predictability_excluding(group, t: int, n: int, offset: int, states: np.ndarray) -> float:
    """E_rest[max_y P(y | rest)] when the window cell at ``offset`` is unknown."""
    order = group.order
    samples = states.shape[0]
    expanded = np.repeat(states, order, axis=0)                     # [samples*order, n]
    values = np.tile(np.arange(order, dtype=np.int64), samples)
    expanded[np.arange(expanded.shape[0]), offset % n] = values      # vary the excluded cell
    targets = iterate(group, expanded, t)[:, 0].reshape(samples, order)
    counts = np.apply_along_axis(lambda row: np.bincount(row, minlength=order).max(), 1, targets)
    return float(counts.mean() / order)


def analyse_task(group_name: str, t: int, n: int, samples: int, seed: int) -> Dict:
    group = get_group(group_name)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(t, n)))
    states = rng.integers(0, group.order, size=(samples, n), dtype=np.int64)
    counts = word_occurrence_counts(t)
    per_offset = {j: predictability_excluding(group, t, n, j, states) for j in range(t + 1)}
    best_offset = max(per_offset, key=per_offset.get)
    chance = 1.0 / group.order
    return {
        "group": group_name, "T": t, "n": n, "samples": samples,
        "window_cells": t + 1,
        "word_letters": 2 ** t,
        "occurrences_per_offset": counts,
        "chance": chance,
        "predictability_excluding_offset": per_offset,
        "max_proper_subset_predictability": per_offset[best_offset],
        "most_informative_excluded_offset": best_offset,
        "ratio_to_chance": per_offset[best_offset] / chance,
        "full_window_determines_target": True,  # y is a function of the whole window by construction
        "interaction_order_required": t + 1,
        "window_pattern_space": float(group.order) ** (t + 1),
    }


def run_signal_analysis(group_name: str, t_values: Sequence[int], n: int, samples: int, seed: int) -> Dict:
    per_t: List[Dict] = [analyse_task(group_name, t, n, samples, seed) for t in t_values]
    return {
        "question": "best achievable accuracy from any proper subset of the dependency window",
        "method": "exact enumeration of the excluded cell's 60 values, averaged over sampled states",
        "per_T": {str(entry["T"]): entry for entry in per_t},
        "summary": {str(e["T"]): {"max_proper_subset_accuracy": e["max_proper_subset_predictability"],
                                  "chance": e["chance"], "ratio_to_chance": e["ratio_to_chance"]} for e in per_t},
    }


def predictability_excluding_set(group, t: int, n: int, offsets: Sequence[int], states: np.ndarray,
                                 chunk: int = 64) -> float:
    """E_known[max_y P(y | known)] when every cell in ``offsets`` is unknown.

    The unknown cells are enumerated exactly (|G|^len(offsets) completions), in
    chunks of sampled states to bound memory.
    """
    order = group.order
    grids = np.meshgrid(*[np.arange(order, dtype=np.int64)] * len(offsets), indexing="ij")
    combos = np.stack([g.ravel() for g in grids], axis=1)            # [order^k, k]
    total = combos.shape[0]
    best = []
    for start in range(0, states.shape[0], chunk):
        block = states[start:start + chunk]
        expanded = np.repeat(block, total, axis=0)
        tiled = np.tile(combos, (block.shape[0], 1))
        for column, offset in enumerate(offsets):
            expanded[:, offset % n] = tiled[:, column]
        targets = iterate(group, expanded, t)[:, 0].reshape(block.shape[0], total)
        best.extend(np.bincount(row, minlength=order).max() / total for row in targets)
    return float(np.mean(best))


def signal_order_sweep(group_name: str, t: int, n: int, samples: int, seed: int, max_excluded: int = 2) -> Dict:
    """Best predictability achievable when k window cells are unknown, k = 1..max_excluded.

    Exhaustive over which cells are unknown, so the result is the best any
    subset of size (T+1-k) can do.
    """
    from itertools import combinations

    group = get_group(group_name)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(t, n, max_excluded)))
    states = rng.integers(0, group.order, size=(samples, n), dtype=np.int64)
    out = {}
    for k in range(1, max_excluded + 1):
        scores = {offsets: predictability_excluding_set(group, t, n, offsets, states)
                  for offsets in combinations(range(t + 1), k)}
        best = max(scores, key=scores.get)
        out[k] = {"known_cells": t + 1 - k, "best_excluded_offsets": list(best), "best_predictability": scores[best],
                  "ratio_to_chance": scores[best] * group.order,
                  "all": {",".join(map(str, o)): round(v, 5) for o, v in sorted(scores.items())}}
    return out
