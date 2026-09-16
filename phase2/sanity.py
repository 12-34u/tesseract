"""Sanity baselines on a fixed split (PHASE2_BENCHMARK_DESIGN.md §4.14): chance, copy-input, oracle.

The oracle uses the independent pure-Python reference (implementation B),
not the generator that produced the targets. Its 100 % score therefore checks
the stored targets end to end.
"""

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from phase2.data import Split
from phase2.metrics import compute_metrics
from phase2.verification import A5_GENERATORS, cayley_table_from_elements, generate_by_closure, reference_targets


@lru_cache(maxsize=None)
def independent_table(group_name: str) -> Tuple[Tuple[int, ...], ...]:
    if group_name == "A5":
        return tuple(tuple(row) for row in cayley_table_from_elements(generate_by_closure(A5_GENERATORS)).tolist())
    if group_name.startswith("Z"):
        order = int(group_name[1:])
        return tuple(tuple((a + b) % order for b in range(order)) for a in range(order))
    raise ValueError(f"no independent implementation for {group_name!r}")


def _summary(predictions: np.ndarray, split: Split, order: int) -> Dict:
    metrics = compute_metrics(predictions, split.targets, order).to_dict()
    metrics.pop("per_position_accuracy")
    return metrics


def t2_shortcut_baseline(split: Split) -> Dict:
    """Amendment 01 baseline for the main task at T=2: predict s[i]·s[i+2].

    It is exact precisely when s[i+1]² = e, because F²(s)[i] = s[i]·s[i+1]²·s[i+2].
    """
    if split.task != "main" or split.t != 2:
        raise ValueError("the T=2 shortcut baseline applies only to the main task at T=2")
    table = independent_table(split.group)
    order, n = len(table), split.n
    identity = next(e for e in range(order) if list(table[e]) == list(range(order)))
    squares_to_identity = {x for x in range(order) if table[x][x] == identity}
    rows = split.inputs.tolist()
    predictions = np.array([[table[row[i]][row[(i + 2) % n]] for i in range(n)] for row in rows])
    condition = np.array([[row[(i + 1) % n] in squares_to_identity for i in range(n)] for row in rows])
    return {
        **_summary(predictions, split, order),
        "formula": "s[i]*s[i+2]",
        "condition": "s[i+1]^2 = e",
        "condition_token_rate_percent": float(condition.mean() * 100.0),
        "agreement_iff_condition": bool(np.array_equal(predictions == split.targets, condition)),
    }


def sanity_baselines(split: Split, seed: int) -> Dict:
    table = independent_table(split.group)
    order, n = len(table), split.n
    chance_predictions = np.random.default_rng(seed).integers(0, order, size=split.targets.shape)
    oracle_predictions = np.array([reference_targets(table, split.task, row, split.t) for row in split.inputs.tolist()])
    extra = {"t2_shortcut_s0_s2": t2_shortcut_baseline(split)} if split.task == "main" and split.t == 2 else {}
    return {
        **extra,
        "split": split.key,
        "chance_expected": {"token_accuracy": 100.0 / order, "exact_match": 100.0 * (1.0 / order) ** n,
                            "chance_normalised_token_accuracy": 0.0},
        "chance_sampled": _summary(chance_predictions, split, order),
        "copy_input": _summary(split.inputs, split, order),
        "oracle_independent_simulator": _summary(oracle_predictions, split, order),
    }
