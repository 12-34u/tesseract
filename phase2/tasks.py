"""Benchmark tasks: targets for the main task and its two controls.

PHASE2_BENCHMARK_DESIGN.md §4–§5. All three tasks share the input S₀ and the
dependency window s[i..i+T], and all three coincide at T = 1. They differ only
in composition depth:

* ``main``    y[i] = F^T(S₀)[i]                       depth T
* ``c1_span`` y[i] = s[i] · s[i+T]                    depth 1 at every T
* ``c2_word`` y[i] = s[i] · s[i+1] · … · s[i+T]        depth ⌈log2(T+1)⌉

Indices are taken mod n.
"""

import numpy as np

from phase2.automaton import iterate
from phase2.groups import FiniteGroup

TASKS = ("main", "c1_span", "c2_word")


def make_targets(group: FiniteGroup, task: str, states, t: int) -> np.ndarray:
    if t < 1:
        raise ValueError(f"T must be >= 1, got {t}")
    if task == "main":
        return iterate(group, states, t)

    s = iterate(group, states, 0)  # validated copy

    def cell(offset: int) -> np.ndarray:
        return np.roll(s, -offset, axis=-1)

    if task == "c1_span":
        return group.table[s, cell(t)]
    if task == "c2_word":
        product = s
        for j in range(1, t + 1):
            product = group.table[product, cell(j)]
        return product
    raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
