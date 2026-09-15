"""Group cellular automaton — primary implementation.

State: a ring of n cells, each a group element id.
Transition: F(s)[i] = s[i] · s[(i + 1) mod n]   (PHASE2_BENCHMARK_DESIGN.md §4.3)
"""

import numpy as np

from phase2.groups import FiniteGroup


def _validated(group: FiniteGroup, states) -> np.ndarray:
    states = np.asarray(states)
    if not np.issubdtype(states.dtype, np.integer):
        raise TypeError(f"states must be integer ids, got dtype {states.dtype}")
    if states.ndim < 1 or states.shape[-1] < 2:
        raise ValueError(f"states must have a last (ring) dimension of size >= 2, got shape {states.shape}")
    if states.size and (states.min() < 0 or states.max() >= group.order):
        raise ValueError(f"state ids must be in [0, {group.order})")
    return states.astype(np.int64, copy=True)


def ca_step(group: FiniteGroup, states) -> np.ndarray:
    """One synchronous update; ``states`` has shape [..., n]."""
    s = _validated(group, states)
    return group.table[s, np.roll(s, -1, axis=-1)]


def iterate(group: FiniteGroup, states, steps: int) -> np.ndarray:
    """F^steps(states) by explicit iteration (steps = 0 returns a copy)."""
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    s = _validated(group, states)
    for _ in range(steps):
        s = group.table[s, np.roll(s, -1, axis=-1)]
    return s
