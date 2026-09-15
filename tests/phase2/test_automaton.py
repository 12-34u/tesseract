"""Group cellular automaton F(s)[i] = s[i] · s[(i+1) mod n]."""

import numpy as np
import pytest

from phase2.automaton import ca_step, iterate
from phase2.groups import get_group


def reference_step(table, state):
    n = len(state)
    return [int(table[state[i], state[(i + 1) % n]]) for i in range(n)]


@pytest.mark.parametrize("name", ["A5", "Z60"])
def test_step_matches_cellwise_definition(name):
    group = get_group(name)
    states = np.random.default_rng(1).integers(0, group.order, size=(20, 17))
    stepped = ca_step(group, states)
    for state, out in zip(states.tolist(), stepped.tolist()):
        assert out == reference_step(group.table, state)


def test_wraparound_uses_first_cell():
    group = get_group("Z60")
    state = np.array([[1, 2, 3, 40]])
    assert ca_step(group, state).tolist() == [[3, 5, 43, 41]]  # last cell: 40 + 1


def test_iterate_zero_returns_a_copy():
    group = get_group("A5")
    states = np.random.default_rng(2).integers(0, 60, size=(4, 17))
    out = iterate(group, states, 0)
    assert np.array_equal(out, states) and out is not states


@pytest.mark.parametrize("a, b", [(1, 1), (1, 3), (2, 2), (3, 5), (4, 4)])
def test_iteration_composes(a, b):
    group = get_group("A5")
    states = np.random.default_rng(3).integers(0, 60, size=(16, 17))
    assert np.array_equal(iterate(group, states, a + b), iterate(group, iterate(group, states, a), b))


def test_invalid_inputs():
    group = get_group("A5")
    with pytest.raises(ValueError):
        iterate(group, np.array([[0, 60]]), 1)
    with pytest.raises(ValueError):
        iterate(group, np.array([[0, 1]]), -1)
    with pytest.raises(TypeError):
        ca_step(group, np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError):
        ca_step(group, np.array([3]))
