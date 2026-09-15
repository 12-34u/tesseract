"""Main task, C1 span control, C2 word control, and the Z60 ablation."""

import numpy as np
import pytest

from phase2.groups import get_group
from phase2.tasks import TASKS, make_targets
from phase2.verification import reference_targets, verify_tasks, verify_z60_closed_form, z60_closed_form


@pytest.fixture
def states():
    return np.random.default_rng(11).integers(0, 60, size=(24, 17))


@pytest.mark.parametrize("group", ["A5", "Z60"])
def test_all_tasks_coincide_at_t1(group, states):
    g = get_group(group)
    first = make_targets(g, "main", states, 1)
    for task in TASKS:
        assert np.array_equal(make_targets(g, task, states, 1), first)


@pytest.mark.parametrize("group", ["A5", "Z60"])
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("t", [1, 2, 4, 8])
def test_targets_match_pure_python_reference(group, task, t, states):
    g = get_group(group)
    produced = make_targets(g, task, states[:6], t).tolist()
    table = g.table.tolist()
    assert produced == [reference_targets(table, task, row, t) for row in states[:6].tolist()]


@pytest.mark.parametrize("t", [2, 4, 8])
def test_tasks_differ_for_t_at_least_2_on_a5(t, states):
    g = get_group("A5")
    main, c1, c2 = (make_targets(g, task, states, t) for task in TASKS)
    assert not np.array_equal(main, c1) and not np.array_equal(main, c2) and not np.array_equal(c1, c2)


def test_c1_by_hand_on_z60():
    state = np.arange(17)[None, :]
    assert make_targets(get_group("Z60"), "c1_span", state, 4)[0, :3].tolist() == [0 + 4, 1 + 5, 2 + 6]
    assert make_targets(get_group("Z60"), "c1_span", state, 4)[0, 16] == (16 + 3) % 60  # wraps to cell 3


def test_c2_by_hand_on_z60():
    state = np.arange(17)[None, :]
    assert make_targets(get_group("Z60"), "c2_word", state, 3)[0, 0] == 0 + 1 + 2 + 3


def test_z60_main_task_has_a_closed_form_shortcut(states):
    assert np.array_equal(make_targets(get_group("Z60"), "main", states, 8), z60_closed_form(states, 8))
    assert verify_z60_closed_form()["passed"]


def test_closed_form_does_not_hold_for_a5(states):
    assert not np.array_equal(make_targets(get_group("A5"), "main", states, 4), z60_closed_form(states, 4))


@pytest.mark.parametrize("group", ["A5", "Z60"])
def test_task_verification_report_passes(group):
    from phase2.verification import A5_GENERATORS, cayley_table_from_elements, generate_by_closure
    table = cayley_table_from_elements(generate_by_closure(A5_GENERATORS)) if group == "A5" else get_group("Z60").table
    assert verify_tasks(group, np.asarray(table))["passed"]


def test_invalid_task_arguments(states):
    with pytest.raises(ValueError):
        make_targets(get_group("A5"), "main", states, 0)
    with pytest.raises(ValueError):
        make_targets(get_group("A5"), "copy", states, 1)
