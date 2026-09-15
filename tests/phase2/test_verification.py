"""Independent verification system (implementation B) and its power to detect errors."""

from itertools import permutations

import numpy as np
import pytest

from phase2.groups import get_group, is_even_permutation
from phase2.verification import (
    A5_GENERATORS,
    cayley_table_from_elements,
    check_group_axioms,
    derived_series,
    evaluate_word,
    generate_by_closure,
    parity_from_cycles,
    run_verification,
    symbolic_word,
    verify_a5,
    verify_cyclic,
    verify_word_expansion,
    word_run_count,
)


def test_a5_verification_passes():
    report = verify_a5()
    assert report["passed"], {k: v for k, v in report["checks"].items() if not v}


def test_z60_verification_passes():
    report = verify_cyclic(60)
    assert report["passed"], {k: v for k, v in report["checks"].items() if not v}


def test_full_verification_report_passes():
    assert run_verification()["passed"]


def test_parity_methods_agree_on_all_permutations():
    for perm in permutations(range(5)):
        assert (parity_from_cycles(perm) == 0) == is_even_permutation(perm)


def test_closure_generates_exactly_a5():
    elements = generate_by_closure(A5_GENERATORS)
    assert len(elements) == 60 and all(parity_from_cycles(p) == 0 for p in elements)


def test_axiom_checker_detects_a_corrupted_table():
    table = get_group("A5").table.copy()
    table[7, [3, 4]] = table[7, [4, 3]]  # swap two products: still a Latin row, no longer a group
    report = check_group_axioms(table)
    assert report["closure"] and not report["associativity"]


def test_axiom_checker_detects_missing_identity():
    assert not check_group_axioms(np.array([[1, 1], [1, 1]]))["identity"]
    # A relabelled Z2 (identity = element 1) is still a valid group.
    assert check_group_axioms(np.array([[1, 0], [0, 1]]))["identity"]


def test_solvability_detection():
    assert derived_series(get_group("Z60").table) == (True, [60, 1])
    solvable, series = derived_series(get_group("A5").table)
    assert not solvable and series == [60, 60]


def test_non_generating_set_is_detected():
    # A single 5-cycle generates only Z5, so it cannot pass as A5.
    assert len(generate_by_closure([(1, 2, 3, 4, 0)])) == 5
    with pytest.raises(ValueError, match="not closed"):
        cayley_table_from_elements([(0, 1, 2, 3, 4), (1, 2, 3, 4, 0)])


@pytest.mark.parametrize("t, letters, runs", [(0, 1, 1), (1, 2, 2), (2, 4, 3), (4, 16, 12), (8, 256, 192)])
def test_word_structure(t, letters, runs):
    word = symbolic_word(t)
    assert len(word) == letters and word_run_count(word) == runs


def test_t2_word_by_hand():
    assert symbolic_word(2) == [0, 1, 1, 2]  # s_i · s_{i+1} · s_{i+1} · s_{i+2}


@pytest.mark.parametrize("group", ["A5", "Z60"])
def test_iteration_matches_word_expansion_up_to_t8(group):
    table_b = cayley_table_from_elements(generate_by_closure(A5_GENERATORS)) if group == "A5" \
        else np.array([[(a + b) % 60 for b in range(60)] for a in range(60)])
    report = verify_word_expansion(group, table_b, t_max=8, n=17, num_states=64, seed=5)
    assert report["passed"], report["per_T"]


def test_word_check_detects_a_wrong_transition():
    # Word evaluated with the transposed (opposite-order) table differs from F for non-abelian A5.
    table = get_group("A5").table
    states = np.random.default_rng(0).integers(0, 60, size=(32, 17))
    from phase2.automaton import iterate
    assert not np.array_equal(iterate(get_group("A5"), states, 2), evaluate_word(table.T, states, symbolic_word(2)))
