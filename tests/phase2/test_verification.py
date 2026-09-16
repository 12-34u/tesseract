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


# ---------------------------------------------------------------------------
# The T = 2 span-control shortcut, cross-checked against implementation B
# ---------------------------------------------------------------------------
#
# phase2/amendments.py owns the authoritative check (verify_involution_shortcut)
# and Amendment 01 records the expected agreement. These tests add what that
# check does not cover: confirmation from the *independent* Cayley table, and
# confinement of the shortcut to the diagnostic T.


def test_involution_count_from_the_independent_table_gives_the_amendment_figure():
    """Amendment 01's 16/60 must also hold for implementation B's table.

    F^2(s)[i] = s[i]·s[i+1]^2·s[i+2] collapses to the C1 span control exactly
    when s[i+1]^2 = e. Deriving the count from the closure-generated table
    checks the amendment's constant against a construction that shares no code
    with the generator it documents.
    """
    from phase2.config import load_amendment

    table = cayley_table_from_elements(generate_by_closure(A5_GENERATORS))
    identity = next(e for e in range(60) if all(table[e][x] == x for x in range(60)))
    involutions = sum(1 for x in range(60) if table[x][x] == identity)
    assert involutions == 16
    assert involutions / 60 == pytest.approx(4 / 15)

    recorded = {s.t: s for s in load_amendment("phase2/amendment_01").documented_shortcuts}[2]
    assert recorded.expected_token_agreement == pytest.approx(involutions / 60, abs=1e-12)


def test_the_t2_shortcut_does_not_reach_the_primary_depth_conditions():
    """The documented shortcut must stay confined to the diagnostic T.

    Amendment 01 keeps T=2 as a diagnostic and rests the depth claim on
    T = 4 and 8. That is only sound if the two-cell collapse does not recur
    there, so this pins the agreement at the primary T values near chance.
    """
    from phase2.automaton import iterate
    from phase2.config import load_amendment

    amendment = load_amendment("phase2/amendment_01")
    group = get_group("A5")
    states = np.random.default_rng(5).integers(0, 60, size=(4000, 17))
    for t in amendment.primary_depth_t_values:
        span = group.table[states, np.roll(states, -t, axis=-1)]
        agreement = (iterate(group, states, t) == span).mean()
        assert agreement < 0.05, f"T={t} agreement {agreement:.4f} is not near chance (1/60)"
