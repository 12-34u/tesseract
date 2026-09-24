"""Low-order signal analysis: hand-checkable cases on A5 and Z2."""

import numpy as np
import pytest

from phase2.groups import get_group
from phase2.signal_analysis import analyse_task, predictability_excluding, word_occurrence_counts


@pytest.mark.parametrize("t, expected", [(1, {0: 1, 1: 1}), (2, {0: 1, 1: 2, 2: 1}), (4, {0: 1, 1: 4, 2: 6, 3: 4, 4: 1})])
def test_occurrence_counts_are_binomial(t, expected):
    assert word_occurrence_counts(t) == expected


def test_t1_gives_no_signal_from_either_single_cell():
    # y = s0·s1: fixing one cell leaves y uniform, so a single cell predicts at chance.
    report = analyse_task("A5", 1, 17, samples=200, seed=0)
    assert report["max_proper_subset_predictability"] == pytest.approx(1 / 60, abs=0.005)


def test_t2_middle_cell_matches_the_known_involution_shortcut():
    # y = s0·s1²·s2 equals s0·s2 exactly when s1² = e (16 of 60 elements), cf. Amendment 01.
    report = analyse_task("A5", 2, 17, samples=400, seed=1)
    assert report["most_informative_excluded_offset"] == 1
    assert report["predictability_excluding_offset"][1] == pytest.approx(16 / 60, abs=0.02)
    assert report["predictability_excluding_offset"][0] == pytest.approx(1 / 60, abs=0.01)


def test_excluding_a_once_occurring_cell_is_always_chance():
    group = get_group("A5")
    states = np.random.default_rng(2).integers(0, 60, size=(200, 17), dtype=np.int64)
    for t, offset in ((2, 0), (2, 2), (4, 0), (4, 4)):
        assert predictability_excluding(group, t, 17, offset, states) == pytest.approx(1 / 60, abs=0.01)
