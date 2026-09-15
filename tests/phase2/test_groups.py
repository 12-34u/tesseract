"""Primary group implementation: A₅ convention, products, identity, inverses."""

import numpy as np
import pytest

from phase2.groups import a5_permutations, get_group, group_from_table, is_even_permutation


def id_of(perm):
    return int(np.flatnonzero((a5_permutations() == np.array(perm)).all(axis=1))[0])


def test_a5_elements_and_labelling():
    group = get_group("A5")
    perms = a5_permutations()
    assert group.order == 60 and perms.shape == (60, 5)
    assert perms[0].tolist() == [0, 1, 2, 3, 4] and group.identity == 0
    rows = [tuple(p) for p in perms.tolist()]
    assert rows == sorted(rows) and len(set(rows)) == 60
    assert all(is_even_permutation(p) for p in rows)


def test_product_is_composition():
    group, perms = get_group("A5"), a5_permutations()
    rng = np.random.default_rng(0)
    for a, b in rng.integers(0, 60, size=(300, 2)):
        c = group.multiply(a, b)
        assert perms[c].tolist() == [int(perms[a][perms[b][x]]) for x in range(5)]


def test_hand_checked_product():
    # (0 1 2)·(2 3 4): x=0→b→0→a→1, 1→1→2, 2→3→3, 3→4→4, 4→2→0  ⇒ (0 1 2 3 4)
    a, b = id_of((1, 2, 0, 3, 4)), id_of((0, 1, 3, 4, 2))
    assert get_group("A5").multiply(a, b) == id_of((1, 2, 3, 4, 0))


def test_non_commutative():
    a, b = id_of((1, 2, 0, 3, 4)), id_of((0, 1, 3, 4, 2))
    table = get_group("A5").table
    assert table[a, b] != table[b, a]


@pytest.mark.parametrize("name", ["A5", "Z60", "Z2"])
def test_identity_and_inverses(name):
    group = get_group(name)
    ids = np.arange(group.order)
    assert (group.table[ids, group.inverse] == group.identity).all()
    assert (group.table[group.inverse, ids] == group.identity).all()
    assert (group.table[group.identity] == ids).all()


def test_cyclic_groups_are_modular_addition():
    ids = np.arange(60)
    assert np.array_equal(get_group("Z60").table, (ids[:, None] + ids[None, :]) % 60)
    assert get_group("Z2").table.tolist() == [[0, 1], [1, 0]]


def test_tables_are_read_only():
    with pytest.raises(ValueError):
        get_group("A5").table[0, 0] = 1


def test_unknown_group():
    with pytest.raises(ValueError, match="Unknown group"):
        get_group("S5")


@pytest.mark.parametrize(
    "table, message",
    [
        ([[0, 1], [1, 1]], "right inverse"),
        ([[1, 1], [1, 1]], "identity"),
        ([[0, 1, 2], [1, 2, 0]], "square"),
        ([[0, 5], [5, 0]], "ids"),
    ],
)
def test_invalid_tables_rejected(table, message):
    with pytest.raises(ValueError, match=message):
        group_from_table("bad", table)
