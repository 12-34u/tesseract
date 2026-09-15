"""Finite groups for the Phase 2 benchmark — primary implementation.

Group elements are integer ids 0..order-1, and a group is fully described by
its Cayley table: ``table[a, b]`` is the id of ``a · b``. Ids are labels
only. The exception is cyclic groups, where the label *is* the residue.

A₅ convention (PHASE2_BENCHMARK_DESIGN.md §4.1):
* elements are the even permutations of (0, 1, 2, 3, 4), stored as image
  arrays p with p[x] = image of x;
* ids follow the lexicographic order of those arrays, so id 0 is the identity;
* the product is composition: (a · b)(x) = a(b(x)).

An independent construction, used for verification, lives in
``phase2/verification.py``.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np

BENCHMARK_GROUPS = ("A5", "Z60")


@dataclass(frozen=True, eq=False)
class FiniteGroup:
    name: str
    table: np.ndarray  # int64 [order, order], read-only; table[a, b] = a · b
    identity: int
    inverse: np.ndarray  # int64 [order], read-only

    @property
    def order(self) -> int:
        return int(self.table.shape[0])

    def multiply(self, a, b):
        """Vectorised product a · b over broadcastable id arrays."""
        return self.table[a, b]


def group_from_table(name: str, table) -> FiniteGroup:
    """Validate a Cayley table's identity and inverses and wrap it as a group.

    Associativity is deliberately not checked here; the independent
    verifier checks it.
    """
    table = np.array(table, dtype=np.int64)
    order = table.shape[0]
    if table.ndim != 2 or table.shape != (order, order):
        raise ValueError(f"{name}: Cayley table must be square, got shape {table.shape}")
    if table.min() < 0 or table.max() >= order:
        raise ValueError(f"{name}: Cayley table entries must be ids in [0, {order})")

    ids = np.arange(order)
    identities = [e for e in range(order) if np.array_equal(table[e], ids) and np.array_equal(table[:, e], ids)]
    if len(identities) != 1:
        raise ValueError(f"{name}: expected exactly one identity element, found {identities}")
    identity = identities[0]

    is_identity = table == identity
    if not (is_identity.sum(axis=1) == 1).all():
        raise ValueError(f"{name}: every element needs exactly one right inverse")
    inverse = is_identity.argmax(axis=1)
    if not (table[inverse, ids] == identity).all():
        raise ValueError(f"{name}: right inverses are not left inverses")

    table.setflags(write=False)
    inverse.setflags(write=False)
    return FiniteGroup(name=name, table=table, identity=identity, inverse=inverse)


def is_even_permutation(perm) -> bool:
    """Parity by counting inversions."""
    inversions = sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j])
    return inversions % 2 == 0


def a5_permutations() -> np.ndarray:
    """The 60 even permutations of 5 points in lexicographic order, shape [60, 5]."""
    return np.array([p for p in permutations(range(5)) if is_even_permutation(p)], dtype=np.int64)


def _build_a5() -> FiniteGroup:
    perms = a5_permutations()
    products = perms[:, perms]  # products[a, b, x] = perms[a][perms[b][x]] = (a · b)(x)
    weights = 5 ** np.arange(5)
    perm_keys = perms @ weights
    product_keys = products @ weights
    sorter = np.argsort(perm_keys)
    table = sorter[np.searchsorted(perm_keys[sorter], product_keys)]
    if not np.array_equal(perm_keys[table], product_keys):
        raise RuntimeError("A5 product produced a permutation outside the element set")
    return group_from_table("A5", table)


def _build_cyclic(order: int) -> FiniteGroup:
    ids = np.arange(order)
    return group_from_table(f"Z{order}", (ids[:, None] + ids[None, :]) % order)


_BUILDERS = {
    "A5": _build_a5,
    "Z60": lambda: _build_cyclic(60),
    "Z2": lambda: _build_cyclic(2),  # used only to test that the gates can detect collapse
}


@lru_cache(maxsize=None)
def get_group(name: str) -> FiniteGroup:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown group {name!r}; available: {sorted(_BUILDERS)}")
    return _BUILDERS[name]()
