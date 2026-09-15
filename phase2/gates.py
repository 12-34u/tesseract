"""Pre-training validity gates G1 and G2 (PHASE2_BENCHMARK_DESIGN.md §6).

G1 — non-degenerate dynamics:
  * uniformity: output tokens of F^T are ≈ uniform (TV distance threshold);
  * cycles: no repeated global state within ``cycle_steps`` iterations;
  * sensitivity: changing any one cell in the window s[i..i+T] changes F^T(s)[i]
    in at least ``sensitivity_min`` of cases, and cells outside the window
    never do.

G2 — no hidden algebraic collapse (T >= 2): F^T is not reproduced by any
shallow candidate: F^(T') for T' < T, any single cell s[i+a], any two-cell
product s[i+a]·s[i+b] (0 <= a, b <= T; this includes the Rule-90-type
collapse and the C1 span control), or a window product in either order
(C2 style).

The candidate formulas are written out here directly, independently of
``phase2/tasks.py``. A self-test runs both gates on Z2, where F^8 collapses to
s[i] + s[i+8]; the gates must flag that collapse, or their passing verdict on
A5 means nothing.
"""

from typing import Dict

import numpy as np

from phase2.automaton import iterate
from phase2.config import BenchmarkConfig, GateConfig
from phase2.groups import FiniteGroup, get_group

_GROUP_CODES = {"A5": 0, "Z60": 1, "Z2": 2}
_TEST_CODES = {"uniformity": 0, "cycles": 1, "sensitivity": 2, "shortcuts": 3}


def _rng(seed: int, group: str, t: int, test: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(_GROUP_CODES[group], t, _TEST_CODES[test])))


def _states(rng: np.random.Generator, group: FiniteGroup, count: int, n: int) -> np.ndarray:
    return rng.integers(0, group.order, size=(count, n), dtype=np.int64)


# ============================================================================
# G1
# ============================================================================


def g1_uniformity(group: FiniteGroup, t: int, n: int, num_states: int, rng: np.random.Generator) -> Dict:
    outputs = iterate(group, _states(rng, group, num_states, n), t)
    counts = np.bincount(outputs.ravel(), minlength=group.order)
    expected = outputs.size / group.order
    freq = counts / outputs.size
    return {
        "tokens": int(outputs.size),
        "tv_distance": float(0.5 * np.abs(freq - 1.0 / group.order).sum()),
        "chi2": float(((counts - expected) ** 2 / expected).sum()),
        "chi2_df": group.order - 1,
        "min_frequency": float(freq.min()),
        "max_frequency": float(freq.max()),
    }


def g1_cycles(group: FiniteGroup, n: int, num_states: int, max_steps: int, rng: np.random.Generator) -> Dict:
    states = _states(rng, group, num_states, n)
    first_seen = [dict() for _ in range(num_states)]
    repeats = {}  # row -> (step of repeat, cycle length)
    for step in range(max_steps + 1):
        for row in range(num_states):
            if row in repeats:
                continue
            key = states[row].tobytes()
            if key in first_seen[row]:
                repeats[row] = (step, step - first_seen[row][key])
            else:
                first_seen[row][key] = step
        states = iterate(group, states, 1)
    return {
        "states": num_states,
        "steps": max_steps,
        "rows_with_repeated_state": len(repeats),
        "shortest_cycle_length": min((c for _, c in repeats.values()), default=None),
        "earliest_repeat_step": min((s for s, _ in repeats.values()), default=None),
    }


def g1_sensitivity(group: FiniteGroup, t: int, n: int, num_states: int, rng: np.random.Generator) -> Dict:
    states = _states(rng, group, num_states, n)
    base = iterate(group, states, t)
    rows = np.arange(num_states)
    cells = rng.integers(0, n, size=num_states)

    def changed_fraction(offset: int) -> float:
        perturbed = states.copy()
        position = (cells + offset) % n
        # Replace the cell with a uniformly chosen *different* element.
        perturbed[rows, position] = (perturbed[rows, position] + rng.integers(1, group.order, size=num_states)) % group.order
        return float((iterate(group, perturbed, t)[rows, cells] != base[rows, cells]).mean())

    inside = {j: changed_fraction(j) for j in range(t + 1)}
    outside = {j: changed_fraction(j) for j in (-1, t + 1)}
    return {
        "inside_window_changed_fraction": inside,
        "min_inside_fraction": min(inside.values()),
        "outside_window_changed_fraction": outside,
        "max_outside_fraction": max(outside.values()),
    }


# ============================================================================
# G2
# ============================================================================


def shallow_candidates(group: FiniteGroup, states: np.ndarray, t: int) -> Dict[str, np.ndarray]:
    table = group.table

    def cell(a: int) -> np.ndarray:
        return np.roll(states, -a, axis=1)

    candidates: Dict[str, np.ndarray] = {}
    for t_prime in range(t):
        candidates[f"F^{t_prime}"] = iterate(group, states, t_prime)
    for a in range(t + 1):
        candidates[f"s[i+{a}]"] = cell(a)
        for b in range(t + 1):
            candidates[f"s[i+{a}]*s[i+{b}]"] = table[cell(a), cell(b)]
    forward = cell(0)
    for j in range(1, t + 1):
        forward = table[forward, cell(j)]
    backward = cell(t)
    for j in range(t - 1, -1, -1):
        backward = table[backward, cell(j)]
    candidates["window_product_forward"] = forward
    candidates["window_product_reversed"] = backward
    return candidates


def g2_shortcuts(group: FiniteGroup, t: int, n: int, num_states: int, rng: np.random.Generator,
                 agreement_max: float) -> Dict:
    if t < 2:
        return {"applicable": False, "reason": "T=1 is the primitive itself (identical to the controls by design)",
                "passed": None}
    states = _states(rng, group, num_states, n)
    target = iterate(group, states, t)
    token_agreement, sequence_agreement = {}, {}
    for name, prediction in shallow_candidates(group, states, t).items():
        equal = prediction == target
        token_agreement[name] = float(equal.mean())
        sequence_agreement[name] = float(equal.all(axis=1).mean())
    worst = max(token_agreement, key=token_agreement.get)
    return {
        "applicable": True,
        "num_candidates": len(token_agreement),
        "chance_token_agreement": 1.0 / group.order,
        "max_token_agreement": token_agreement[worst],
        "max_token_agreement_candidate": worst,
        "max_sequence_agreement": max(sequence_agreement.values()),
        "token_agreement": token_agreement,
        "passed": token_agreement[worst] <= agreement_max and max(sequence_agreement.values()) == 0.0,
    }


# ============================================================================
# Gate runner
# ============================================================================


def evaluate_group(group_name: str, t_values, n: int, gates: GateConfig) -> Dict:
    group = get_group(group_name)
    per_t = {}
    for t in t_values:
        uniformity = g1_uniformity(group, t, n, gates.uniformity_states, _rng(gates.seed, group_name, t, "uniformity"))
        sensitivity = g1_sensitivity(group, t, n, gates.sensitivity_states, _rng(gates.seed, group_name, t, "sensitivity"))
        shortcuts = g2_shortcuts(group, t, n, gates.shortcut_states, _rng(gates.seed, group_name, t, "shortcuts"),
                                 gates.shortcut_agreement_max)
        g1_passed = (uniformity["tv_distance"] <= gates.uniformity_tv_max
                     and sensitivity["min_inside_fraction"] >= gates.sensitivity_min
                     and sensitivity["max_outside_fraction"] == 0.0)
        per_t[t] = {
            "G1_uniformity": uniformity,
            "G1_sensitivity": sensitivity,
            "G1_passed": g1_passed,
            "G2": shortcuts,
            "G2_passed": shortcuts["passed"],
        }
    cycles = g1_cycles(group, n, gates.cycle_states, gates.cycle_steps, _rng(gates.seed, group_name, 0, "cycles"))
    cycles_passed = cycles["rows_with_repeated_state"] == 0
    passed = cycles_passed and all(r["G1_passed"] and r["G2_passed"] is not False for r in per_t.values())
    return {"group": group_name, "n": n, "per_T": per_t, "G1_cycles": cycles, "G1_cycles_passed": cycles_passed,
            "passed": passed}


def gate_self_test(n: int, gates: GateConfig) -> Dict:
    """Run the gates on Z2, where F^8 = s[i] + s[i+8]; both gates must fail there."""
    group = get_group("Z2")
    t = 8
    shortcuts = g2_shortcuts(group, t, n, 2000, _rng(gates.seed, "Z2", t, "shortcuts"), gates.shortcut_agreement_max)
    sensitivity = g1_sensitivity(group, t, n, 2000, _rng(gates.seed, "Z2", t, "sensitivity"))
    return {
        "group": "Z2",
        "T": t,
        "G2_detects_collapse": shortcuts["passed"] is False,
        "G2_max_token_agreement_candidate": shortcuts["max_token_agreement_candidate"],
        "G2_max_token_agreement": shortcuts["max_token_agreement"],
        "G1_sensitivity_detects_collapse": sensitivity["min_inside_fraction"] < gates.sensitivity_min,
        "G1_min_inside_fraction": sensitivity["min_inside_fraction"],
    }


def run_gates(benchmark: BenchmarkConfig) -> Dict:
    primary = evaluate_group(benchmark.group, benchmark.t_values, benchmark.n, benchmark.gates)
    ablations = {name: evaluate_group(name, benchmark.t_values, benchmark.n, benchmark.gates)
                 for name in benchmark.ablation_groups}
    self_test = gate_self_test(benchmark.n, benchmark.gates)
    self_test_ok = self_test["G2_detects_collapse"] and self_test["G1_sensitivity_detects_collapse"]
    return {
        "primary": primary,
        "ablations_informational": ablations,
        "self_test": self_test,
        "self_test_passed": self_test_ok,
        "passed": primary["passed"] and self_test_ok,
    }
