"""Gates G1/G2: they must pass on a well-defined A5 benchmark and fail on a collapsing one."""

from dataclasses import replace

import numpy as np
import pytest

from phase2.config import load_benchmark
from phase2.gates import (
    g1_cycles,
    g1_sensitivity,
    g1_uniformity,
    g2_shortcuts,
    gate_self_test,
    run_gates,
    shallow_candidates,
)
from phase2.groups import get_group


@pytest.fixture(scope="module")
def small_gates():
    benchmark = load_benchmark()
    return replace(benchmark.gates, uniformity_states=2000, cycle_states=16, cycle_steps=64,
                   sensitivity_states=1000, shortcut_states=1000)


def rng(seed=0):
    return np.random.default_rng(seed)


def test_benchmark_config_loads():
    benchmark = load_benchmark()
    assert (benchmark.group, benchmark.n, benchmark.t_values, benchmark.k_values) == ("A5", 17, (1, 2, 4, 8), (1, 2, 4, 8))


@pytest.mark.parametrize("t", [1, 2, 4, 8])
def test_a5_g1_properties_on_small_samples(t):
    group = get_group("A5")
    uniformity = g1_uniformity(group, t, 17, 2000, rng(t))
    assert uniformity["tv_distance"] <= 0.05
    sensitivity = g1_sensitivity(group, t, 17, 1000, rng(t + 10))
    assert sensitivity["min_inside_fraction"] >= 0.9 and sensitivity["max_outside_fraction"] == 0.0


def test_g2_not_applicable_at_t1():
    assert g2_shortcuts(get_group("A5"), 1, 17, 100, rng(0), 0.05)["passed"] is None


def test_g2_detects_the_a5_involution_shortcut_at_t2():
    """F²(s)[i] = s[i]·s[i+1]²·s[i+2] equals s[i]·s[i+2] exactly when s[i+1]² = e,
    which holds for 16 of A5's 60 elements (the identity plus 15 involutions).
    This is a real property of the benchmark, found by the gate."""
    group = get_group("A5")
    ids = np.arange(60)
    assert int((group.table[ids, ids] == group.identity).sum()) == 16
    report = g2_shortcuts(group, 2, 17, 3000, rng(20), 0.05)
    assert report["passed"] is False
    assert report["max_token_agreement_candidate"] == "s[i+0]*s[i+2]"
    assert report["max_token_agreement"] == pytest.approx(16 / 60, abs=0.02)


@pytest.mark.parametrize("t", [4, 8])
def test_g2_passes_on_a5_at_t4_and_t8(t):
    assert g2_shortcuts(get_group("A5"), t, 17, 3000, rng(t + 20), 0.05)["passed"] is True


def test_g2_flags_the_rule90_type_collapse_on_z2():
    report = g2_shortcuts(get_group("Z2"), 8, 17, 500, rng(1), 0.05)
    assert report["passed"] is False
    assert report["max_token_agreement"] == 1.0
    assert report["max_token_agreement_candidate"] in ("s[i+0]*s[i+8]", "s[i+8]*s[i+0]")


def test_g2_flags_a_task_that_equals_an_earlier_iterate():
    # Z2 at T=2 on a ring: F^2(s)[i] = s[i] + s[i+2] (two-cell) → collapse detected.
    assert g2_shortcuts(get_group("Z2"), 2, 17, 500, rng(2), 0.05)["passed"] is False


def test_sensitivity_flags_ignored_cells_on_z2():
    report = g1_sensitivity(get_group("Z2"), 8, 17, 500, rng(3))
    assert report["inside_window_changed_fraction"][0] == 1.0
    assert report["inside_window_changed_fraction"][4] == 0.0  # middle cells cancel mod 2


def test_uniformity_detects_a_biased_rule():
    class Constant:  # a "group" whose table maps everything to 0
        order = 60
        table = np.zeros((60, 60), dtype=np.int64)
    assert g1_uniformity(Constant, 1, 17, 500, rng(4))["tv_distance"] > 0.9


def test_cycles_detects_a_fixed_point():
    report = g1_cycles(get_group("Z2"), 4, 8, 16, rng(5))  # Z2 ring of 4 is nilpotent → reaches all-zero
    assert report["rows_with_repeated_state"] > 0


def test_shallow_candidates_include_controls_and_iterates():
    states = rng(6).integers(0, 60, size=(4, 17))
    names = shallow_candidates(get_group("A5"), states, 3)
    assert {"F^0", "F^2", "s[i+0]*s[i+3]", "window_product_forward", "window_product_reversed"} <= set(names)
    assert len(names) == 3 + 4 + 16 + 2


def test_self_test_detects_collapse(small_gates):
    report = gate_self_test(17, small_gates)
    assert report["G2_detects_collapse"] and report["G1_sensitivity_detects_collapse"]


def test_run_gates_is_deterministic(small_gates):
    benchmark = replace(load_benchmark(), t_values=(1, 2), ablation_groups=(), gates=small_gates)
    first, second = run_gates(benchmark), run_gates(benchmark)
    assert first == second
    assert first["primary"]["per_T"][1]["G1_passed"] and first["primary"]["per_T"][2]["G2_passed"] is False
    assert first["passed"] is False  # a single failing gate fails the whole benchmark
