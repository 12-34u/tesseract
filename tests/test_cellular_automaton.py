"""Tests for the 1D cellular automaton simulator and dataset.

Verifies:
    - Simulator correctness (hand-checked examples)
    - Determinism
    - T=0 identity
    - T=1 vs manual calculation
    - T=2 double application
    - Binary validity
    - Dataset shapes and DataLoader compatibility
"""

import pytest
import torch
from torch.utils.data import DataLoader

from data.toy_cellular_automaton import (
    CellularAutomatonDataset,
    apply_rule_once,
    rule_to_lookup,
    simulate,
    simulate_trajectory,
)


# ============================================================================
# Simulator Tests
# ============================================================================


class TestRuleLookup:
    """Tests for rule_to_lookup conversion."""

    def test_rule_90_lookup(self):
        """Rule 90 = 01011010 in binary → specific lookup entries."""
        lookup = rule_to_lookup(90)
        assert len(lookup) == 8
        # Rule 90 binary: 01011010
        # bit 0=0, bit 1=1, bit 2=0, bit 3=1, bit 4=1, bit 5=0, bit 6=1, bit 7=0
        expected = [0, 1, 0, 1, 1, 0, 1, 0]
        assert lookup == expected

    def test_rule_30_lookup(self):
        """Rule 30 = 00011110 in binary."""
        lookup = rule_to_lookup(30)
        expected = [0, 1, 1, 1, 1, 0, 0, 0]
        assert lookup == expected

    def test_rule_0(self):
        """Rule 0: all outputs are 0."""
        lookup = rule_to_lookup(0)
        assert lookup == [0] * 8

    def test_rule_255(self):
        """Rule 255: all outputs are 1."""
        lookup = rule_to_lookup(255)
        assert lookup == [1] * 8

    def test_invalid_rule_negative(self):
        with pytest.raises(ValueError, match="0–255"):
            rule_to_lookup(-1)

    def test_invalid_rule_too_large(self):
        with pytest.raises(ValueError, match="0–255"):
            rule_to_lookup(256)


class TestApplyRuleOnce:
    """Tests for single-step rule application."""

    def test_rule_90_single_cell_active(self):
        """Rule 90 on [0,0,1,0,0] with periodic boundaries.

        Rule 90: new[i] = left XOR right
        Positions with periodic wrap:
            i=0: left=state[4]=0, right=state[1]=0 → 0 XOR 0 = 0
            i=1: left=state[0]=0, right=state[2]=1 → 0 XOR 1 = 1
            i=2: left=state[1]=0, right=state[3]=0 → 0 XOR 0 = 0
            i=3: left=state[2]=1, right=state[4]=0 → 1 XOR 0 = 1
            i=4: left=state[3]=0, right=state[0]=0 → 0 XOR 0 = 0
        """
        state = [0, 0, 1, 0, 0]
        result = apply_rule_once(state, 90)
        assert result == [0, 1, 0, 1, 0]

    def test_rule_90_all_zeros(self):
        """All-zero state stays all-zero under Rule 90."""
        state = [0, 0, 0, 0, 0]
        result = apply_rule_once(state, 90)
        assert result == [0, 0, 0, 0, 0]

    def test_rule_90_all_ones(self):
        """All-one state under Rule 90: each cell = 1 XOR 1 = 0."""
        state = [1, 1, 1, 1, 1]
        result = apply_rule_once(state, 90)
        assert result == [0, 0, 0, 0, 0]

    def test_output_length_preserved(self):
        """Output length equals input length."""
        for n in [4, 8, 16, 32]:
            state = [0] * n
            state[n // 2] = 1
            result = apply_rule_once(state, 90)
            assert len(result) == n

    def test_output_is_binary(self):
        """Every output element is 0 or 1."""
        state = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        result = apply_rule_once(state, 90)
        assert all(v in (0, 1) for v in result)

    def test_periodic_boundary(self):
        """Verify wrap-around: first and last cells see each other.

        State: [1, 0, 0, 0, 1]
        Rule 90: new[i] = left XOR right

        i=0: left=state[4]=1, right=state[1]=0 → 1 XOR 0 = 1
        i=4: left=state[3]=0, right=state[0]=1 → 0 XOR 1 = 1
        """
        state = [1, 0, 0, 0, 1]
        result = apply_rule_once(state, 90)
        assert result[0] == 1  # left=1, right=0 → 1
        assert result[4] == 1  # left=0, right=1 → 1


class TestSimulate:
    """Tests for multi-step simulation."""

    def test_t_0_identity(self):
        """T=0: output equals input."""
        state = [1, 0, 1, 1, 0]
        result = simulate(state, 90, steps=0)
        assert result == state

    def test_t_1_matches_single_apply(self):
        """T=1 matches apply_rule_once."""
        state = [0, 0, 1, 0, 0]
        result_sim = simulate(state, 90, steps=1)
        result_once = apply_rule_once(state, 90)
        assert result_sim == result_once

    def test_t_2_is_double_application(self):
        """T=2 applies the rule twice, not once.

        State: [0, 0, 1, 0, 0]
        After T=1: [0, 1, 0, 1, 0]
        After T=2: apply rule to [0, 1, 0, 1, 0]:
            i=0: left=0, right=1 → 0 XOR 1 = 1
            i=1: left=0, right=0 → 0 XOR 0 = 0
            i=2: left=1, right=1 → 1 XOR 1 = 0
            i=3: left=0, right=0 → 0 XOR 0 = 0
            i=4: left=1, right=0 → 1 XOR 0 = 1
        """
        state = [0, 0, 1, 0, 0]
        result = simulate(state, 90, steps=2)
        assert result == [1, 0, 0, 0, 1]

    def test_determinism(self):
        """Same input + same rule + same T → same output."""
        state = [1, 0, 1, 0, 1, 1, 0, 0]
        r1 = simulate(state, 90, steps=4)
        r2 = simulate(state, 90, steps=4)
        assert r1 == r2

    @pytest.mark.parametrize("t", [1, 2, 4, 8])
    def test_output_always_binary(self, t):
        """All outputs remain binary for various T values."""
        state = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
        result = simulate(state, 90, steps=t)
        assert all(v in (0, 1) for v in result)
        assert len(result) == len(state)

    def test_negative_steps_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            simulate([0, 1], 90, steps=-1)

    def test_rule_30_t_1(self):
        """Cross-check with Rule 30.

        Rule 30: new[i] = left XOR (centre OR right)
        State: [0, 0, 1, 0, 0]

        i=0: L=0, C=0, R=0 → 0 XOR (0 OR 0) = 0
        i=1: L=0, C=0, R=1 → 0 XOR (0 OR 1) = 1
        i=2: L=0, C=1, R=0 → 0 XOR (1 OR 0) = 1
        i=3: L=1, C=0, R=0 → 1 XOR (0 OR 0) = 1
        i=4: L=0, C=0, R=0 → 0 XOR (0 OR 0) = 0
        """
        state = [0, 0, 1, 0, 0]
        result = simulate(state, 30, steps=1)
        assert result == [0, 1, 1, 1, 0]


class TestSimulateTrajectory:
    """Tests for full trajectory generation."""

    def test_trajectory_length(self):
        """Trajectory has T+1 entries."""
        state = [0, 1, 0]
        traj = simulate_trajectory(state, 90, steps=4)
        assert len(traj) == 5  # T+1

    def test_trajectory_first_is_initial(self):
        state = [1, 0, 1]
        traj = simulate_trajectory(state, 90, steps=3)
        assert traj[0] == state

    def test_trajectory_last_matches_simulate(self):
        state = [0, 0, 1, 0, 0]
        traj = simulate_trajectory(state, 90, steps=4)
        direct = simulate(state, 90, steps=4)
        assert traj[-1] == direct


# ============================================================================
# Dataset Tests
# ============================================================================


class TestCellularAutomatonDataset:
    """Tests for the CellularAutomatonDataset."""

    def test_shapes(self):
        """Dataset returns correct shapes."""
        ds = CellularAutomatonDataset(num_examples=32, seq_len=16, steps=2)
        assert len(ds) == 32
        inp, tgt = ds[0]
        assert inp.shape == (16,)
        assert tgt.shape == (16,)

    def test_binary_values(self):
        """All inputs and targets are binary {0, 1}."""
        ds = CellularAutomatonDataset(num_examples=64, seq_len=16, steps=4)
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert (inp >= 0).all() and (inp <= 1).all(), f"Input out of range at {i}"
            assert (tgt >= 0).all() and (tgt <= 1).all(), f"Target out of range at {i}"

    def test_dtype(self):
        """Inputs and targets are long tensors."""
        ds = CellularAutomatonDataset(num_examples=8, seq_len=8, steps=1)
        inp, tgt = ds[0]
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_t0_identity(self):
        """With T=0, target equals input."""
        ds = CellularAutomatonDataset(num_examples=16, seq_len=8, steps=0)
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert torch.equal(inp, tgt), f"T=0 mismatch at index {i}"

    def test_t1_matches_simulator(self):
        """Dataset targets match the simulator for T=1."""
        ds = CellularAutomatonDataset(
            num_examples=16, seq_len=8, rule_number=90, steps=1, seed=42
        )
        for i in range(len(ds)):
            inp, tgt = ds[i]
            expected = simulate(inp.tolist(), 90, steps=1)
            assert tgt.tolist() == expected, f"Simulator mismatch at index {i}"

    def test_t2_matches_simulator(self):
        """Dataset targets match the simulator for T=2."""
        ds = CellularAutomatonDataset(
            num_examples=16, seq_len=8, rule_number=90, steps=2, seed=42
        )
        for i in range(len(ds)):
            inp, tgt = ds[i]
            expected = simulate(inp.tolist(), 90, steps=2)
            assert tgt.tolist() == expected, f"Simulator mismatch at index {i}"

    def test_deterministic_with_seed(self):
        """Same seed produces identical data."""
        ds1 = CellularAutomatonDataset(num_examples=16, seq_len=8, steps=2, seed=42)
        ds2 = CellularAutomatonDataset(num_examples=16, seq_len=8, steps=2, seed=42)
        for i in range(len(ds1)):
            inp1, tgt1 = ds1[i]
            inp2, tgt2 = ds2[i]
            assert torch.equal(inp1, inp2)
            assert torch.equal(tgt1, tgt2)

    def test_different_seed_different_data(self):
        """Different seeds produce different data."""
        ds1 = CellularAutomatonDataset(num_examples=32, seq_len=16, steps=1, seed=42)
        ds2 = CellularAutomatonDataset(num_examples=32, seq_len=16, steps=1, seed=99)
        any_different = False
        for i in range(len(ds1)):
            inp1, _ = ds1[i]
            inp2, _ = ds2[i]
            if not torch.equal(inp1, inp2):
                any_different = True
                break
        assert any_different

    def test_dataloader_integration(self):
        """Works with PyTorch DataLoader."""
        ds = CellularAutomatonDataset(num_examples=32, seq_len=8, steps=1)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        inputs, targets = batch
        assert inputs.shape == (8, 8)
        assert targets.shape == (8, 8)
        assert inputs.dtype == torch.long

    @pytest.mark.parametrize("t", [1, 2, 4, 8])
    def test_various_t_values(self, t):
        """Dataset generates valid data for various T."""
        ds = CellularAutomatonDataset(num_examples=16, seq_len=16, steps=t, seed=42)
        for i in range(len(ds)):
            inp, tgt = ds[i]
            expected = simulate(inp.tolist(), 90, steps=t)
            assert tgt.tolist() == expected

    def test_invalid_rule(self):
        with pytest.raises(ValueError):
            CellularAutomatonDataset(rule_number=300)

    def test_rule_30(self):
        """Dataset works with Rule 30."""
        ds = CellularAutomatonDataset(
            num_examples=8, seq_len=8, rule_number=30, steps=2, seed=42
        )
        for i in range(len(ds)):
            inp, tgt = ds[i]
            expected = simulate(inp.tolist(), 30, steps=2)
            assert tgt.tolist() == expected
