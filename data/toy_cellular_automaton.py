"""1D Binary Cellular Automaton Dataset for Tesseract.

Generates (initial_state, target_state) pairs where:
    target_state = apply_rule(initial_state, T times)

This provides a controlled iterative-computation benchmark:
    - T controls the required number of sequential transformations
    - K controls the model's recursive computation depth
    - The research question: does K ≥ T help performance?

Supported rules:
    - Rule 90 (XOR of left and right neighbours)
    - Rule 30 (left XOR (centre OR right))
    - Any valid 1D elementary CA rule (0–255)

Example (Rule 90, T=1):
    Input:  [0, 0, 1, 0, 0]
    Target: [0, 1, 0, 1, 0]

Example (Rule 90, T=2):
    Input:  [0, 0, 1, 0, 0]
    → T=1:  [0, 1, 0, 1, 0]
    Target: [1, 0, 0, 0, 1]
"""

from typing import List, Literal, Tuple

import torch
from torch.utils.data import Dataset


# ============================================================================
# Ground-Truth Simulator
# ============================================================================


def rule_to_lookup(rule_number: int) -> List[int]:
    """Convert an elementary CA rule number (0–255) to an 8-entry lookup table.

    The lookup table maps each 3-bit neighbourhood (left, centre, right)
    to the next cell state {0, 1}.

    Args:
        rule_number: Integer in [0, 255].

    Returns:
        List of 8 ints (each 0 or 1), indexed by neighbourhood pattern.
    """
    if not (0 <= rule_number <= 255):
        raise ValueError(f"Rule number must be 0–255, got {rule_number}")
    return [(rule_number >> i) & 1 for i in range(8)]


def apply_rule_once(state: List[int], rule_number: int) -> List[int]:
    """Apply an elementary CA rule once to a 1D binary state.

    Uses periodic (wrap-around) boundary conditions.

    Args:
        state: List of 0s and 1s.
        rule_number: Elementary CA rule number (0–255).

    Returns:
        New state after one rule application.
    """
    lookup = rule_to_lookup(rule_number)
    n = len(state)
    new_state = []
    for i in range(n):
        left = state[(i - 1) % n]
        centre = state[i]
        right = state[(i + 1) % n]
        # The neighbourhood pattern encodes as: left*4 + centre*2 + right
        pattern = (left << 2) | (centre << 1) | right
        new_state.append(lookup[pattern])
    return new_state


def simulate(state: List[int], rule_number: int, steps: int) -> List[int]:
    """Apply the CA rule `steps` times to produce the final state.

    Args:
        state: Initial 1D binary state.
        rule_number: Elementary CA rule number (0–255).
        steps: Number of rule applications (T).

    Returns:
        State after T applications of the rule.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    current = list(state)
    for _ in range(steps):
        current = apply_rule_once(current, rule_number)
    return current


def simulate_trajectory(
    state: List[int], rule_number: int, steps: int
) -> List[List[int]]:
    """Return the full trajectory: [state_0, state_1, ..., state_T].

    Args:
        state: Initial 1D binary state.
        rule_number: Elementary CA rule number (0–255).
        steps: Number of rule applications (T).

    Returns:
        List of T+1 states showing the full evolution.
    """
    trajectory = [list(state)]
    current = list(state)
    for _ in range(steps):
        current = apply_rule_once(current, rule_number)
        trajectory.append(list(current))
    return trajectory


# ============================================================================
# Dataset
# ============================================================================


class CellularAutomatonDataset(Dataset):
    """1D binary cellular automaton dataset.

    Each example consists of (initial_state, target_state) where the target
    is obtained by applying the CA rule T times to the initial state.

    The binary states use token IDs {0, 1}, compatible with the existing
    Tesseract embedding system (vocab_size ≥ 2).

    Args:
        num_examples: Number of examples to generate.
        seq_len: Length of each 1D state (number of cells).
        rule_number: Elementary CA rule number (0–255). Default: 90.
        steps: Number of rule applications (T). Default: 1.
        seed: Random seed for reproducible generation. Default: 42.
    """

    def __init__(
        self,
        num_examples: int = 256,
        seq_len: int = 16,
        rule_number: int = 90,
        steps: int = 1,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.rule_number = rule_number
        self.steps = steps
        self.vocab_size = 2  # Binary: {0, 1}

        if not (0 <= rule_number <= 255):
            raise ValueError(f"Rule number must be 0–255, got {rule_number}")
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")

        # Generate random binary initial states
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        self.inputs = torch.randint(
            low=0,
            high=2,
            size=(num_examples, seq_len),
            generator=generator,
            dtype=torch.long,
        )

        # Compute targets by simulating the CA
        targets_list = []
        for i in range(num_examples):
            initial = self.inputs[i].tolist()
            final = simulate(initial, rule_number, steps)
            targets_list.append(final)

        self.targets = torch.tensor(targets_list, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (initial_state, target_state) pair.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (input, target), both of shape [seq_len].
        """
        return self.inputs[idx], self.targets[idx]
