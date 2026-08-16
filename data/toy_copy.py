"""Toy Copy/Reverse Task Dataset for Tesseract.

Generates short synthetic integer sequences for overfitting sanity checks.
Supports 'copy' (target = input) and 'reverse' (target = reversed input) modes.

Example (copy mode, vocab_size=16, seq_len=8):
    Input:  [3, 7, 2, 9, 4, 11, 0, 5]
    Target: [3, 7, 2, 9, 4, 11, 0, 5]

Example (reverse mode):
    Input:  [3, 7, 2, 9, 4]
    Target: [4, 9, 2, 7, 3]
"""

from typing import Literal, Tuple

import torch
from torch.utils.data import Dataset


class CopyDataset(Dataset):
    """Synthetic copy/reverse task dataset.

    Args:
        num_examples (int): Number of examples to generate. Default: 256.
        seq_len (int): Length of each sequence. Default: 8.
        vocab_size (int): Size of the token vocabulary. Default: 16.
        task (str): Task type — 'copy' or 'reverse'. Default: 'copy'.
        seed (int | None): Random seed for deterministic generation. Default: 42.
    """

    def __init__(
        self,
        num_examples: int = 256,
        seq_len: int = 8,
        vocab_size: int = 16,
        task: Literal["copy", "reverse"] = "copy",
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.task = task

        if task not in ("copy", "reverse"):
            raise ValueError(f"Unsupported task type: {task!r}. Use 'copy' or 'reverse'.")

        # Generate data deterministically
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        # Random integer token IDs: [num_examples, seq_len] in range [0, vocab_size)
        self.inputs = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_examples, seq_len),
            generator=generator,
            dtype=torch.long,
        )

        # Build targets based on task type
        if task == "copy":
            self.targets = self.inputs.clone()
        elif task == "reverse":
            self.targets = self.inputs.flip(dims=[1])

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) pair for the given index.

        Args:
            idx (int): Dataset index.

        Returns:
            Tuple of (input_sequence, target_sequence), both of shape [seq_len].
        """
        return self.inputs[idx], self.targets[idx]
