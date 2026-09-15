"""Full Tesseract Model — end-to-end differentiable recursive transformer.

Pipeline:
    tokens [B, N]
        → TokenPositionalEmbedding → x_emb [B, N, D]
        → RecursiveReasoner (K steps) → z_L^(K) [B, N, D]
        → Output head (Linear) → logits [B, N, V]
        → CrossEntropyLoss → scalar loss
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.embeddings import TokenPositionalEmbedding
from models.recursive_core import RecursiveReasoner
from utils.config import ModelConfig


class TesseractModel(nn.Module):
    """Tesseract: Recursive transformer model with dual latent states.

    Args:
        vocab_size (int): Vocabulary size (V).
        d_model (int): Model embedding dimension (D).
        num_heads (int): Number of attention heads.
        d_ff (int): Feed-forward hidden dimension.
        max_seq_len (int): Maximum sequence length.
        num_recursive_steps (int): Number of recursive iterations (K).
        alpha (float): EMA decay factor for z_H. Default: 0.9.
        dropout (float): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_recursive_steps: int = 4,
        alpha: float = 0.9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token + Positional Embedding
        self.embedding = TokenPositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
        )

        # Recursive Reasoning Core (shared block, K steps)
        self.reasoner = RecursiveReasoner(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_recursive_steps=num_recursive_steps,
            alpha=alpha,
            dropout=dropout,
        )

        # Output Head: [B, N, D] → [B, N, V]
        self.output_head = nn.Linear(d_model, vocab_size)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    @classmethod
    def from_config(cls, config: ModelConfig, num_recursive_steps: int) -> "TesseractModel":
        """Build a model from a validated :class:`ModelConfig` and recursive depth K."""
        return cls(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            num_recursive_steps=num_recursive_steps,
            alpha=config.alpha,
            dropout=config.dropout,
        )

    @property
    def num_recursive_steps(self) -> int:
        return self.reasoner.num_recursive_steps

    def forward(
        self,
        tokens: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """Forward pass: tokens → logits.

        Args:
            tokens (torch.Tensor): Integer token IDs of shape [B, N].
            return_states (bool): If True, also return intermediate z_H/z_L states.

        Returns:
            Tuple containing:
                - logits: Output logits of shape [B, N, V].
                - state_history: List of state dicts per recursive step, or None.
        """
        # tokens [B, N] → x_emb [B, N, D]
        x_emb = self.embedding(tokens)

        # x_emb [B, N, D] → z_L^(K) [B, N, D]
        z_L_final, state_history = self.reasoner(x_emb, return_states=return_states)

        # z_L^(K) [B, N, D] → logits [B, N, V]
        logits = self.output_head(z_L_final)

        return logits, state_history

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for token-level classification.

        Args:
            logits (torch.Tensor): Predicted logits of shape [B, N, V].
            targets (torch.Tensor): Target token IDs of shape [B, N].

        Returns:
            torch.Tensor: Scalar cross-entropy loss.
        """
        if logits.ndim != 3 or targets.shape != logits.shape[:2]:
            raise ValueError(
                f"Expected logits [B, N, V] and targets [B, N]; got logits "
                f"{list(logits.shape)} and targets {list(targets.shape)}."
            )
        B, N, V = logits.shape

        # Reshape for CrossEntropyLoss: [B*N, V] and [B*N]
        logits_flat = logits.reshape(B * N, V)
        targets_flat = targets.reshape(B * N)

        loss = self.loss_fn(logits_flat, targets_flat)
        return loss
