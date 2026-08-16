"""Token and Positional Embedding module for Tesseract.

Provides learned token embeddings (V → D) and learned positional embeddings
(max_seq_len → D). Output is the sum of both, producing x_emb of shape [B, N, D].
"""

import torch
import torch.nn as nn


class TokenPositionalEmbedding(nn.Module):
    """Learned token + positional embedding layer.

    Args:
        vocab_size (int): Size of token vocabulary (V).
        d_model (int): Model embedding dimension (D).
        max_seq_len (int): Maximum supported sequence length.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding: V → D
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding: max_seq_len → D
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass: tokens [B, N] → x_emb [B, N, D].

        Args:
            tokens (torch.Tensor): Integer token IDs of shape [B, N].

        Returns:
            torch.Tensor: Embedded representation of shape [B, N, D].

        Raises:
            ValueError: If input is not 2D or sequence length exceeds max_seq_len.
        """
        if tokens.ndim != 2:
            raise ValueError(
                f"Expected 2D input tensor [B, N], but got tensor with shape {list(tokens.shape)}."
            )

        B, N = tokens.shape

        if N > self.max_seq_len:
            raise ValueError(
                f"Sequence length {N} exceeds maximum allowed {self.max_seq_len}."
            )

        # Position indices: [0, 1, ..., N-1] → [N]
        positions = torch.arange(N, device=tokens.device)  # [N]

        # Token embedding: [B, N] → [B, N, D]
        tok_emb = self.token_embedding(tokens)

        # Positional embedding: [N] → [N, D] → broadcast to [B, N, D]
        pos_emb = self.position_embedding(positions)  # [N, D]

        # Sum token + positional embeddings → [B, N, D]
        x_emb = tok_emb + pos_emb

        return x_emb
