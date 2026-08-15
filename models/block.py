"""Pre-LayerNorm TransformerBlock implementation for Tesseract.

This block forms the core computation unit of Tesseract. It is designed to be
applied recursively over K iterations using the exact same weight instance.
"""

import torch
import torch.nn as nn
from models.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder Block.

    Follows the architecture:
        y = x + MHA(LN1(x))
        z = y + FFN(LN2(y))

    Designed for repeated, weight-shared recursive execution.

    Args:
        d_model (int): Hidden feature dimension size (D).
        num_heads (int): Number of parallel attention heads (H).
        d_ff (int): Hidden dimension size of feed-forward network (FFN).
        dropout (float): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Sublayer 1: Pre-LN + Multi-Head Self-Attention
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Sublayer 2: Pre-LN + Feed-Forward Network
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D].

        Returns:
            torch.Tensor: Output tensor of shape [B, N, D].
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input tensor [B, N, D], but got tensor with shape {list(x.shape)}."
            )

        B, N, D = x.shape
        if D != self.d_model:
            raise ValueError(
                f"Input dimension D ({D}) does not match block d_model ({self.d_model})."
            )

        # Pre-LN 1: MHA with residual connection
        normed_x = self.ln1(x)
        attn_out = self.attn(normed_x)
        y = x + self.dropout(attn_out)

        # Pre-LN 2: FFN with residual connection
        normed_y = self.ln2(y)
        ffn_out = self.ffn(normed_y)
        z = y + self.dropout(ffn_out)

        return z

