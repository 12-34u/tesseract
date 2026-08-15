"""Multi-Head Self-Attention implementation for Tesseract."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Encoder-style Multi-Head Self-Attention module using Einops.

    Args:
        d_model (int): Model embedding dimension (D).
        num_heads (int): Number of parallel attention heads (H).
        dropout (float): Dropout probability applied to attention weights. Default: 0.0.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Multi-Head Attention.

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
                f"Input dimension D ({D}) does not match module d_model ({self.d_model})."
            )

        # [B, N, D] -> [B, N, 3D]
        qkv = self.qkv_proj(x)

        # Split Q, K, V and reshape to [B, H, N, d_head]
        q, k, v = rearrange(
            qkv,
            "b n (three h d) -> three b h n d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute context: [B, H, N, d_head]
        context = torch.matmul(attn_weights, v)

        # Reshape back to [B, N, D]
        out = rearrange(context, "b h n d -> b n (h d)")

        # Output projection
        output = self.out_proj(out)
        return output
