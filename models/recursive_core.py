"""Recursive Reasoning Core for Tesseract.

Implements the dual latent state (z_H, z_L) recursive architecture.
A SINGLE shared TransformerBlock is applied K times, with z_L capturing
the dynamic output and z_H tracking an EMA-like slower context state.

The entire recursive computation graph remains differentiable (no detach)
to support full BPTT during training.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.block import TransformerBlock


class RecursiveReasoner(nn.Module):
    """Dual-state recursive reasoning module using a shared TransformerBlock.

    Architecture per recursive step k = 1..K:
        m^(k)    = z_L^(k-1) + z_H^(k-1) + x_emb
        z_L^(k)  = f_theta(m^(k))           # shared TransformerBlock
        z_H^(k)  = alpha * z_H^(k-1) + (1 - alpha) * z_L^(k)

    Args:
        d_model (int): Hidden dimension (D).
        num_heads (int): Number of attention heads (H).
        d_ff (int): Feed-forward hidden dimension.
        num_recursive_steps (int): Number of recursive iterations (K).
        alpha (float): EMA decay factor for z_H updates. Default: 0.9.
        dropout (float): Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_recursive_steps: int = 4,
        alpha: float = 0.9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_recursive_steps = num_recursive_steps
        self.alpha = alpha

        # === ONE shared TransformerBlock — reused at every recursive step ===
        self.shared_block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Learnable initial states: [D] vectors, broadcast to [B, N, D] at runtime
        self.learnable_init_H = nn.Parameter(torch.zeros(d_model))
        self.learnable_init_L = nn.Parameter(torch.zeros(d_model))

        # Initialize with small random values for symmetry breaking
        nn.init.normal_(self.learnable_init_H, mean=0.0, std=0.02)
        nn.init.normal_(self.learnable_init_L, mean=0.0, std=0.02)

    def forward(
        self,
        x_emb: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """Execute K recursive steps on input embeddings.

        Args:
            x_emb (torch.Tensor): Embedded input of shape [B, N, D].
            return_states (bool): If True, return intermediate state history.

        Returns:
            Tuple containing:
                - z_L^(K): Final low-level state, shape [B, N, D].
                - state_history: List of dicts with 'z_H' and 'z_L' at each step,
                  or None if return_states is False.
        """
        if x_emb.ndim != 3:
            raise ValueError(
                f"Expected 3D input [B, N, D], got shape {list(x_emb.shape)}."
            )

        B, N, D = x_emb.shape

        if D != self.d_model:
            raise ValueError(
                f"Input dimension D ({D}) does not match d_model ({self.d_model})."
            )

        # === Initialize z_H^(0) and z_L^(0) by broadcasting [D] → [B, N, D] ===
        # Using unsqueeze + expand avoids unnecessary memory allocation (.repeat)
        z_H = self.learnable_init_H.unsqueeze(0).unsqueeze(0).expand(B, N, D)  # [B, N, D]
        z_L = self.learnable_init_L.unsqueeze(0).unsqueeze(0).expand(B, N, D)  # [B, N, D]

        state_history: Optional[List[Dict[str, torch.Tensor]]] = [] if return_states else None

        # === Recursive loop: k = 1 .. K ===
        for k in range(self.num_recursive_steps):
            # Merge input: m^(k) = z_L^(k-1) + z_H^(k-1) + x_emb
            merged = z_L + z_H + x_emb  # [B, N, D]

            # Apply shared TransformerBlock: z_L^(k) = f_theta(m^(k))
            z_L = self.shared_block(merged)  # [B, N, D]

            # EMA update: z_H^(k) = alpha * z_H^(k-1) + (1 - alpha) * z_L^(k)
            z_H = self.alpha * z_H + (1 - self.alpha) * z_L  # [B, N, D]

            # NOTE: No .detach() — full BPTT through all K steps

            if return_states:
                state_history.append({
                    "z_H": z_H,
                    "z_L": z_L,
                })

        return z_L, state_history
