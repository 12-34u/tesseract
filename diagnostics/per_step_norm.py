"""Per-step normalisation of the recursive states, behind a config flag.

The residual confound is a statement about the hidden state *inside* the
recursion: with nothing bounding z_L between block executions, its norm grows
with the number of executions, so a K=8 model starts training in a different
regime from a K=1 model. A normalisation placed just before the vocabulary
projection cannot address this — it makes step-0 loss K-invariant by
construction while leaving every intermediate state untouched.

This module implements the alternative — normalise z_L and z_H after each block
call — as a wrapper around the frozen reasoner, switched by
:class:`PerStepNormConfig`. Turning the flag off must reproduce the frozen
recursion exactly, which is what makes an on/off comparison meaningful.

`models/` is frozen and `phase2/` is owned by the GPU box, so nothing here edits
either. The recursion is re-derived from the recorded update rule rather than
imported, which means an on/off run here is also an independent check on the
implementation that lives in `phase2/batch01.py`.

Mac CPU diagnostic code. It answers a structural question and produces no
pre-registered Arm result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# The recorded update rule, reproduced here so the wrapper can be read on its
# own (models/recursive_core.py is the authority):
#
#     m^(k)   = z_L^(k-1) + z_H^(k-1) + x_emb
#     z_L^(k) = f_theta(m^(k))
#     z_H^(k) = alpha * z_H^(k-1) + (1 - alpha) * z_L^(k)
#
# With the flag ON, z_L and z_H are each normalised at the end of the step,
# after the interpolation that produces z_H, so the states that enter step k+1
# are both bounded.


@dataclass(frozen=True)
class PerStepNormConfig:
    """Whether to normalise the recursive states, and which ones.

    ``enabled`` is the flag the diagnostic toggles. The two per-state switches
    exist so that a follow-up can attribute any effect to one state or the
    other; the diagnostic itself leaves both on, because the confound is about
    the stream as a whole.
    """

    enabled: bool = False
    normalise_z_L: bool = True
    normalise_z_H: bool = True
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.enabled and not (self.normalise_z_L or self.normalise_z_H):
            raise ValueError(
                "per-step norm is enabled but neither state is selected; "
                "this is silently identical to enabled=False"
            )


def per_step_norm_parameter_cost(d_model: int, config: PerStepNormConfig) -> int:
    """Parameters the flag adds: a weight and a bias per feature, per normalised state.

    The LayerNorms are shared across recursion steps, exactly as the block is,
    so the cost does not depend on K.
    """
    if not config.enabled:
        return 0
    states = int(config.normalise_z_L) + int(config.normalise_z_H)
    return states * 2 * d_model


class PerStepNormRecursion(nn.Module):
    """The frozen recursion, with the per-step normalisation flag applied.

    Wraps an existing reasoner and reuses its parameters — the same shared
    TransformerBlock and the same learnable initial states — so that flag ON and
    flag OFF differ by the normalisation and nothing else.

    No ``.detach()`` anywhere: BPTT runs through all K steps, as in the frozen
    reasoner.
    """

    def __init__(self, reasoner: nn.Module, config: PerStepNormConfig) -> None:
        super().__init__()
        for attribute in ("d_model", "num_recursive_steps", "alpha", "shared_block",
                          "learnable_init_H", "learnable_init_L"):
            if not hasattr(reasoner, attribute):
                raise ValueError(
                    f"reasoner is missing {attribute!r}; the per-step norm applies to "
                    "the shared-block family, whose recursion exposes these"
                )
        self.inner = reasoner
        self.config = config
        self.d_model = reasoner.d_model
        self.num_recursive_steps = reasoner.num_recursive_steps
        self.alpha = reasoner.alpha
        enabled = config.enabled
        self.norm_L = nn.LayerNorm(self.d_model, eps=config.eps) if enabled and config.normalise_z_L else None
        self.norm_H = nn.LayerNorm(self.d_model, eps=config.eps) if enabled and config.normalise_z_H else None

    @property
    def shared_block(self) -> nn.Module:
        return self.inner.shared_block

    def forward(
        self, x_emb: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        if x_emb.ndim != 3:
            raise ValueError(f"Expected 3D input [B, N, D], got shape {list(x_emb.shape)}.")
        b, n, d = x_emb.shape
        if d != self.d_model:
            raise ValueError(f"Input dimension D ({d}) does not match d_model ({self.d_model}).")

        z_H = self.inner.learnable_init_H.unsqueeze(0).unsqueeze(0).expand(b, n, d)
        z_L = self.inner.learnable_init_L.unsqueeze(0).unsqueeze(0).expand(b, n, d)
        history: Optional[List[Dict[str, torch.Tensor]]] = [] if return_states else None

        for _ in range(self.num_recursive_steps):
            merged = z_L + z_H + x_emb
            z_L = self.shared_block(merged)
            z_H = self.alpha * z_H + (1 - self.alpha) * z_L
            if self.norm_L is not None:
                z_L = self.norm_L(z_L)
            if self.norm_H is not None:
                z_H = self.norm_H(z_H)
            if return_states:
                history.append({"z_H": z_H, "z_L": z_L})

        return z_L, history


def apply_per_step_norm(model: nn.Module, config: PerStepNormConfig) -> Optional[str]:
    """Wrap ``model.reasoner`` with the flag applied. Returns the path patched, or None.

    With the flag off, nothing is wrapped: the model is left byte-identical to
    the frozen one, so the OFF condition cannot be contaminated by this module.
    """
    if not config.enabled:
        return None
    if not hasattr(model, "reasoner"):
        raise ValueError(
            "model has no `reasoner`; the per-step norm applies to the shared-block family"
        )
    model.reasoner = PerStepNormRecursion(model.reasoner, config)
    return "reasoner"


def hidden_state_rms(
    model: nn.Module, inputs: np.ndarray, device: torch.device
) -> List[Dict[str, float]]:
    """RMS of z_L and z_H after each recursion step, for an untrained model.

    RMS is taken over the feature dimension, then averaged over tokens and
    examples, so values are comparable across widths. This is the quantity the
    confound is about; step-0 loss is only a symptom of it.
    """
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(inputs)).to(device)
    with torch.no_grad():
        _, history = model(x, return_states=True)
    if not history:
        raise ValueError("model returned no state history; call with return_states=True support")
    rows = []
    for step, state in enumerate(history, start=1):
        row: Dict[str, float] = {"step": step}
        for key in ("z_L", "z_H"):
            value = state[key].float()
            row[f"{key}_rms"] = float(value.pow(2).mean(dim=-1).sqrt().mean().item())
            row[f"{key}_max_abs"] = float(value.abs().max().item())
        rows.append(row)
    return rows


def growth_ratio(rows: List[Dict[str, float]], key: str = "z_L_rms") -> float:
    """Last step's RMS over the first step's. 1.0 means the stream did not grow.

    At K=1 there is a single step, so the ratio is 1.0 by construction and says
    nothing; boundedness across K has to be read from the absolute values too.
    """
    return rows[-1][key] / rows[0][key]
