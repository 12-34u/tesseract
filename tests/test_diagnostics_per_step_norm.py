"""Tests for the Mac CPU per-step-normalisation diagnostic.

The diagnostic's value rests on two properties: with the flag off it must
reproduce the frozen recursion exactly, and with the flag on the state entering
each step must be bounded. Both are pinned here.
"""

import numpy as np
import pytest
import torch

from diagnostics.per_step_norm import (
    PerStepNormConfig,
    PerStepNormRecursion,
    apply_per_step_norm,
    growth_ratio,
    hidden_state_rms,
    per_step_norm_parameter_cost,
)
from models.recursive_core import RecursiveReasoner
from utils.param_count import count_parameters

D_MODEL = 32
N = 6
B = 4


def core(k: int) -> RecursiveReasoner:
    torch.manual_seed(0)
    return RecursiveReasoner(d_model=D_MODEL, num_heads=4, d_ff=64, num_recursive_steps=k, alpha=0.9)


def embeddings() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, N, D_MODEL)


def test_flag_off_reproduces_the_frozen_recursion_exactly():
    x = embeddings()
    frozen = core(4)
    wrapped = PerStepNormRecursion(core(4), PerStepNormConfig(enabled=False))
    expected, _ = frozen(x)
    actual, _ = wrapped(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_flag_off_adds_no_parameters_and_does_not_wrap():
    model = torch.nn.Module()
    model.reasoner = core(4)
    assert apply_per_step_norm(model, PerStepNormConfig(enabled=False)) is None
    assert isinstance(model.reasoner, RecursiveReasoner)


def test_flag_on_bounds_the_state_at_every_step_and_every_k():
    x = embeddings()
    for k in (1, 2, 4, 8):
        wrapped = PerStepNormRecursion(core(k), PerStepNormConfig(enabled=True))
        _, history = wrapped(x, return_states=True)
        assert len(history) == k
        for state in history:
            for key in ("z_L", "z_H"):
                rms = state[key].pow(2).mean(dim=-1).sqrt().mean().item()
                assert rms == pytest.approx(1.0, abs=0.01), f"K={k} {key} rms {rms}"


def test_flag_off_grows_with_k_which_is_the_confound():
    x = embeddings()
    off = PerStepNormRecursion(core(8), PerStepNormConfig(enabled=False))
    _, history = off(x, return_states=True)
    rows = [{"z_L_rms": s["z_L"].pow(2).mean(dim=-1).sqrt().mean().item()} for s in history]
    assert growth_ratio(rows) > 2.0


def test_parameter_cost_is_two_layer_norms_and_independent_of_k():
    on = PerStepNormConfig(enabled=True)
    assert per_step_norm_parameter_cost(D_MODEL, on) == 4 * D_MODEL
    assert per_step_norm_parameter_cost(D_MODEL, PerStepNormConfig(enabled=False)) == 0
    counts = set()
    for k in (1, 2, 4, 8):
        wrapped = PerStepNormRecursion(core(k), on)
        counts.add(count_parameters(wrapped)["trainable"] - count_parameters(core(k))["trainable"])
    assert counts == {4 * D_MODEL}


def test_non_affine_normalisation_adds_no_parameters():
    config = PerStepNormConfig(enabled=True, elementwise_affine=False)
    assert per_step_norm_parameter_cost(D_MODEL, config) == 0
    for k in (1, 2, 4, 8):
        wrapped = PerStepNormRecursion(core(k), config)
        added = count_parameters(wrapped)["trainable"] - count_parameters(core(k))["trainable"]
        assert added == 0, f"K={k} added {added}"
        assert wrapped.norm_L.weight is None and wrapped.norm_L.bias is None


def test_non_affine_has_nothing_to_learn_at_the_normalisation():
    """The bound cannot drift during training because no gradient reaches the norm."""
    x = embeddings()
    wrapped = PerStepNormRecursion(core(4), PerStepNormConfig(enabled=True, elementwise_affine=False))
    assert list(wrapped.norm_L.parameters()) == []
    assert list(wrapped.norm_H.parameters()) == []
    out, _ = wrapped(x)
    out.sum().backward()  # must not fail for want of parameters
    assert all(torch.isfinite(p.grad).all() for p in wrapped.parameters() if p.grad is not None)


def test_non_affine_matches_affine_at_initialisation():
    """Gain is 1 and bias 0 at init, so the untrained measurement cannot tell them apart."""
    x = embeddings()
    affine = PerStepNormRecursion(core(4), PerStepNormConfig(enabled=True))
    non_affine = PerStepNormRecursion(core(4), PerStepNormConfig(enabled=True, elementwise_affine=False))
    a, _ = affine(x)
    b, _ = non_affine(x)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_non_affine_bounds_the_state_at_every_k():
    x = embeddings()
    for k in (1, 2, 4, 8):
        wrapped = PerStepNormRecursion(core(k), PerStepNormConfig(enabled=True, elementwise_affine=False))
        _, history = wrapped(x, return_states=True)
        for state in history:
            for key in ("z_L", "z_H"):
                rms = state[key].pow(2).mean(dim=-1).sqrt().mean().item()
                assert rms == pytest.approx(1.0, abs=0.01), f"K={k} {key} rms {rms}"


def test_one_sided_normalisation_costs_half():
    config = PerStepNormConfig(enabled=True, normalise_z_H=False)
    assert per_step_norm_parameter_cost(D_MODEL, config) == 2 * D_MODEL
    wrapped = PerStepNormRecursion(core(2), config)
    assert wrapped.norm_H is None and wrapped.norm_L is not None


def test_enabling_with_neither_state_selected_is_rejected():
    with pytest.raises(ValueError, match="neither state is selected"):
        PerStepNormConfig(enabled=True, normalise_z_L=False, normalise_z_H=False)


def test_bptt_still_reaches_the_first_step_with_the_flag_on():
    x = embeddings().requires_grad_(True)
    wrapped = PerStepNormRecursion(core(4), PerStepNormConfig(enabled=True))
    out, _ = wrapped(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_a_reasoner_without_the_recursion_surface_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        PerStepNormRecursion(torch.nn.Linear(2, 2), PerStepNormConfig(enabled=True))


def test_hidden_state_rms_reports_one_row_per_step():
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, tokens, return_states=False):
            x = torch.zeros(tokens.shape[0], tokens.shape[1], D_MODEL)
            return self.inner(x, return_states=return_states)

    model = Wrapper(PerStepNormRecursion(core(4), PerStepNormConfig(enabled=True)))
    rows = hidden_state_rms(model, np.zeros((3, N), dtype=np.int64), torch.device("cpu"))
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert all(r["z_L_rms"] == pytest.approx(1.0, abs=0.01) for r in rows)
