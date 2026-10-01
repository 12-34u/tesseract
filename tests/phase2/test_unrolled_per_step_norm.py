"""Per-step normalisation for the unrolled family (Batch 05 §4.1, option 1).

Written before the implementation. Addendum 01 §A1.4 recorded that per-step
normalisation "applies to the shared-block family, which is the family that has
a recursion", and that the other baselines "need the equivalent check made
separately before any cross-family comparison under the fix". Measuring that
check found `apply_per_step_norm` refusing on the unrolled family, which has
4.98x hidden-state growth across four blocks — well outside the 1.05 gate and on
exactly the axis a compute-matched comparison varies.

The properties below are what make the fix usable for a cross-family comparison:
it must bound the stream, cost nothing in parameters, reuse the existing blocks
rather than reinitialise them, and reproduce the recorded update rule with only
the normalisation added. Any one of those failing makes Batch 05 uninterpretable
in a different way than the confound did.
"""

import numpy as np
import pytest
import torch
from torch import nn


from phase2.batch01 import apply_per_step_norm, hidden_state_norms
from phase2.config import ModelVariant, load_benchmark, resolve_model_variant
from phase2.data import SplitStore
from phase2.models import build_model
from phase2.runner import splits_dir
from utils.param_count import count_parameters
from utils.seed import set_seed

CPU = torch.device("cpu")
VOCAB = 60
LIMIT = 1.05


@pytest.fixture(scope="module")
def setup():
    benchmark = load_benchmark()
    cfg = resolve_model_variant(ModelVariant(name="medium", base="phase2/prototype_medium"), VOCAB)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    val = store.get(benchmark.group, "main", "val", 3, benchmark.n, benchmark.val_size)
    return benchmark, cfg, val


def built(family, depth, cfg, n, seed=0):
    set_seed(seed)
    return build_model(family, depth, cfg, n)


def growth(model, inputs):
    norms = hidden_state_norms(model, inputs, CPU)
    return norms[-1]["z_L_rms"] / norms[0]["z_L_rms"]


# ---------------------------------------------------------------------------
# Before: the unrolled family has the confound
# ---------------------------------------------------------------------------


def test_unrolled_is_unbounded_without_the_fix(setup):
    """The measurement that motivates this work; pinned so it cannot silently change."""
    benchmark, cfg, val = setup
    model = built("unrolled", 4, cfg, benchmark.n).model
    ratio = growth(model, val.inputs[:64])
    assert ratio > LIMIT, f"expected unbounded growth, got {ratio:.4f}x"
    assert ratio > 3.0, f"expected growth of several-fold, got {ratio:.4f}x"


# ---------------------------------------------------------------------------
# After: the fix applies, bounds the stream, and costs nothing
# ---------------------------------------------------------------------------


def test_the_fix_applies_to_the_unrolled_family(setup):
    benchmark, cfg, val = setup
    model = built("unrolled", 4, cfg, benchmark.n).model
    patched = apply_per_step_norm(model, elementwise_affine=False)
    assert patched, "apply_per_step_norm must report what it patched"


def test_the_fix_bounds_the_unrolled_stream(setup):
    benchmark, cfg, val = setup
    model = built("unrolled", 4, cfg, benchmark.n).model
    before = growth(model, val.inputs[:64])
    apply_per_step_norm(model, elementwise_affine=False)
    after = growth(model, val.inputs[:64])
    assert before > LIMIT
    assert after < LIMIT, f"expected bounded growth, got {after:.4f}x"
    assert after == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_the_fix_bounds_the_unrolled_stream_at_every_depth(setup, depth):
    benchmark, cfg, val = setup
    model = built("unrolled", depth, cfg, benchmark.n).model
    apply_per_step_norm(model, elementwise_affine=False)
    assert growth(model, val.inputs[:64]) < LIMIT


def test_the_fix_adds_no_parameters_to_the_unrolled_family(setup):
    """Non-affine, so arm C's parameter count must not move — the Batch 05
    comparison against arm B turns on it."""
    benchmark, cfg, val = setup
    model = built("unrolled", 4, cfg, benchmark.n).model
    before = count_parameters(model)
    apply_per_step_norm(model, elementwise_affine=False)
    assert count_parameters(model) == before


def test_the_normalisers_are_non_affine(setup):
    benchmark, cfg, val = setup
    model = built("unrolled", 4, cfg, benchmark.n).model
    apply_per_step_norm(model, elementwise_affine=False)
    norms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    added = [m for m in norms if m.elementwise_affine is False]
    assert added, "expected at least one non-affine LayerNorm"
    assert all(m.weight is None and m.bias is None for m in added)


def test_the_fix_reuses_the_existing_blocks_rather_than_rebuilding_them(setup):
    """A rebuilt block would reinitialise weights and silently change the run."""
    benchmark, cfg, val = setup
    model = built("unrolled", 4, cfg, benchmark.n).model
    before = [p.data_ptr() for p in model.parameters()]
    apply_per_step_norm(model, elementwise_affine=False)
    after = {p.data_ptr() for p in model.parameters()}
    assert set(before) <= after, "existing parameter tensors must be reused, not copied"


def test_the_two_families_apply_the_same_fix(setup):
    """Both must normalise z_H built from the un-normalised z_L.

    If the two families ordered this differently, Batch 05 would be comparing a
    shared-block model under one fix against an unrolled model under another,
    and any difference between arms A and C would be unattributable.
    """
    benchmark, cfg, val = setup
    tokens = torch.from_numpy(np.ascontiguousarray(val.inputs[:8]))

    shared = built("tesseract", 4, cfg, benchmark.n).model
    apply_per_step_norm(shared, elementwise_affine=False)
    unrolled = built("unrolled", 4, cfg, benchmark.n).model
    apply_per_step_norm(unrolled, elementwise_affine=False)
    shared.eval()
    unrolled.eval()

    # Same recurrence shape: four recorded steps, both states unit-RMS at each.
    with torch.no_grad():
        _, shared_hist = shared(tokens, return_states=True)
        _, unrolled_hist = unrolled(tokens, return_states=True)
    assert len(shared_hist) == len(unrolled_hist) == 4
    for hist in (shared_hist, unrolled_hist):
        for step in hist:
            for key in ("z_L", "z_H"):
                rms = step[key].float().pow(2).mean(dim=-1).sqrt().mean().item()
                assert rms == pytest.approx(1.0, abs=0.02), f"{key} RMS {rms:.4f} is not bounded"


def test_the_fix_preserves_the_recorded_update_rule(setup):
    """The SAME rule the shared family uses, which is the point.

        z_L_raw = block(z_L + z_H + x_emb)
        z_H     = a*z_H + (1 - a)*z_L_raw      <- un-normalised z_L, as in
        z_L     = norm_L(z_L_raw)                 PerStepNormalisedReasoner
        z_H     = norm_H(z_H)

    The ordering matters: z_H is built from the un-normalised z_L and then
    normalised, not from the normalised one. Taking z_H from the normalised z_L
    gives a different function, so a cross-family comparison would be comparing
    two different fixes. Computed by hand against the patched model's own blocks.
    """
    benchmark, cfg, val = setup
    build = built("unrolled", 4, cfg, benchmark.n)
    model = build.model
    apply_per_step_norm(model, elementwise_affine=False)
    model.eval()

    tokens = torch.from_numpy(np.ascontiguousarray(val.inputs[:8]))
    with torch.no_grad():
        logits, history = model(tokens, return_states=True)

        inner = getattr(model, "inner", model)
        x_emb = inner.embedding(tokens)
        b, n, d = x_emb.shape
        z_H = inner.learnable_init_H.expand(b, n, d)
        z_L = inner.learnable_init_L.expand(b, n, d)
        norm = nn.LayerNorm(d, elementwise_affine=False)
        for block in inner.blocks:
            z_L_raw = block(z_L + z_H + x_emb)
            z_H = inner.alpha * z_H + (1 - inner.alpha) * z_L_raw
            z_L = norm(z_L_raw)
            z_H = norm(z_H)
        expected = inner.output_head(z_L)

    assert torch.allclose(logits, expected, atol=1e-5)
    assert len(history) == 4


def test_the_fix_still_works_on_the_shared_block_family(setup):
    """The dispatch must not disturb the family it already served."""
    benchmark, cfg, val = setup
    model = built("tesseract", 8, cfg, benchmark.n).model
    before = growth(model, val.inputs[:64])
    patched = apply_per_step_norm(model, elementwise_affine=False)
    assert patched == "reasoner"
    assert before > LIMIT
    assert growth(model, val.inputs[:64]) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# The gate Batch 05 needs: all three arms bounded
# ---------------------------------------------------------------------------


def test_all_three_batch05_arms_pass_the_bounded_growth_gate(setup):
    benchmark, cfg, val = setup
    for family, depth in (("tesseract", 4), ("width_scaled", 4), ("unrolled", 4)):
        model = built(family, depth, cfg, benchmark.n).model
        apply_per_step_norm(model, elementwise_affine=False)
        ratio = growth(model, val.inputs[:64])
        assert ratio < LIMIT, f"{family} growth {ratio:.4f}x exceeds the gate"


def test_the_width_scaled_arms_bound_is_vacuous_not_earned(setup):
    """It executes one block, so there is no between-step growth to bound.

    Recorded as a test so the Batch 05 report cannot present it as the fix
    working there.
    """
    benchmark, cfg, val = setup
    model = built("width_scaled", 4, cfg, benchmark.n).model
    apply_per_step_norm(model, elementwise_affine=False)
    assert len(hidden_state_norms(model, val.inputs[:64], CPU)) == 1
