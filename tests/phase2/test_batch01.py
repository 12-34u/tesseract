"""Batch 01 diagnostic machinery (PHASE2_BATCH_01_PREREG.md).

These pin the two things that were silently wrong when the arms were first
written, and the one claim the arms must not be allowed to make by
construction.
"""

import numpy as np
import pytest
import torch
from torch import nn

from phase2.batch01 import (
    CurriculumStream,
    FixedSetStream,
    WarmupCosine,
    apply_final_layer_norm,
    find_output_head,
    hidden_state_norms,
    layer_norm_parameter_cost,
    step_zero_loss,
)
from phase2.config import ModelVariant, load_benchmark, resolve_model_variant
from phase2.data import SplitStore
from phase2.models import build_model
from phase2.runner import splits_dir
from utils.param_count import count_parameters
from utils.seed import set_seed

VOCAB = 60


@pytest.fixture(scope="module")
def setup():
    benchmark = load_benchmark()
    cfg = resolve_model_variant(ModelVariant(name="medium", base="phase2/prototype_medium"), VOCAB)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    val = store.get(benchmark.group, "main", "val", 4, benchmark.n, benchmark.val_size)
    return benchmark, cfg, val


def model_for(cfg, n, k, fix=False, family="tesseract"):
    set_seed(0)
    model = build_model(family, k, cfg, n).model
    if fix:
        apply_final_layer_norm(model, VOCAB)
    return model


# ---------------------------------------------------------------------------
# (b) the LayerNorm must sit directly before the vocabulary projection
# ---------------------------------------------------------------------------


def test_output_head_is_the_vocab_projection_not_the_widest_layer(setup):
    """The feed-forward expansion is wider than the vocabulary.

    Selecting the widest Linear picks the 1024-unit FFN instead of the 60-unit
    output head, which silently makes the fix a no-op. It must be matched on
    out_features == vocab_size.
    """
    _, cfg, _ = setup
    model = model_for(cfg, 17, 8)
    head, name, d_model = find_output_head(model, VOCAB)
    assert head.out_features == VOCAB
    assert d_model == cfg.d_model
    widest = max((m for m in model.modules() if isinstance(m, nn.Linear)), key=lambda m: m.out_features)
    assert widest.out_features > VOCAB and widest is not head


def test_layer_norm_is_inserted_directly_before_the_vocab_projection(setup):
    _, cfg, _ = setup
    model = model_for(cfg, 17, 8, fix=True)
    wrapper = dict(model.named_modules())["output_head"]
    assert isinstance(wrapper.norm, nn.LayerNorm)
    assert isinstance(wrapper.head, nn.Linear)
    assert wrapper.head.out_features == VOCAB
    assert wrapper.norm.normalized_shape == (cfg.d_model,)


def test_find_output_head_refuses_an_ambiguous_model():
    class TwoHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, VOCAB)
            self.b = nn.Linear(8, VOCAB)

    with pytest.raises(ValueError, match="exactly one"):
        find_output_head(TwoHeads(), VOCAB)


# ---------------------------------------------------------------------------
# (a) the fix must actually change step-0 loss at K=8
# ---------------------------------------------------------------------------


def test_fix_changes_step_zero_loss_at_k8(setup):
    """A no-op patch would leave step-0 loss untouched, which is how the first
    implementation failed: it reported 33.23 both with and without the fix."""
    benchmark, cfg, val = setup
    off = step_zero_loss(model_for(cfg, benchmark.n, 8), val, torch.device("cpu"), 512)
    on = step_zero_loss(model_for(cfg, benchmark.n, 8, fix=True), val, torch.device("cpu"), 512)
    assert off > 20.0, "K=8 without the fix should start far above chance"
    assert abs(off - on) > 1.0, "the fix must change step-0 loss at K=8"
    assert on == pytest.approx(np.log(VOCAB), abs=0.3), "with the fix, K=8 should start near chance"


def test_fix_leaves_k1_step_zero_loss_close_to_chance_either_way(setup):
    benchmark, cfg, val = setup
    off = step_zero_loss(model_for(cfg, benchmark.n, 1), val, torch.device("cpu"), 512)
    on = step_zero_loss(model_for(cfg, benchmark.n, 1, fix=True), val, torch.device("cpu"), 512)
    for value in (off, on):
        assert value == pytest.approx(np.log(VOCAB), abs=0.5)


# ---------------------------------------------------------------------------
# What the fix does NOT do: the confound itself is untouched
# ---------------------------------------------------------------------------


def test_output_side_fix_does_not_change_the_residual_growth(setup):
    """Equal step-0 loss under the fix is a property of the construction.

    The LayerNorm normalises the read-out and never touches the recursion, so
    the hidden-state growth that constitutes the confound is identical with it
    on and off. This is pinned so nobody can later report the output-side fix
    as having removed the confound.
    """
    benchmark, cfg, val = setup
    inputs = val.inputs[:64]
    off = hidden_state_norms(model_for(cfg, benchmark.n, 8), inputs, torch.device("cpu"))
    on = hidden_state_norms(model_for(cfg, benchmark.n, 8, fix=True), inputs, torch.device("cpu"))
    assert [r["z_L_rms"] for r in off] == pytest.approx([r["z_L_rms"] for r in on])
    assert off[-1]["z_L_rms"] / off[0]["z_L_rms"] > 5.0, "the stream should still grow steeply with K"


def test_residual_growth_rises_with_recursion_depth(setup):
    benchmark, cfg, val = setup
    inputs = val.inputs[:64]
    lasts = []
    for k in (1, 2, 4, 8):
        rows = hidden_state_norms(model_for(cfg, benchmark.n, k), inputs, torch.device("cpu"))
        assert len(rows) == k
        lasts.append(rows[-1]["z_L_rms"])
    assert lasts == sorted(lasts), "final-state norm must be non-decreasing in K"
    assert lasts[-1] > 5 * lasts[0]


# ---------------------------------------------------------------------------
# The fix must reach every family, and keep parameter matching honest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["tesseract", "unrolled", "param_matched", "width_scaled"])
def test_fix_applies_identically_to_every_family(setup, family):
    _, cfg, _ = setup
    model = model_for(cfg, 17, 8, family=family)
    before = count_parameters(model)["trainable"]
    _, _, d_model = find_output_head(model, VOCAB)
    apply_final_layer_norm(model, VOCAB)
    after = count_parameters(model)["trainable"]
    assert after - before == layer_norm_parameter_cost(d_model)


def test_parameter_matching_survives_the_added_layer_norm(setup):
    """param_matched targets Tesseract's count; the norm must not break that."""
    _, cfg, _ = setup
    counts = {}
    for family in ("tesseract", "param_matched"):
        model = model_for(cfg, 17, 8, fix=True, family=family)
        counts[family] = count_parameters(model)["trainable"]
    relative = abs(counts["param_matched"] - counts["tesseract"]) / counts["tesseract"]
    assert relative < 0.01, f"parameter matching drifted to {relative:.2%} with the fix applied"


# ---------------------------------------------------------------------------
# Streams and schedule
# ---------------------------------------------------------------------------


def test_fixed_set_stream_repeats_one_frozen_set(setup):
    benchmark, _, val = setup
    stream = FixedSetStream(benchmark.group, "main", 4, benchmark.n, 128, 512,
                            benchmark.data_seed, 0, excluded_inputs=[val.inputs])
    assert stream.inputs.shape == (512, benchmark.n)
    seen = set()
    for _ in range(20):
        inputs, _ = stream.next_batch()
        seen.update(map(bytes, inputs.astype(np.uint8)))
    assert len(seen) <= 512
    frozen = {bytes(row) for row in stream.inputs.astype(np.uint8)}
    assert seen <= frozen, "every batch must come from the frozen set"


def test_fixed_set_stream_excludes_validation_inputs(setup):
    benchmark, _, val = setup
    stream = FixedSetStream(benchmark.group, "main", 4, benchmark.n, 128, 512,
                            benchmark.data_seed, 0, excluded_inputs=[val.inputs])
    excluded = {bytes(row) for row in val.inputs.astype(np.uint8)}
    assert not ({bytes(row) for row in stream.inputs.astype(np.uint8)} & excluded)


def test_curriculum_totals_the_committed_step_count(setup):
    """The committed schedule is 5,000 at each of T=1,2,4 then 20,000 at target."""
    benchmark, _, val = setup
    stream = CurriculumStream(benchmark.group, "main", benchmark.n, 128, benchmark.data_seed, 0,
                              [1, 2, 4], 5000, 4, 20000, excluded_inputs=[val.inputs])
    boundaries = stream.stage_boundaries()
    assert [b["t"] for b in boundaries] == [1, 2, 4, 4]
    assert boundaries[-1]["ends_at_step"] == 35000


def test_curriculum_advances_through_its_stages(setup):
    benchmark, _, val = setup
    stream = CurriculumStream(benchmark.group, "main", benchmark.n, 128, benchmark.data_seed, 0,
                              [1, 2, 4], 10, 4, 20, excluded_inputs=[val.inputs])
    seen = []
    for _ in range(45):
        stream.next_batch()
        seen.append(stream.current_t)
    assert seen[0] == 1 and seen[9] == 1
    assert seen[10] == 2
    assert seen[-1] == 4


def test_warmup_cosine_matches_its_preregistered_shape():
    schedule = WarmupCosine(peak=1e-3, final=1e-4, warmup_steps=2000, total_steps=40000)
    assert schedule.lr_at(0) == pytest.approx(1e-3 / 2000)
    assert schedule.lr_at(1999) == pytest.approx(1e-3)
    assert schedule.lr_at(40000) == pytest.approx(1e-4)
    rising = [schedule.lr_at(s) for s in range(0, 2000, 200)]
    assert rising == sorted(rising)
    falling = [schedule.lr_at(s) for s in range(2000, 40000, 4000)]
    assert falling == sorted(falling, reverse=True)


# ---------------------------------------------------------------------------
# Arm 2c — per-step normalisation must actually bound the stream
# ---------------------------------------------------------------------------


def test_per_step_norm_bounds_the_residual_growth(setup):
    """The fix Arm 2c tests, and the property it is chosen for.

    The output-side norm leaves growth at 17.7x; this must hold the state
    bounded at every recursion step and every K, which is what the Arm 2c-i
    gate checks before any training runs.
    """
    from phase2.batch01 import apply_per_step_norm

    benchmark, cfg, val = setup
    inputs = val.inputs[:64]
    for k in (1, 2, 4, 8):
        set_seed(0)
        model = build_model("tesseract", k, cfg, benchmark.n).model
        apply_per_step_norm(model)
        rows = hidden_state_norms(model, inputs, torch.device("cpu"))
        assert len(rows) == k
        growth = rows[-1]["z_L_rms"] / rows[0]["z_L_rms"]
        assert growth < 1.05, f"K={k} still grows {growth:.2f}x under per-step normalisation"


def test_per_step_norm_reuses_the_shared_block_and_keeps_params_k_independent(setup):
    """The fix must not add a second block or make parameters depend on K."""
    from models.block import TransformerBlock
    from phase2.batch01 import apply_per_step_norm, per_step_norm_parameter_cost

    _, cfg, _ = setup
    counts = []
    for k in (1, 8):
        set_seed(0)
        model = build_model("tesseract", k, cfg, 17).model
        before = count_parameters(model)["trainable"]
        apply_per_step_norm(model)
        after = count_parameters(model)["trainable"]
        assert after - before == per_step_norm_parameter_cost(cfg.d_model)
        assert len([m for m in model.modules() if isinstance(m, TransformerBlock)]) == 1
        counts.append(after)
    assert counts[0] == counts[1], "parameter count must stay independent of K"


def test_per_step_norm_preserves_the_recorded_update_rule(setup):
    """Same shared block, same learnable inits, same K, full BPTT retained."""
    from phase2.batch01 import PerStepNormalisedReasoner, apply_per_step_norm

    _, cfg, _ = setup
    set_seed(0)
    model = build_model("tesseract", 4, cfg, 17).model
    original_block = model.reasoner.shared_block
    original_alpha = model.reasoner.alpha
    apply_per_step_norm(model)
    assert isinstance(model.reasoner, PerStepNormalisedReasoner)
    assert model.reasoner.shared_block is original_block
    assert model.reasoner.alpha == original_alpha
    assert model.reasoner.num_recursive_steps == 4

    x = torch.randint(0, VOCAB, (2, 17))
    logits, history = model(x, return_states=True)
    assert len(history) == 4
    logits.sum().backward()
    assert original_block.attn.qkv_proj.weight.grad is not None, "gradient must reach the shared block"


def test_match_parameters_accounts_for_added_normalisation(setup):
    """Without the correction the matched baseline inherits a drift."""
    from phase2.models import match_parameters, model_parameter_count

    _, cfg, _ = setup
    target_plain = model_parameter_count(VOCAB, cfg.d_model, cfg.d_ff, cfg.max_seq_len, 1)
    target_fixed = target_plain + 4 * cfg.d_model

    d_un, f_un = match_parameters(cfg, 8, target_fixed)
    uncorrected = model_parameter_count(VOCAB, d_un, f_un, cfg.max_seq_len, 8) + 4 * d_un
    d_c, f_c = match_parameters(cfg, 8, target_fixed, normalisation_parameters_per_d_model=4)
    corrected = model_parameter_count(VOCAB, d_c, f_c, cfg.max_seq_len, 8) + 4 * d_c

    assert abs(corrected - target_fixed) <= abs(uncorrected - target_fixed)
    assert abs(corrected - target_fixed) / target_fixed < 0.01


def test_match_parameters_is_unchanged_when_no_normalisation_is_added(setup):
    """The default must reproduce the recorded baseline widths exactly."""
    from phase2.models import match_parameters

    _, cfg, _ = setup
    for layers in (1, 2, 4, 8):
        target = 837_436
        assert match_parameters(cfg, layers, target) == \
            match_parameters(cfg, layers, target, normalisation_parameters_per_d_model=0)


# ---------------------------------------------------------------------------
# Addendum 02 — non-affine per-step normalisation, and arm isolation
# ---------------------------------------------------------------------------


def test_per_step_norm_is_non_affine_and_free(setup):
    """Addendum 02: no learnable gain, and therefore no added parameters.

    An affine gain is a free parameter: nothing in the objective stops it
    growing with depth during training, which would undo the bound while the
    fix was still nominally in place.
    """
    from phase2.batch01 import apply_per_step_norm, per_step_norm_parameter_cost

    _, cfg, _ = setup
    set_seed(0)
    model = build_model("tesseract", 8, cfg, 17).model
    before = count_parameters(model)["trainable"]
    apply_per_step_norm(model)
    after = count_parameters(model)["trainable"]

    assert after == before, "non-affine normalisation must add no parameters"
    assert per_step_norm_parameter_cost(cfg.d_model) == 0
    assert model.reasoner.norm_L.weight is None and model.reasoner.norm_L.bias is None
    assert model.reasoner.norm_H.weight is None and model.reasoner.norm_H.bias is None
    assert model.reasoner.elementwise_affine is False


def test_affine_variant_remains_available_and_costs_parameters(setup):
    """Addendum 01's variant is not removed, only no longer the default."""
    from phase2.batch01 import apply_per_step_norm, per_step_norm_parameter_cost

    _, cfg, _ = setup
    set_seed(0)
    model = build_model("tesseract", 8, cfg, 17).model
    before = count_parameters(model)["trainable"]
    apply_per_step_norm(model, elementwise_affine=True)
    after = count_parameters(model)["trainable"]
    assert after - before == per_step_norm_parameter_cost(cfg.d_model, elementwise_affine=True) == 4 * cfg.d_model


def test_non_affine_norm_still_bounds_the_stream(setup):
    from phase2.batch01 import apply_per_step_norm

    benchmark, cfg, val = setup
    for k in (1, 2, 4, 8):
        set_seed(0)
        model = build_model("tesseract", k, cfg, benchmark.n).model
        apply_per_step_norm(model)
        rows = hidden_state_norms(model, val.inputs[:64], torch.device("cpu"))
        assert rows[-1]["z_L_rms"] / rows[0]["z_L_rms"] < 1.05
        assert rows[-1]["z_L_rms"] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("arm", ["2b", "3", "4"])
def test_arms_other_than_2c_are_unchanged_by_the_addendum_02_change(arm):
    """The Addendum 02 change must reach Arm 2c and nothing else.

    These are the resolved settings for the arms that were already running or
    queued when the change landed, recorded byte-for-byte as they were before
    it. A change that leaked into them would alter one of these values.
    """
    from phase2.batch01 import resolved_arm_config

    expected = {
        "2b": {"fix": "output", "curriculum": False, "steps": 20000, "seeds": 3, "fixed_set": None},
        "3": {"fix": "none", "curriculum": True, "steps": 35000, "seeds": 3, "fixed_set": None},
        "4": {"fix": "none", "curriculum": False, "steps": 40000, "seeds": 3, "fixed_set": None},
    }[arm]
    assert resolved_arm_config(arm) == expected
    assert resolved_arm_config(arm)["fix"] != "per_step", "only Arm 2c may use per-step normalisation"


def test_only_arm_2c_resolves_to_per_step_normalisation():
    from phase2.batch01 import ARM_CONFIG

    per_step = [arm for arm, cfg in ARM_CONFIG.items() if cfg["fix"] == "per_step"]
    assert per_step == ["2c"]


@pytest.mark.parametrize("arm", ["2b", "3", "4"])
def test_arms_other_than_2c_build_models_with_no_added_parameters(setup, arm):
    """The strongest isolation check: the models those arms build are untouched."""
    from phase2.batch01 import apply_final_layer_norm, resolved_arm_config

    _, cfg, _ = setup
    fix = resolved_arm_config(arm)["fix"]
    set_seed(0)
    model = build_model("tesseract", 8, cfg, 17).model
    baseline = count_parameters(model)["trainable"]
    if fix == "output":
        apply_final_layer_norm(model, VOCAB)
        assert count_parameters(model)["trainable"] == baseline + 2 * cfg.d_model
    else:
        assert count_parameters(model)["trainable"] == baseline
