"""The selectable Tesseract widths: small, medium and the research-scale large.

The defining property under test is that recursive depth comes from executing
ONE shared block K times, never from more trainable blocks. So for every width,
the parameter count must be identical at K = 1, 2, 4 and 8, and the model must
contain exactly one TransformerBlock object however deep the recursion goes.
"""

from dataclasses import replace

import pytest
import torch

from models.block import TransformerBlock
from phase2.config import MODEL_BASES, MODEL_FAMILIES, ModelVariant, model_family, resolve_model_variant
from phase2.models import build_model, forward_flops, model_parameter_count
from training.trainer import build_optimizer, train_step, verify_bptt
from utils.config import ConfigError, OptimizerConfig
from utils.param_count import count_parameters
from utils.seed import set_seed

VOCAB, N = 60, 17
K_VALUES = (1, 2, 4, 8)

# Expected shapes, stated independently of the config files so a silent edit to
# a config is caught rather than absorbed.
EXPECTED = {
    "small": {"base": "prototype_small", "d_model": 128, "num_heads": 4, "d_ff": 512, "parameters": 222_140},
    "medium": {"base": "phase2/prototype_medium", "d_model": 256, "num_heads": 8, "d_ff": 1024, "parameters": 837_436},
    "large": {"base": "phase2/prototype_large", "d_model": 768, "num_heads": 24, "d_ff": 3072, "parameters": 7_230_780},
}


def config_for(name):
    family = model_family(name)
    return resolve_model_variant(ModelVariant(family.name, family.base), VOCAB)


@pytest.fixture(scope="module")
def configs():
    return {name: config_for(name) for name in EXPECTED}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_all_three_widths_and_removes_none():
    assert [f.name for f in MODEL_FAMILIES] == ["small", "medium", "large"]
    assert MODEL_BASES == tuple(EXPECTED[name]["base"] for name in ("small", "medium", "large"))


def test_registry_rejects_an_unknown_base():
    with pytest.raises(ConfigError, match="unknown model base"):
        model_family("prototype_enormous")


def test_registry_parameter_counts_match_the_measured_models(configs):
    for family in MODEL_FAMILIES:
        measured = build_model("tesseract", 1, configs[family.name], N).parameter_count
        assert measured == family.approx_parameters, f"{family.name}: registry says {family.approx_parameters}"


# ---------------------------------------------------------------------------
# Architecture and parameter counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED))
def test_model_shape_matches_the_expected_configuration(name, configs):
    expected, cfg = EXPECTED[name], configs[name]
    assert (cfg.d_model, cfg.num_heads, cfg.d_ff) == (expected["d_model"], expected["num_heads"], expected["d_ff"])
    assert cfg.vocab_size == VOCAB
    assert cfg.max_seq_len == 64


@pytest.mark.parametrize("name", list(EXPECTED))
def test_head_dimension_is_32_across_the_whole_family(name, configs):
    """Widths scale by adding heads and FFN width, not by widening each head."""
    cfg = configs[name]
    assert cfg.d_model % cfg.num_heads == 0
    assert cfg.d_model // cfg.num_heads == 32


@pytest.mark.parametrize("name", list(EXPECTED))
def test_ffn_is_four_times_d_model(name, configs):
    cfg = configs[name]
    assert cfg.d_ff == 4 * cfg.d_model


@pytest.mark.parametrize("name", list(EXPECTED))
def test_exact_parameter_count(name, configs):
    built = build_model("tesseract", 1, configs[name], N)
    assert built.parameter_count == EXPECTED[name]["parameters"]
    assert count_parameters(built.model)["trainable"] == EXPECTED[name]["parameters"]


def test_large_model_is_in_the_research_scale_band(configs):
    count = build_model("tesseract", 1, configs["large"], N).parameter_count
    assert 6_000_000 <= count <= 7_500_000
    assert count > 8 * build_model("tesseract", 1, configs["small"], N).parameter_count


@pytest.mark.parametrize("name", list(EXPECTED))
def test_parameter_breakdown_sums_to_the_total(name, configs):
    """The analytic breakdown must account for every trainable tensor."""
    cfg = configs[name]
    d, f, v, s = cfg.d_model, cfg.d_ff, cfg.vocab_size, cfg.max_seq_len
    parts = {
        "token_embedding": v * d,
        "positional_embedding": s * d,
        "ln1": 2 * d,
        "attention": 4 * (d * d + d),
        "ln2": 2 * d,
        "ffn_w1": d * f + f,
        "ffn_w2": f * d + d,
        "init_z_H": d,
        "init_z_L": d,
        "output_head": d * v + v,
    }
    assert sum(parts.values()) == model_parameter_count(v, d, f, s, 1)
    assert sum(parts.values()) == build_model("tesseract", 1, cfg, N).parameter_count


# ---------------------------------------------------------------------------
# Constant parameters across K — the defining property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED))
def test_parameter_count_is_identical_across_k(name, configs):
    counts = {k: build_model("tesseract", k, configs[name], N).parameter_count for k in K_VALUES}
    assert len(set(counts.values())) == 1, f"{name}: parameter count varies with K: {counts}"
    assert counts[1] == counts[8] == EXPECTED[name]["parameters"]


@pytest.mark.parametrize("name", list(EXPECTED))
def test_parameter_layout_is_identical_across_k(name, configs):
    layouts = {k: [(n, tuple(p.shape)) for n, p in build_model("tesseract", k, configs[name], N).model.named_parameters()]
               for k in K_VALUES}
    assert all(layout == layouts[1] for layout in layouts.values())


# ---------------------------------------------------------------------------
# Shared block integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", K_VALUES)
def test_large_model_has_exactly_one_block_instance(k, configs):
    model = build_model("tesseract", k, configs["large"], N).model
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    assert len(blocks) == 1
    assert len({id(b) for b in blocks}) == 1


@pytest.mark.parametrize("k", K_VALUES)
def test_large_model_executes_the_shared_block_k_times(k, configs):
    built = build_model("tesseract", k, configs["large"], N)
    calls = []
    handles = [m.register_forward_hook(lambda *_: calls.append(1))
               for m in built.model.modules() if isinstance(m, TransformerBlock)]
    try:
        built.model(torch.randint(0, VOCAB, (2, N)))
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == k == built.block_executions


@pytest.mark.parametrize("k", K_VALUES)
def test_no_duplicated_or_cloned_parameter_tensors(k, configs):
    """Every parameter tensor appears once: nothing is cloned per recursion step."""
    model = build_model("tesseract", k, configs["large"], N).model
    ids = [id(p) for p in model.parameters()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("k", K_VALUES)
def test_block_weights_are_the_same_objects_the_reasoner_owns(k, configs):
    model = build_model("tesseract", k, configs["large"], N).model
    block_ids = {id(p) for p in model.reasoner.shared_block.parameters()}
    reasoner_ids = {id(p) for p in model.reasoner.parameters()}
    assert block_ids <= reasoner_ids
    # The reasoner owns the block plus exactly the two learnable initial states.
    assert len(reasoner_ids - block_ids) == 2


def test_same_seed_gives_identical_weights_at_every_k(configs):
    states = {}
    for k in K_VALUES:
        set_seed(0)
        states[k] = build_model("tesseract", k, configs["large"], N).model.state_dict()
    reference = states[1]
    for k, state in states.items():
        assert set(state) == set(reference)
        assert all(torch.equal(state[key], reference[key]) for key in reference), f"K={k} differs"


# ---------------------------------------------------------------------------
# BPTT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", K_VALUES)
def test_bptt_spans_every_recursion_step_at_large_scale(k, configs):
    model = build_model("tesseract", k, configs["large"], N).model
    tokens = torch.randint(0, VOCAB, (2, N))
    report = verify_bptt(model, tokens, tokens)
    assert report.passed, report.failures
    assert [s.step for s in report.steps] == list(range(1, k + 1))
    assert all(s.grad_z_L_norm and s.grad_z_L_norm > 0 for s in report.steps)


def test_large_model_takes_a_finite_training_step(configs):
    """One optimizer step, to prove the model is trainable. Not an experiment."""
    model = build_model("tesseract", 2, configs["large"], N).model
    tokens = torch.randint(0, VOCAB, (2, N))
    metrics = train_step(model, tokens, tokens, build_optimizer(model, OptimizerConfig("adamw", 1e-3, 0.01)))
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert torch.isfinite(torch.tensor(metrics["gradient_norm"]))


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", K_VALUES)
def test_a_checkpoint_from_any_k_loads_into_every_other_k(k, configs):
    state = build_model("tesseract", k, configs["large"], N).model.state_dict()
    for other in K_VALUES:
        build_model("tesseract", other, configs["large"], N).model.load_state_dict(state, strict=True)


def test_checkpoint_round_trips_through_torch_save(tmp_path, configs):
    built = build_model("tesseract", 4, configs["large"], N)
    path = tmp_path / "large.pt"
    torch.save(built.model.state_dict(), path)
    reloaded = build_model("tesseract", 4, configs["large"], N).model
    reloaded.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    built.model.eval()
    reloaded.eval()
    tokens = torch.randint(0, VOCAB, (2, N))
    with torch.no_grad():
        assert torch.equal(built.model(tokens)[0], reloaded(tokens)[0])


def test_a_large_checkpoint_does_not_load_into_a_smaller_width(configs):
    state = build_model("tesseract", 1, configs["large"], N).model.state_dict()
    with pytest.raises(RuntimeError):
        build_model("tesseract", 1, configs["small"], N).model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Baseline families at large scale (existing matching rules, unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_large_param_matched_matches_tesseract_parameters(depth, configs):
    cfg = configs["large"]
    built = build_model("param_matched", depth, cfg, N)
    target = build_model("tesseract", 1, cfg, N).parameter_count
    assert abs(built.parameter_count - target) / target < 0.01
    assert built.matched["target_parameter_count"] == target
    assert "head dimension" in built.matched["known_confound"]


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_large_width_scaled_matches_tesseract_compute(depth, configs):
    cfg = configs["large"]
    built = build_model("width_scaled", depth, cfg, N)
    target = forward_flops(cfg.d_model, cfg.d_ff, VOCAB, N, depth)
    assert built.block_executions == 1
    assert abs(built.forward_flops_per_sequence - target) / target < 0.001


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_large_unrolled_has_l_distinct_blocks(depth, configs):
    built = build_model("unrolled", depth, configs["large"], N)
    blocks = [m for m in built.model.modules() if isinstance(m, TransformerBlock)]
    assert len(blocks) == depth
    assert len({id(b) for b in blocks}) == depth
    assert built.forward_flops_per_sequence == build_model("tesseract", depth, configs["large"], N).forward_flops_per_sequence


def test_large_unrolled_parameters_grow_with_depth_unlike_tesseract(configs):
    cfg = configs["large"]
    unrolled = {d: build_model("unrolled", d, cfg, N).parameter_count for d in K_VALUES}
    tesseract = {d: build_model("tesseract", d, cfg, N).parameter_count for d in K_VALUES}
    assert len(set(tesseract.values())) == 1
    assert sorted(unrolled.values()) == list(unrolled.values()) and unrolled[8] > unrolled[1]


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def test_large_flops_scale_linearly_with_k(configs):
    cfg = configs["large"]
    head = 2 * N * cfg.d_model * VOCAB
    one = forward_flops(cfg.d_model, cfg.d_ff, VOCAB, N, 1) - head
    for k in K_VALUES:
        assert forward_flops(cfg.d_model, cfg.d_ff, VOCAB, N, k) - head == k * one


def test_large_costs_more_compute_per_block_than_medium(configs):
    head = lambda c: 2 * N * c.d_model * VOCAB  # noqa: E731
    large, medium = configs["large"], configs["medium"]
    per_block = lambda c: forward_flops(c.d_model, c.d_ff, VOCAB, N, 1) - head(c)  # noqa: E731
    assert per_block(large) > per_block(medium)


# ---------------------------------------------------------------------------
# Nothing defaults to large
# ---------------------------------------------------------------------------


def test_phase2_experiment_does_not_default_to_any_width():
    from phase2.config import PENDING_P1B, load_phase2_experiment

    config = load_phase2_experiment()
    assert config.model_base is None
    assert config.max_steps is None
    assert config.pending == ["model_base", "max_steps"]
    raw = (__import__("pathlib").Path("configs/phase2/phase2_experiment.yaml")).read_text(encoding="utf-8")
    assert f"model_base: {PENDING_P1B}" in raw
    for family in MODEL_FAMILIES:
        assert f"model_base: {family.base}" not in raw


def test_capacity_diagnostic_still_uses_only_small_and_medium():
    """P1b is running with two widths; adding a third must not change it."""
    from phase2.config import CapacityDiagnosticConfig
    from utils.config import load_config

    config = load_config("phase2/capacity_diagnostic_p1b", CapacityDiagnosticConfig)
    assert [m.name for m in config.models] == ["small", "medium"]
    assert config.max_steps == 20000 and config.t == 8


def test_model_config_validation_still_rejects_bad_shapes(configs):
    with pytest.raises(ConfigError):
        replace(configs["large"], num_heads=7)  # 768 % 7 != 0
