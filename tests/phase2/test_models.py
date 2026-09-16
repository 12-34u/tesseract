"""Phase 2 model families: unrolled equivalence, no sharing, parameter/compute matching."""

import torch
import pytest

from models.tesseract import TesseractModel
from phase2.models import (
    UnrolledTransformer,
    build_model,
    forward_flops,
    match_parameters,
    model_parameter_count,
    width_scaled_d_ff,
)
from training.trainer import build_optimizer, train_step, verify_bptt
from utils.config import OptimizerConfig, load_prototype_config
from utils.param_count import count_parameters


@pytest.fixture(scope="module")
def cfg():
    from dataclasses import replace
    return replace(load_prototype_config().model, vocab_size=60)


def tokens(batch=4, n=17, seed=0):
    return torch.randint(0, 60, (batch, n), generator=torch.Generator().manual_seed(seed))


def test_unrolled_with_one_layer_equals_tesseract_k1(cfg):
    torch.manual_seed(0)
    tesseract = TesseractModel.from_config(cfg, 1).eval()
    unrolled = UnrolledTransformer.from_config(cfg, 1).eval()
    mapped = {k.replace("reasoner.shared_block.", "blocks.0.").replace("reasoner.", ""): v
              for k, v in tesseract.state_dict().items()}
    unrolled.load_state_dict(mapped, strict=True)
    x = tokens()
    with torch.no_grad():
        assert torch.equal(tesseract(x)[0], unrolled(x)[0])


@pytest.mark.parametrize("layers", [1, 2, 4, 8])
def test_unrolled_parameter_count_formula(cfg, layers):
    model = UnrolledTransformer.from_config(cfg, layers)
    expected = model_parameter_count(cfg.vocab_size, cfg.d_model, cfg.d_ff, cfg.max_seq_len, layers)
    assert count_parameters(model)["trainable"] == expected
    assert count_parameters(TesseractModel.from_config(cfg, layers))["trainable"] == model_parameter_count(
        cfg.vocab_size, cfg.d_model, cfg.d_ff, cfg.max_seq_len, 1)


def test_unrolled_blocks_do_not_share_parameters(cfg):
    model = UnrolledTransformer.from_config(cfg, 4)
    ids = [{id(p) for p in block.parameters()} for block in model.blocks]
    assert all(ids[i].isdisjoint(ids[j]) for i in range(4) for j in range(i + 1, 4))


def test_unrolled_executes_each_block_once(cfg):
    model = UnrolledTransformer.from_config(cfg, 3)
    calls = []
    for i, block in enumerate(model.blocks):
        block.register_forward_hook(lambda *_, i=i: calls.append(i))
    model(tokens())
    assert calls == [0, 1, 2]


def test_unrolled_bptt_and_training_step(cfg):
    model = UnrolledTransformer.from_config(cfg, 4)
    x = tokens()
    assert verify_bptt(model, x, x).passed
    metrics = train_step(model, x, x, build_optimizer(model, OptimizerConfig("adamw", 1e-3, 0.01)))
    assert torch.isfinite(torch.tensor(metrics["loss"]))


@pytest.mark.parametrize("layers", [1, 2, 4, 8])
def test_param_matched_is_close_to_tesseract(cfg, layers):
    built = build_model("param_matched", layers, cfg, 17)
    target = count_parameters(TesseractModel.from_config(cfg, 1))["trainable"]
    assert abs(built.parameter_count - target) / target < 0.01
    assert built.matched["target_parameter_count"] == target
    if layers == 1:
        assert (built.d_model, built.d_ff) == (cfg.d_model, cfg.d_ff)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_width_scaled_matches_tesseract_compute(cfg, k):
    built = build_model("width_scaled", k, cfg, 17)
    target = forward_flops(cfg.d_model, cfg.d_ff, cfg.vocab_size, 17, k)
    assert built.block_executions == 1
    assert abs(built.forward_flops_per_sequence - target) / target < 0.001
    if k == 1:
        assert built.d_ff == cfg.d_ff


def test_width_scaled_closed_form(cfg):
    # f = (K-1)(2d + n/2) + K·f0  (derived from the FLOP formula)
    assert width_scaled_d_ff(cfg, 8, 17) == round(7 * (2 * 128 + 17 / 2) + 8 * 512)


def test_flops_scale_linearly_with_block_executions(cfg):
    head = 2 * 17 * cfg.d_model * cfg.vocab_size
    one = forward_flops(cfg.d_model, cfg.d_ff, cfg.vocab_size, 17, 1) - head
    assert forward_flops(cfg.d_model, cfg.d_ff, cfg.vocab_size, 17, 8) - head == 8 * one


def test_build_model_reports_and_rejects(cfg):
    built = build_model("tesseract", 4, cfg, 17)
    assert built.block_executions == 4 and built.describe()["parameter_count"] == built.parameter_count
    with pytest.raises(ValueError):
        build_model("transformer_xl", 4, cfg, 17)
    with pytest.raises(ValueError):
        UnrolledTransformer.from_config(cfg, 0)


def test_match_parameters_rejects_impossible_target(cfg):
    with pytest.raises(ValueError):
        match_parameters(cfg, 8, 10)


@pytest.mark.parametrize("family", ["tesseract", "unrolled", "param_matched", "width_scaled"])
@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_declared_block_executions_equal_measured_calls(cfg, family, depth):
    """block_executions is a compute claim; measure it instead of trusting it.

    A forward hook on every TransformerBlock counts real executions, so a
    Tesseract that silently stopped recursing, or a width_scaled model that
    secretly ran more than one block, would fail here.
    """
    from models.block import TransformerBlock

    built = build_model(family, depth, cfg, 17)
    calls = []
    handles = [m.register_forward_hook(lambda *_: calls.append(1))
               for m in built.model.modules() if isinstance(m, TransformerBlock)]
    try:
        built.model(tokens())
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == built.block_executions


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_tesseract_reuses_exactly_one_block_object(cfg, k):
    from models.block import TransformerBlock

    built = build_model("tesseract", k, cfg, 17)
    blocks = [m for m in built.model.modules() if isinstance(m, TransformerBlock)]
    assert len(blocks) == 1
    assert built.parameter_count == build_model("tesseract", 1, cfg, 17).parameter_count


@pytest.mark.parametrize("layers", [2, 4, 8])
def test_param_matched_records_and_keeps_its_head_confound(cfg, layers):
    """The accepted confound must stay recorded, and stay reported honestly.

    Amendment 02 D3 accepts that param_matched shrinks d_model to hit
    Tesseract's parameter budget, which lowers the attention head dimension.
    The same move also narrows the d_model x vocab output head, so the
    baseline differs in read-out capacity too. Both are pinned here so a later
    refactor cannot quietly drop the disclosure or "fix" the configuration.
    """
    tesseract = build_model("tesseract", layers, cfg, 17)
    matched = build_model("param_matched", layers, cfg, 17)

    assert matched.d_model < tesseract.d_model
    assert matched.head_dim < tesseract.head_dim
    assert matched.matched["head_dim"] == matched.head_dim
    assert matched.matched["reference_head_dim"] == tesseract.head_dim
    assert "head dimension" in matched.matched["known_confound"]

    # The output head and embedding shrink with d_model, not just the heads.
    assert matched.d_model * cfg.vocab_size < tesseract.d_model * cfg.vocab_size
