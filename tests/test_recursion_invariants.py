"""Scientific invariants of the recursive core: weight sharing, the update rule,
parameter invariance, BPTT connectivity, argument validation and dropout."""

import pytest
import torch
import torch.nn as nn

from models.block import TransformerBlock
from models.recursive_core import RecursiveReasoner
from models.tesseract import TesseractModel
from training.trainer import verify_bptt
from utils.config import load_prototype_config

K_VALUES = [1, 2, 4, 8]


@pytest.fixture(scope="module")
def prototype():
    return load_prototype_config().model


def build(model_config, k, seed=0):
    torch.manual_seed(seed)
    return TesseractModel.from_config(model_config, k)


def probe(model_config, batch=2, seq_len=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, model_config.vocab_size, (batch, seq_len), generator=generator)
    targets = torch.randint(0, model_config.vocab_size, (batch, seq_len), generator=generator)
    return tokens, targets


# ---------------------------------------------------------------------------
# Weight sharing and the update rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", K_VALUES)
def test_forward_equals_manual_unroll_of_one_block(prototype, k):
    """The model output equals K explicit applications of the SAME block under
    the documented rule m = z_L + z_H + x; z_L = f(m); z_H = a*z_H + (1-a)*z_L."""
    model = build(prototype, k).eval()
    tokens, _ = probe(prototype)
    with torch.no_grad():
        logits, history = model(tokens, return_states=True)

        r = model.reasoner
        x = model.embedding(tokens)
        z_h = r.learnable_init_H.expand_as(x)
        z_l = r.learnable_init_L.expand_as(x)
        for step in range(k):
            z_l = r.shared_block(z_l + z_h + x)
            z_h = r.alpha * z_h + (1 - r.alpha) * z_l
            assert torch.equal(history[step]["z_L"], z_l)
            assert torch.equal(history[step]["z_H"], z_h)
        assert torch.equal(logits, model.output_head(z_l))


def test_k_changes_depth_but_not_the_parameter_set(prototype):
    layouts = {k: [(n, tuple(p.shape)) for n, p in build(prototype, k).named_parameters()] for k in K_VALUES}
    assert all(layout == layouts[1] for layout in layouts.values())

    # A checkpoint from any K loads strictly into every other K.
    state = build(prototype, 8).state_dict()
    for k in K_VALUES:
        build(prototype, k).load_state_dict(state, strict=True)


@pytest.mark.parametrize("k", K_VALUES)
def test_reasoner_parameters_are_one_block_plus_initial_states(prototype, k):
    model = build(prototype, k)
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    assert len(blocks) == 1
    expected = {id(p) for p in blocks[0].parameters()}
    expected |= {id(model.reasoner.learnable_init_H), id(model.reasoner.learnable_init_L)}
    assert {id(p) for p in model.reasoner.parameters()} == expected


# ---------------------------------------------------------------------------
# BPTT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", K_VALUES)
def test_bptt_reaches_every_recursive_state(prototype, k):
    model = build(prototype, k)
    tokens, targets = probe(prototype)
    report = verify_bptt(model, tokens, targets)

    assert report.passed, report.failures
    assert [s.step for s in report.steps] == list(range(1, k + 1))
    for s in report.steps:
        assert s.grad_z_L_norm is not None and s.grad_z_L_norm > 0
    assert report.steps[-1].grad_z_H_norm is None  # z_H^(K) is not consumed downstream
    assert all(p.grad is None for p in model.parameters()), "verify_bptt must clear gradients"


@pytest.mark.parametrize("k", [2, 4, 8])
def test_initial_low_state_receives_gradient(prototype, k):
    """learnable_init_L enters the computation only at step 1; a non-zero,
    finite gradient means the loss reaches step 1 through the recursion."""
    model = build(prototype, k)
    tokens, targets = probe(prototype)
    logits, _ = model(tokens)
    model.compute_loss(logits, targets).backward()
    grad = model.reasoner.learnable_init_L.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


class DetachedReasoner(RecursiveReasoner):
    """Deliberately broken recursion (truncated BPTT) used to prove the check works."""

    def forward(self, x_emb, return_states=False):
        B, N, D = x_emb.shape
        z_h = self.learnable_init_H.expand(B, N, D)
        z_l = self.learnable_init_L.expand(B, N, D)
        history = []
        for _ in range(self.num_recursive_steps):
            z_l = self.shared_block(z_l + z_h + x_emb)
            z_h = self.alpha * z_h + (1 - self.alpha) * z_l
            history.append({"z_H": z_h, "z_L": z_l})
            z_l, z_h = z_l.detach(), z_h.detach()
        return history[-1]["z_L"], history if return_states else None


def test_verify_bptt_detects_detached_recursion(prototype):
    model = build(prototype, 4)
    model.reasoner = DetachedReasoner(prototype.d_model, prototype.num_heads, prototype.d_ff, num_recursive_steps=4)
    tokens, targets = probe(prototype)
    report = verify_bptt(model, tokens, targets)
    assert not report.passed
    assert any("z_L^(1)" in failure for failure in report.failures)


# ---------------------------------------------------------------------------
# Validation and dropout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_k", [0, -1, 2.0, True])
def test_invalid_k_rejected(bad_k):
    with pytest.raises(ValueError, match="num_recursive_steps"):
        RecursiveReasoner(d_model=16, num_heads=2, d_ff=32, num_recursive_steps=bad_k)


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.1])
def test_invalid_alpha_rejected(bad_alpha):
    with pytest.raises(ValueError, match="alpha"):
        RecursiveReasoner(d_model=16, num_heads=2, d_ff=32, alpha=bad_alpha)


def test_compute_loss_rejects_mismatched_targets(prototype):
    model = build(prototype, 1)
    logits = torch.zeros(2, 8, prototype.vocab_size)
    with pytest.raises(ValueError):
        model.compute_loss(logits, torch.zeros(2, 7, dtype=torch.long))


def test_each_dropout_site_applied_once_per_block_call():
    block = TransformerBlock(d_model=16, num_heads=2, d_ff=32, dropout=0.1)
    calls = []
    for module in block.modules():
        if isinstance(module, nn.Dropout):
            module.register_forward_hook(lambda *_: calls.append(1))
    block(torch.randn(2, 4, 16))
    # attention weights, attention residual, FFN hidden, FFN residual
    assert len(calls) == 4


def test_dropout_only_active_in_train_mode():
    torch.manual_seed(0)
    model = TesseractModel(vocab_size=8, d_model=32, num_heads=2, d_ff=64, max_seq_len=16,
                           num_recursive_steps=2, dropout=0.3)
    tokens = torch.randint(0, 8, (4, 8))
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(tokens)[0], model(tokens)[0])
    model.train()
    with torch.no_grad():
        assert not torch.equal(model(tokens)[0], model(tokens)[0])
