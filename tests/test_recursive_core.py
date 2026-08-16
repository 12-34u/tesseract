"""Comprehensive unit tests for Day 3: RecursiveReasoner, TesseractModel, Embeddings, and CopyDataset."""

import pytest
import torch
from torch.utils.data import DataLoader

from models.embeddings import TokenPositionalEmbedding
from models.recursive_core import RecursiveReasoner
from models.tesseract import TesseractModel
from data.toy_copy import CopyDataset
from utils.param_count import count_parameters


# ============================================================================
# Embedding Tests
# ============================================================================

class TestTokenPositionalEmbedding:
    """Tests for the TokenPositionalEmbedding module."""

    def test_output_shape(self):
        """Verify embedding output shape is [B, N, D]."""
        emb = TokenPositionalEmbedding(vocab_size=16, d_model=128, max_seq_len=64)
        tokens = torch.randint(0, 16, (2, 8))
        x_emb = emb(tokens)
        assert x_emb.shape == (2, 8, 128), f"Expected (2, 8, 128), got {x_emb.shape}"

    def test_various_batch_and_seq(self):
        """Test across multiple batch sizes and sequence lengths."""
        emb = TokenPositionalEmbedding(vocab_size=16, d_model=64, max_seq_len=32)
        for B in [1, 2, 4]:
            for N in [4, 8, 16]:
                tokens = torch.randint(0, 16, (B, N))
                x_emb = emb(tokens)
                assert x_emb.shape == (B, N, 64)

    def test_sequence_length_validation(self):
        """Verify ValueError when sequence length exceeds max_seq_len."""
        emb = TokenPositionalEmbedding(vocab_size=16, d_model=128, max_seq_len=8)
        tokens = torch.randint(0, 16, (2, 16))  # N=16 > max_seq_len=8
        with pytest.raises(ValueError, match="exceeds maximum"):
            emb(tokens)

    def test_invalid_input_dimensions(self):
        """Verify ValueError for non-2D input."""
        emb = TokenPositionalEmbedding(vocab_size=16, d_model=128, max_seq_len=64)
        with pytest.raises(ValueError):
            emb(torch.randint(0, 16, (8,)))  # 1D
        with pytest.raises(ValueError):
            emb(torch.randint(0, 16, (2, 8, 3)))  # 3D

    def test_backward_pass(self):
        """Verify gradients flow through embeddings."""
        emb = TokenPositionalEmbedding(vocab_size=16, d_model=128, max_seq_len=64)
        tokens = torch.randint(0, 16, (2, 8))
        x_emb = emb(tokens)
        loss = x_emb.mean()
        loss.backward()

        for name, param in emb.named_parameters():
            # Only the accessed embeddings will have non-None grad
            assert param.grad is not None, f"Gradient for {name} is None"


# ============================================================================
# RecursiveReasoner Tests
# ============================================================================

class TestRecursiveReasoner:
    """Tests for the RecursiveReasoner module."""

    @pytest.fixture
    def default_config(self):
        return dict(d_model=128, num_heads=4, d_ff=512, dropout=0.0)

    def test_state_shapes(self, default_config):
        """Test 1 — Verify z_H and z_L shapes are [B, N, D] at every step."""
        B, N, D = 2, 8, 128
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=4)
        x_emb = torch.randn(B, N, D)

        z_L_final, state_history = reasoner(x_emb, return_states=True)

        assert z_L_final.shape == (B, N, D), f"z_L final shape {z_L_final.shape}"

        assert len(state_history) == 4, f"Expected 4 steps, got {len(state_history)}"
        for k, states in enumerate(state_history):
            assert states["z_H"].shape == (B, N, D), \
                f"z_H shape at step {k+1}: {states['z_H'].shape}"
            assert states["z_L"].shape == (B, N, D), \
                f"z_L shape at step {k+1}: {states['z_L'].shape}"

    def test_k_equals_1(self, default_config):
        """Test 2 — Verify K=1 runs correctly."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=1)
        x_emb = torch.randn(2, 8, 128)

        z_L_final, state_history = reasoner(x_emb, return_states=True)

        assert z_L_final.shape == (2, 8, 128)
        assert len(state_history) == 1
        assert torch.isfinite(z_L_final).all()

    @pytest.mark.parametrize("k", [2, 4, 8])
    def test_k_various(self, default_config, k):
        """Test 3 — K=2,4,8 produce no shape errors, NaNs, or infinities."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=k)
        x_emb = torch.randn(2, 8, 128)

        z_L_final, state_history = reasoner(x_emb, return_states=True)

        assert z_L_final.shape == (2, 8, 128)
        assert torch.isfinite(z_L_final).all(), f"z_L contains NaN/Inf at K={k}"

        for step, states in enumerate(state_history):
            assert torch.isfinite(states["z_H"]).all(), \
                f"z_H contains NaN/Inf at step {step+1} (K={k})"
            assert torch.isfinite(states["z_L"]).all(), \
                f"z_L contains NaN/Inf at step {step+1} (K={k})"

    def test_state_changes_across_steps(self, default_config):
        """Test 4 — Verify states change meaningfully across recursive steps."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=4)
        x_emb = torch.randn(2, 8, 128)

        _, state_history = reasoner(x_emb, return_states=True)

        # Check z_L changes between consecutive steps
        for k in range(1, len(state_history)):
            diff_L = torch.norm(state_history[k]["z_L"] - state_history[k - 1]["z_L"])
            assert diff_L.item() > 1e-6, \
                f"z_L did not change meaningfully between step {k} and {k+1}: norm diff = {diff_L.item()}"

            diff_H = torch.norm(state_history[k]["z_H"] - state_history[k - 1]["z_H"])
            assert diff_H.item() > 1e-6, \
                f"z_H did not change meaningfully between step {k} and {k+1}: norm diff = {diff_H.item()}"

    def test_z_H_and_z_L_are_distinct(self, default_config):
        """Test — z_H and z_L are not aliases and have different values."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=4)
        x_emb = torch.randn(2, 8, 128)

        _, state_history = reasoner(x_emb, return_states=True)

        for k, states in enumerate(state_history):
            # Check they are not the same tensor (different data pointers)
            assert states["z_H"].data_ptr() != states["z_L"].data_ptr(), \
                f"z_H and z_L share the same tensor at step {k+1}"

            # Check their values are not trivially identical
            diff = torch.norm(states["z_H"] - states["z_L"])
            assert diff.item() > 1e-6, \
                f"z_H and z_L have identical values at step {k+1}: norm diff = {diff.item()}"

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_weight_sharing_parameter_count_independent_of_k(self, default_config, k):
        """Test — Parameter count is identical regardless of K."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=k)
        param_count = count_parameters(reasoner)["trainable"]

        # All K values should yield the same count
        reasoner_k1 = RecursiveReasoner(**default_config, num_recursive_steps=1)
        param_count_k1 = count_parameters(reasoner_k1)["trainable"]

        assert param_count == param_count_k1, \
            f"K={k} has {param_count} params, K=1 has {param_count_k1} params — weight sharing violated!"

    def test_single_block_instance(self, default_config):
        """Test — Only one TransformerBlock exists in the RecursiveReasoner."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=8)

        # Count TransformerBlock instances as direct children
        from models.block import TransformerBlock
        block_modules = [m for m in reasoner.modules() if isinstance(m, TransformerBlock)]

        assert len(block_modules) == 1, \
            f"Expected 1 TransformerBlock instance, found {len(block_modules)}"

    def test_no_state_return_when_disabled(self, default_config):
        """Test — return_states=False returns None for state_history."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=4)
        x_emb = torch.randn(2, 8, 128)

        z_L_final, state_history = reasoner(x_emb, return_states=False)

        assert z_L_final.shape == (2, 8, 128)
        assert state_history is None

    def test_backward_through_recursive_steps(self, default_config):
        """Test — Full BPTT: gradients flow through all K recursive steps."""
        reasoner = RecursiveReasoner(**default_config, num_recursive_steps=4)
        x_emb = torch.randn(2, 8, 128, requires_grad=True)

        z_L_final, _ = reasoner(x_emb, return_states=False)
        loss = z_L_final.mean()
        loss.backward()

        # Input gradients
        assert x_emb.grad is not None, "Input gradient is None"
        assert torch.isfinite(x_emb.grad).all(), "Input gradient has NaN/Inf"

        # Shared block gradients
        for name, param in reasoner.shared_block.named_parameters():
            assert param.grad is not None, f"shared_block.{name} gradient is None"
            assert torch.isfinite(param.grad).all(), \
                f"shared_block.{name} gradient has NaN/Inf"
            assert (param.grad != 0).any(), \
                f"shared_block.{name} gradient is all zeroes"

        # Learnable init state gradients
        assert reasoner.learnable_init_H.grad is not None, "init_H gradient is None"
        assert reasoner.learnable_init_L.grad is not None, "init_L gradient is None"


# ============================================================================
# TesseractModel Tests
# ============================================================================

class TestTesseractModel:
    """Tests for the full TesseractModel."""

    @pytest.fixture
    def model_config(self):
        return dict(
            vocab_size=16,
            d_model=128,
            num_heads=4,
            d_ff=512,
            max_seq_len=64,
            num_recursive_steps=4,
            alpha=0.9,
            dropout=0.0,
        )

    def test_forward_logit_shape(self, model_config):
        """Test — tokens [B,N] → logits [B,N,V]."""
        model = TesseractModel(**model_config)
        tokens = torch.randint(0, 16, (2, 8))

        logits, _ = model(tokens)

        assert logits.shape == (2, 8, 16), f"Expected (2, 8, 16), got {logits.shape}"

    def test_forward_with_state_history(self, model_config):
        """Test — Forward with return_states=True returns state history."""
        model = TesseractModel(**model_config)
        tokens = torch.randint(0, 16, (2, 8))

        logits, state_history = model(tokens, return_states=True)

        assert logits.shape == (2, 8, 16)
        assert state_history is not None
        assert len(state_history) == 4

    def test_forward_plus_loss_and_backward(self, model_config):
        """Test 20 — Full forward + loss + backward sanity check."""
        model = TesseractModel(**model_config)

        # Random inputs and targets
        tokens = torch.randint(0, 16, (2, 8))
        targets = torch.randint(0, 16, (2, 8))

        # Forward
        logits, _ = model(tokens)

        # Loss
        loss = model.compute_loss(logits, targets)

        # Verify loss is finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        # Backward
        loss.backward()

        # Verify gradients exist and are finite for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} is None"
                assert torch.isfinite(param.grad).all(), \
                    f"Gradient for {name} contains NaN/Inf"

        # Verify at least some gradients are non-zero
        nonzero_grads = sum(
            1 for p in model.parameters()
            if p.requires_grad and p.grad is not None and (p.grad != 0).any()
        )
        assert nonzero_grads > 0, "All gradients are zero — computation graph may be broken"

    def test_output_no_nans(self, model_config):
        """Test — Forward pass produces no NaNs or Infs in logits."""
        model = TesseractModel(**model_config)
        tokens = torch.randint(0, 16, (2, 8))

        logits, _ = model(tokens)

        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_parameter_count_constant_across_k(self, k):
        """Test 18 — Parameter count is identical for K=1,2,4,8."""
        config = dict(
            vocab_size=16, d_model=128, num_heads=4, d_ff=512,
            max_seq_len=64, num_recursive_steps=k, alpha=0.9, dropout=0.0,
        )
        model = TesseractModel(**config)
        param_count = count_parameters(model)["trainable"]

        # Reference: K=1
        ref_config = dict(config, num_recursive_steps=1)
        ref_model = TesseractModel(**ref_config)
        ref_count = count_parameters(ref_model)["trainable"]

        assert param_count == ref_count, \
            f"K={k}: {param_count} params != K=1: {ref_count} params"


# ============================================================================
# CopyDataset Tests
# ============================================================================

class TestCopyDataset:
    """Tests for the toy copy/reverse dataset."""

    def test_copy_task_shapes(self):
        """Test — Dataset returns correct shapes."""
        ds = CopyDataset(num_examples=32, seq_len=8, vocab_size=16, task="copy")
        assert len(ds) == 32

        inp, tgt = ds[0]
        assert inp.shape == (8,)
        assert tgt.shape == (8,)

    def test_copy_task_identity(self):
        """Test — Copy task: input == target."""
        ds = CopyDataset(num_examples=16, seq_len=8, vocab_size=16, task="copy")
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert torch.equal(inp, tgt), f"Copy mismatch at index {i}"

    def test_reverse_task(self):
        """Test — Reverse task: target is reversed input."""
        ds = CopyDataset(num_examples=16, seq_len=8, vocab_size=16, task="reverse")
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert torch.equal(tgt, inp.flip(0)), f"Reverse mismatch at index {i}"

    def test_deterministic_with_seed(self):
        """Test — Same seed produces identical data."""
        ds1 = CopyDataset(num_examples=16, seq_len=8, vocab_size=16, seed=42)
        ds2 = CopyDataset(num_examples=16, seq_len=8, vocab_size=16, seed=42)
        for i in range(len(ds1)):
            inp1, _ = ds1[i]
            inp2, _ = ds2[i]
            assert torch.equal(inp1, inp2), f"Non-deterministic at index {i}"

    def test_different_seed_different_data(self):
        """Test — Different seeds produce different data."""
        ds1 = CopyDataset(num_examples=16, seq_len=8, vocab_size=16, seed=42)
        ds2 = CopyDataset(num_examples=16, seq_len=8, vocab_size=16, seed=99)
        any_different = False
        for i in range(len(ds1)):
            inp1, _ = ds1[i]
            inp2, _ = ds2[i]
            if not torch.equal(inp1, inp2):
                any_different = True
                break
        assert any_different, "Different seeds produced identical data"

    def test_vocab_range(self):
        """Test — All tokens are within [0, vocab_size)."""
        ds = CopyDataset(num_examples=64, seq_len=16, vocab_size=16, seed=42)
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert (inp >= 0).all() and (inp < 16).all(), f"Input out of range at {i}"
            assert (tgt >= 0).all() and (tgt < 16).all(), f"Target out of range at {i}"

    def test_dataloader_integration(self):
        """Test — Dataset works with PyTorch DataLoader."""
        ds = CopyDataset(num_examples=32, seq_len=8, vocab_size=16, task="copy")
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        batch = next(iter(loader))
        inputs, targets = batch

        assert inputs.shape == (8, 8)
        assert targets.shape == (8, 8)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long

    def test_invalid_task(self):
        """Test — Invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported task"):
            CopyDataset(task="invalid")
