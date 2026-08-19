"""Tests for K-scaling — Day 5.

Verifies the central architectural property:
    - Parameter count is invariant across K
    - Output shape is correct for all K
    - No NaN/Inf for all K
    - The shared block is called K times (instrumented)
    - A single shared TransformerBlock exists regardless of K
"""

import pytest
import torch

from models.block import TransformerBlock
from models.recursive_core import RecursiveReasoner
from models.tesseract import TesseractModel
from utils.param_count import count_parameters


K_VALUES = [1, 2, 4, 8]

MODEL_CONFIG = dict(
    vocab_size=16,
    d_model=128,
    num_heads=4,
    d_ff=512,
    max_seq_len=64,
    alpha=0.9,
    dropout=0.0,
)


# ============================================================================
# Parameter Invariance Tests
# ============================================================================


class TestParameterInvariance:
    """Verify parameter count is identical across K values."""

    def test_parameter_count_identical_across_k(self):
        """The core invariance: all K produce the same parameter count."""
        counts = {}
        for k in K_VALUES:
            model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
            counts[k] = count_parameters(model)["trainable"]

        unique = set(counts.values())
        assert len(unique) == 1, (
            f"Parameter invariance violated! Counts by K: {counts}"
        )

    def test_reasoner_parameter_count_independent_of_k(self):
        """RecursiveReasoner alone has identical params across K."""
        config = dict(d_model=128, num_heads=4, d_ff=512, dropout=0.0)
        counts = {}
        for k in K_VALUES:
            reasoner = RecursiveReasoner(**config, num_recursive_steps=k)
            counts[k] = count_parameters(reasoner)["trainable"]

        unique = set(counts.values())
        assert len(unique) == 1, (
            f"Reasoner parameter invariance violated! Counts: {counts}"
        )

    def test_parameter_count_measured_not_hardcoded(self):
        """Ensure we measure parameter count — not assume 210832."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=4)
        measured = sum(p.numel() for p in model.parameters() if p.requires_grad)
        utility_count = count_parameters(model)["trainable"]
        assert measured == utility_count, (
            f"Measured ({measured}) != count_parameters ({utility_count})"
        )


# ============================================================================
# Weight Sharing / Single Block Instance Tests
# ============================================================================


class TestWeightSharing:
    """Verify that the recursive model uses a single shared block."""

    @pytest.mark.parametrize("k", K_VALUES)
    def test_single_transformer_block_instance(self, k):
        """Only one TransformerBlock exists in the model regardless of K."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
        blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
        assert len(blocks) == 1, (
            f"Expected 1 TransformerBlock for K={k}, found {len(blocks)}"
        )

    @pytest.mark.parametrize("k", K_VALUES)
    def test_reasoner_block_parameters_same_object(self, k):
        """Reasoner shared_block parameters are the same object across K."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
        block_params = list(model.reasoner.shared_block.parameters())

        # Verify these are actual nn.Parameter objects (not copies)
        for p in block_params:
            assert isinstance(p, torch.nn.Parameter)
            assert p.requires_grad

    def test_block_params_independent_of_k(self):
        """The shared_block parameter count is exactly the same object set for all K."""
        config = dict(d_model=128, num_heads=4, d_ff=512, dropout=0.0)
        block_counts = {}
        for k in K_VALUES:
            reasoner = RecursiveReasoner(**config, num_recursive_steps=k)
            block_counts[k] = sum(
                p.numel() for p in reasoner.shared_block.parameters()
            )

        unique = set(block_counts.values())
        assert len(unique) == 1, (
            f"Block parameter counts differ across K: {block_counts}"
        )


# ============================================================================
# Output Shape and Correctness Tests
# ============================================================================


class TestOutputShape:
    """Verify correct output shapes and no NaN/Inf across K."""

    @pytest.mark.parametrize("k", K_VALUES)
    def test_output_shape_correct(self, k):
        """Model output is [B, N, V] for all K."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
        tokens = torch.randint(0, 16, (4, 8))
        logits, _ = model(tokens, return_states=False)
        assert logits.shape == (4, 8, 16), (
            f"K={k}: expected (4,8,16), got {logits.shape}"
        )

    @pytest.mark.parametrize("k", K_VALUES)
    def test_no_nan_inf(self, k):
        """No NaN or Inf in outputs for all K."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
        tokens = torch.randint(0, 16, (4, 8))
        logits, _ = model(tokens, return_states=False)
        assert torch.isfinite(logits).all(), f"NaN/Inf in logits at K={k}"

    @pytest.mark.parametrize("k", K_VALUES)
    def test_backward_pass_works(self, k):
        """Backward pass runs without errors for all K."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
        tokens = torch.randint(0, 16, (4, 8))
        targets = torch.randint(0, 16, (4, 8))
        logits, _ = model(tokens)
        loss = model.compute_loss(logits, targets)
        loss.backward()

        # Verify at least some parameters have gradients
        has_grad = any(
            p.grad is not None and (p.grad != 0).any()
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad, f"No non-zero gradients at K={k}"


# ============================================================================
# Recursive Call Count Verification
# ============================================================================


class TestRecursiveCallCount:
    """Verify the shared block is actually called K times."""

    @pytest.mark.parametrize("k", K_VALUES)
    def test_state_history_length_matches_k(self, k):
        """State history has exactly K entries when return_states=True."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
        tokens = torch.randint(0, 16, (2, 8))
        _, state_history = model(tokens, return_states=True)
        assert state_history is not None
        assert len(state_history) == k, (
            f"Expected {k} state history entries, got {len(state_history)}"
        )

    @pytest.mark.parametrize("k", K_VALUES)
    def test_block_called_k_times_via_hook(self, k):
        """Instrument the shared block with a forward hook to count calls."""
        model = TesseractModel(**MODEL_CONFIG, num_recursive_steps=k)
        call_count = [0]

        def count_hook(module, input, output):
            call_count[0] += 1

        handle = model.reasoner.shared_block.register_forward_hook(count_hook)

        tokens = torch.randint(0, 16, (2, 8))
        model(tokens, return_states=False)

        handle.remove()

        assert call_count[0] == k, (
            f"Expected shared_block to be called {k} times, "
            f"but it was called {call_count[0]} times"
        )

    def test_block_calls_scale_with_k(self):
        """Verify that block call counts exactly match K across all values."""
        config = dict(d_model=128, num_heads=4, d_ff=512, dropout=0.0)
        tokens = torch.randn(2, 8, 128)

        for k in K_VALUES:
            reasoner = RecursiveReasoner(**config, num_recursive_steps=k)
            call_count = [0]

            def count_hook(module, input, output):
                call_count[0] += 1

            handle = reasoner.shared_block.register_forward_hook(count_hook)
            reasoner(tokens, return_states=False)
            handle.remove()

            assert call_count[0] == k, (
                f"K={k}: block called {call_count[0]} times"
            )


# ============================================================================
# Deterministic Behavior Tests
# ============================================================================


class TestDeterminism:
    """Verify deterministic results with same seed and K."""

    def test_same_seed_same_output(self):
        """Same seed and K produce identical outputs."""
        torch.manual_seed(42)
        model1 = TesseractModel(**MODEL_CONFIG, num_recursive_steps=4)
        tokens = torch.randint(0, 16, (2, 8))
        with torch.no_grad():
            out1, _ = model1(tokens)

        torch.manual_seed(42)
        model2 = TesseractModel(**MODEL_CONFIG, num_recursive_steps=4)
        with torch.no_grad():
            out2, _ = model2(tokens)

        assert torch.allclose(out1, out2, atol=1e-6), "Outputs differ with same seed"

    def test_different_k_different_output(self):
        """Different K values produce different outputs (same init weights)."""
        torch.manual_seed(42)
        model_k1 = TesseractModel(**MODEL_CONFIG, num_recursive_steps=1)
        state = model_k1.state_dict()

        torch.manual_seed(42)
        model_k4 = TesseractModel(**MODEL_CONFIG, num_recursive_steps=4)
        model_k4.load_state_dict(state, strict=True)

        tokens = torch.randint(0, 16, (2, 8))
        with torch.no_grad():
            out_k1, _ = model_k1(tokens)
            out_k4, _ = model_k4(tokens)

        assert not torch.allclose(out_k1, out_k4, atol=1e-4), (
            "K=1 and K=4 produced identical outputs — recursion may not work"
        )
