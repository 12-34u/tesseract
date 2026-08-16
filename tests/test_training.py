"""Unit tests for Day 4: Training loop, gradient instrumentation, and BPTT verification."""

import pytest
import torch

from models.tesseract import TesseractModel
from data.toy_copy import CopyDataset
from training.trainer import (
    train_step,
    verify_parameter_update,
    compute_token_accuracy,
    compute_exact_match_accuracy,
    retain_state_gradients,
    collect_state_gradient_norms,
    collect_state_norms,
)
from utils.param_count import count_parameters
from utils.seed import set_seed


@pytest.fixture
def model_and_data():
    """Create a small TesseractModel and matching dataset for testing."""
    set_seed(42)
    model = TesseractModel(
        vocab_size=16,
        d_model=128,
        num_heads=4,
        d_ff=512,
        max_seq_len=64,
        num_recursive_steps=4,
        alpha=0.9,
        dropout=0.0,
    )
    dataset = CopyDataset(
        num_examples=16,
        seq_len=16,
        vocab_size=16,
        task="copy",
        seed=42,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, dataset, optimizer


class TestTrainingStep:
    """Tests for the core training step."""

    def test_single_step_executes(self, model_and_data):
        """A single training step executes without errors."""
        model, dataset, optimizer = model_and_data
        inputs = dataset.inputs
        targets = dataset.targets

        metrics = train_step(model, inputs, targets, optimizer, instrument_gradients=True)

        assert "loss" in metrics
        assert "token_accuracy" in metrics
        assert "exact_match_accuracy" in metrics
        assert "gradient_norm" in metrics

    def test_loss_is_finite(self, model_and_data):
        """Loss is finite after a training step."""
        model, dataset, optimizer = model_and_data
        metrics = train_step(model, dataset.inputs, dataset.targets, optimizer)

        assert torch.isfinite(torch.tensor(metrics["loss"])), \
            f"Loss is not finite: {metrics['loss']}"

    def test_gradients_exist_after_step(self, model_and_data):
        """All model parameters have gradients after a training step."""
        model, dataset, optimizer = model_and_data

        # Zero grads and do a step
        optimizer.zero_grad()
        logits, _ = model(dataset.inputs)
        loss = model.compute_loss(logits, dataset.targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} is None"
                assert torch.isfinite(param.grad).all(), \
                    f"Gradient for {name} has NaN/Inf"

    def test_parameters_update_after_step(self, model_and_data):
        """Parameters change after optimizer.step()."""
        model, dataset, optimizer = model_and_data

        changed, delta = verify_parameter_update(
            model, dataset.inputs, dataset.targets, optimizer,
        )

        assert changed, "No parameters changed after optimizer step"
        assert delta > 0, f"Total delta norm is zero: {delta}"

    def test_gradient_instrumentation_works(self, model_and_data):
        """Recursive state gradient norms are captured correctly."""
        model, dataset, optimizer = model_and_data

        metrics = train_step(
            model, dataset.inputs, dataset.targets, optimizer,
            instrument_gradients=True,
        )

        # Should have gradient norms for K=4 steps
        for k in range(1, 5):
            key = f"grad_zL_step_{k}"
            assert key in metrics, f"Missing gradient key: {key}"
            assert metrics[key] >= 0, f"Negative gradient norm for {key}"

    def test_recursive_gradients_are_nonzero(self, model_and_data):
        """BPTT: Gradient norms at each recursive step are non-zero."""
        model, dataset, optimizer = model_and_data

        metrics = train_step(
            model, dataset.inputs, dataset.targets, optimizer,
            instrument_gradients=True,
        )

        for k in range(1, 5):
            zL_grad = metrics.get(f"grad_zL_step_{k}", 0.0)
            assert zL_grad > 0, \
                f"z_L gradient at step {k} is zero — BPTT may be broken"

    def test_recursive_gradients_are_finite(self, model_and_data):
        """All recursive state gradients are finite (no NaN/Inf)."""
        model, dataset, optimizer = model_and_data

        metrics = train_step(
            model, dataset.inputs, dataset.targets, optimizer,
            instrument_gradients=True,
        )

        for k in range(1, 5):
            zL_grad = metrics.get(f"grad_zL_step_{k}", 0.0)
            zH_grad = metrics.get(f"grad_zH_step_{k}", 0.0)
            assert torch.isfinite(torch.tensor(zL_grad)), \
                f"z_L gradient at step {k} is not finite: {zL_grad}"
            assert torch.isfinite(torch.tensor(zH_grad)), \
                f"z_H gradient at step {k} is not finite: {zH_grad}"


class TestAccuracyMetrics:
    """Tests for token and exact-match accuracy computation."""

    def test_perfect_token_accuracy(self):
        """100% token accuracy when predictions match targets."""
        logits = torch.zeros(2, 4, 8)
        targets = torch.zeros(2, 4, dtype=torch.long)
        # Set logits so argmax matches targets (all zeros)
        logits[:, :, 0] = 10.0

        acc = compute_token_accuracy(logits, targets)
        assert acc == 100.0

    def test_zero_token_accuracy(self):
        """0% token accuracy when all predictions are wrong."""
        logits = torch.zeros(2, 4, 8)
        targets = torch.zeros(2, 4, dtype=torch.long)
        # Set logits so argmax is always 1, but target is always 0
        logits[:, :, 1] = 10.0

        acc = compute_token_accuracy(logits, targets)
        assert acc == 0.0

    def test_perfect_exact_match(self):
        """100% exact match when all sequences fully match."""
        logits = torch.zeros(2, 4, 8)
        targets = torch.zeros(2, 4, dtype=torch.long)
        logits[:, :, 0] = 10.0

        acc = compute_exact_match_accuracy(logits, targets)
        assert acc == 100.0

    def test_partial_exact_match(self):
        """Exact match = 0% when one token is wrong per sequence."""
        logits = torch.zeros(2, 4, 8)
        targets = torch.zeros(2, 4, dtype=torch.long)
        logits[:, :, 0] = 10.0
        # Break one token in each sequence
        logits[0, 0, 0] = -10.0
        logits[0, 0, 1] = 10.0
        logits[1, 0, 0] = -10.0
        logits[1, 0, 1] = 10.0

        em_acc = compute_exact_match_accuracy(logits, targets)
        assert em_acc == 0.0

        # Token accuracy should still be high (3/4 correct per sequence)
        tok_acc = compute_token_accuracy(logits, targets)
        assert tok_acc == 75.0


class TestBPTTVerification:
    """Tests specifically verifying BPTT through the recursive computation."""

    def test_retain_grad_on_intermediate_states(self):
        """Intermediate z_L/z_H tensors can retain gradients."""
        set_seed(42)
        model = TesseractModel(
            vocab_size=16, d_model=128, num_heads=4, d_ff=512,
            max_seq_len=64, num_recursive_steps=4,
        )
        tokens = torch.randint(0, 16, (2, 8))
        targets = torch.randint(0, 16, (2, 8))

        logits, state_history = model(tokens, return_states=True)
        retain_state_gradients(state_history)

        loss = model.compute_loss(logits, targets)
        loss.backward()

        # All intermediate z_L should have gradients
        for k, states in enumerate(state_history, 1):
            assert states["z_L"].grad is not None, \
                f"z_L.grad is None at step {k}"
            assert torch.isfinite(states["z_L"].grad).all(), \
                f"z_L.grad has NaN/Inf at step {k}"

    def test_gradient_chain_across_steps(self):
        """Gradients reach all K recursive steps (BPTT chain verified)."""
        set_seed(42)
        model = TesseractModel(
            vocab_size=16, d_model=128, num_heads=4, d_ff=512,
            max_seq_len=64, num_recursive_steps=4,
        )
        tokens = torch.randint(0, 16, (2, 8))
        targets = torch.randint(0, 16, (2, 8))

        logits, state_history = model(tokens, return_states=True)
        retain_state_gradients(state_history)

        loss = model.compute_loss(logits, targets)
        loss.backward()

        grad_norms = collect_state_gradient_norms(state_history)

        # Every z_L gradient should be non-zero
        for k in range(1, 5):
            key = f"grad_zL_step_{k}"
            assert grad_norms[key] > 0, \
                f"Gradient at step {k} is zero — BPTT chain is broken!"

        # Verify gradients are finite
        for key, val in grad_norms.items():
            assert torch.isfinite(torch.tensor(val)), \
                f"Gradient {key} is not finite: {val}"
