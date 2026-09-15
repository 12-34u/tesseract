"""Training step, training loop, parameter updates and gradient instrumentation."""

import pytest
import torch

from data.toy_copy import CopyDataset
from models.tesseract import TesseractModel
from training.trainer import (
    collect_state_gradient_norms,
    retain_state_gradients,
    total_gradient_norm,
    train_full_batch,
    train_step,
    verify_parameter_update,
)
from utils.seed import set_seed


def make_model(k=4, seed=42):
    set_seed(seed)
    return TesseractModel(vocab_size=16, d_model=128, num_heads=4, d_ff=512, max_seq_len=64,
                          num_recursive_steps=k, alpha=0.9, dropout=0.0)


@pytest.fixture
def model_and_data():
    model = make_model()
    dataset = CopyDataset(num_examples=16, seq_len=16, vocab_size=16, task="copy", seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, dataset, optimizer


class TestTrainingStep:
    def test_single_step_returns_finite_metrics(self, model_and_data):
        model, dataset, optimizer = model_and_data
        metrics = train_step(model, dataset.inputs, dataset.targets, optimizer)
        assert {"loss", "token_accuracy", "exact_match_accuracy", "gradient_norm"} <= metrics.keys()
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["gradient_norm"] > 0

    def test_gradients_exist_for_all_parameters(self, model_and_data):
        model, dataset, _ = model_and_data
        logits, _ = model(dataset.inputs)
        model.compute_loss(logits, dataset.targets).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} has NaN/Inf"

    def test_parameters_update_after_step(self, model_and_data):
        model, dataset, optimizer = model_and_data
        changed, delta = verify_parameter_update(model, dataset.inputs, dataset.targets, optimizer)
        assert changed and delta > 0

    def test_zero_grad_prevents_gradient_accumulation(self, model_and_data):
        """Two identical steps with lr=0 must report the same gradient norm."""
        model, dataset, _ = model_and_data
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        first = train_step(model, dataset.inputs, dataset.targets, optimizer)["gradient_norm"]
        second = train_step(model, dataset.inputs, dataset.targets, optimizer)["gradient_norm"]
        assert first == pytest.approx(second, rel=1e-6)

    def test_total_gradient_norm_matches_definition(self, model_and_data):
        model, dataset, _ = model_and_data
        logits, _ = model(dataset.inputs)
        model.compute_loss(logits, dataset.targets).backward()
        expected = sum(p.grad.double().pow(2).sum() for p in model.parameters()).sqrt().item()
        assert total_gradient_norm(model) == pytest.approx(expected, rel=1e-6)

    def test_instrumentation_records_every_recursive_step(self, model_and_data):
        model, dataset, optimizer = model_and_data
        metrics = train_step(model, dataset.inputs, dataset.targets, optimizer, instrument_gradients=True)
        for k in range(1, 5):
            assert metrics[f"grad_zL_step_{k}"] > 0
            assert metrics[f"zL_norm_step_{k}"] > 0
        for k in range(1, 4):
            assert metrics[f"grad_zH_step_{k}"] > 0
        # z_H^(K) feeds nothing: its gradient is absent, and reported as such (not 0.0).
        assert metrics["grad_zH_step_4"] is None

    def test_instrumentation_does_not_change_the_update(self):
        dataset = CopyDataset(num_examples=8, seq_len=8, vocab_size=16, seed=0)
        params = []
        for instrument in (False, True):
            model = make_model(k=2, seed=0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            train_step(model, dataset.inputs, dataset.targets, optimizer, instrument_gradients=instrument)
            params.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        assert torch.equal(params[0], params[1])


class TestTrainFullBatch:
    def test_stops_early_below_threshold(self, model_and_data):
        model, dataset, optimizer = model_and_data
        result = train_full_batch(model, dataset.inputs, dataset.targets, optimizer,
                                  max_steps=5, early_stop_loss=1e9)
        assert result.stopped_reason == "early_stop" and result.steps == 1

    def test_runs_to_max_steps(self, model_and_data):
        model, dataset, optimizer = model_and_data
        seen = []
        result = train_full_batch(model, dataset.inputs, dataset.targets, optimizer, max_steps=3,
                                  early_stop_loss=0.0, on_step=lambda step, m: seen.append(step))
        assert result.stopped_reason == "max_steps" and result.steps == 3 and seen == [1, 2, 3]
        assert result.final_loss < result.initial_loss

    def test_non_finite_loss_is_reported(self, model_and_data):
        model, dataset, optimizer = model_and_data
        with torch.no_grad():
            model.output_head.bias.fill_(float("nan"))
        result = train_full_batch(model, dataset.inputs, dataset.targets, optimizer,
                                  max_steps=10, early_stop_loss=0.01)
        assert result.stopped_reason == "non_finite_loss" and result.steps == 1

    def test_deterministic_given_seed(self):
        dataset = CopyDataset(num_examples=8, seq_len=8, vocab_size=16, seed=0)
        losses = []
        for _ in range(2):
            model = make_model(k=2, seed=7)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            history = []
            train_full_batch(model, dataset.inputs, dataset.targets, optimizer, max_steps=3,
                             early_stop_loss=0.0, on_step=lambda s, m: history.append(m["loss"]))
            losses.append(history)
        assert losses[0] == losses[1]


def test_retained_state_gradients_are_finite():
    model = make_model()
    tokens = torch.randint(0, 16, (2, 8))
    logits, history = model(tokens, return_states=True)
    retain_state_gradients(history)
    model.compute_loss(logits, torch.randint(0, 16, (2, 8))).backward()
    norms = collect_state_gradient_norms(history)
    for k in range(1, 5):
        assert norms[f"grad_zL_step_{k}"] is not None
        assert torch.isfinite(torch.tensor(norms[f"grad_zL_step_{k}"]))
