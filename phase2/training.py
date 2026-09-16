"""Fixed-step training protocol (PHASE2_BENCHMARK_DESIGN.md §4.12).

* Every grid cell runs exactly ``max_steps`` optimizer steps, with no early
  stopping on training loss. The only run that may stop on a criterion is the
  pilot's calibration run, which is not a grid cell and uses a pre-registered
  ``stop_condition``.
* Every step draws a fresh batch from the training stream.
* Validation (loss and accuracies) runs every ``eval_every`` steps and at the
  last step. The checkpoint with the best chance-normalised validation token
  accuracy (earliest on ties) is kept. The test split, when given, is
  evaluated once, on that checkpoint; it is never used for selection.
* Gradient norms are tracked on every step. A non-finite loss stops the run,
  and the stop is reported.

Each update is Phase 1's ``training.trainer.train_step``, unmodified.
"""

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase2.config import TrainingConfig
from phase2.data import Split, TrainingStream
from phase2.metrics import SplitMetrics, compute_metrics
from training.trainer import build_optimizer, train_step


@torch.no_grad()
def predict_with_loss(model: nn.Module, inputs: np.ndarray, targets: Optional[np.ndarray], device: torch.device,
                      batch_size: int) -> Tuple[np.ndarray, Optional[float]]:
    """Eval-mode predictions and, if targets are given, mean per-token cross-entropy."""
    was_training = model.training
    model.eval()
    predictions, loss_sum, tokens = [], 0.0, 0
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch_size])).to(device)
        logits = model(batch, return_states=False)[0]
        predictions.append(logits.argmax(dim=-1).cpu())
        if targets is not None:
            y = torch.from_numpy(np.ascontiguousarray(targets[start:start + batch_size])).to(device)
            loss_sum += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
            tokens += y.numel()
    model.train(was_training)
    return torch.cat(predictions).numpy(), (loss_sum / tokens if targets is not None else None)


def predict(model: nn.Module, inputs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    return predict_with_loss(model, inputs, None, device, batch_size)[0]


def evaluate_split(model: nn.Module, split: Split, device: torch.device, vocab_size: int, batch_size: int) -> SplitMetrics:
    return compute_metrics(predict(model, split.inputs, device, batch_size), split.targets, vocab_size)


def evaluate_split_with_loss(model: nn.Module, split: Split, device: torch.device, vocab_size: int,
                             batch_size: int) -> Tuple[SplitMetrics, float]:
    predictions, loss = predict_with_loss(model, split.inputs, split.targets, device, batch_size)
    return compute_metrics(predictions, split.targets, vocab_size), loss


@dataclass
class TrainOutcome:
    steps_run: int
    stopped_reason: str  # "budget" | "non_finite_loss" | "stop_condition"
    best_step: Optional[int]
    best_val: Optional[SplitMetrics]
    best_val_loss: Optional[float]
    final_val: Optional[SplitMetrics]
    final_val_loss: Optional[float]
    test: Optional[SplitMetrics]
    test_predictions: Optional[np.ndarray]
    history: List[Dict]
    gradient_norm_stats: Dict
    best_state: Optional[Dict]
    final_state: Optional[Dict]
    train_seconds: float
    eval_seconds: float
    rejected_training_rows: int

    @property
    def seconds_per_train_step(self) -> float:
        return self.train_seconds / max(self.steps_run, 1)


def train_fixed_steps(
    model: nn.Module,
    stream: TrainingStream,
    val_split: Split,
    test_split: Optional[Split],
    training: TrainingConfig,
    device: torch.device,
    vocab_size: int,
    stop_condition: Optional[Callable[[SplitMetrics], bool]] = None,
    log: Callable[[str], None] = print,
) -> TrainOutcome:
    optimizer = build_optimizer(model, training.optimizer)
    history: List[Dict] = []
    best_state, best_score, best_step, best_val, best_val_loss = None, -math.inf, None, None, None
    final_val, final_val_loss = None, None
    stopped_reason = "budget"
    train_seconds = eval_seconds = 0.0
    grad_sum, grad_count, grad_max, grad_non_finite = 0.0, 0, 0.0, 0
    interval_losses: List[float] = []
    interval_grads: List[float] = []
    step = 0

    for step in range(1, training.max_steps + 1):
        inputs, targets = stream.next_batch()
        start = time.perf_counter()
        metrics = train_step(model, torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device), optimizer)
        train_seconds += time.perf_counter() - start

        loss, grad = metrics["loss"], metrics["gradient_norm"]
        if math.isfinite(grad):
            grad_sum, grad_count, grad_max = grad_sum + grad, grad_count + 1, max(grad_max, grad)
        else:
            grad_non_finite += 1
        if not math.isfinite(loss):
            history.append({"step": step, "train_loss_last_batch": loss, "gradient_norm_last": grad})
            stopped_reason = "non_finite_loss"
            log(f"    step {step}: non-finite loss — stopping")
            break
        interval_losses.append(loss)
        interval_grads.append(grad)

        if step % training.eval_every == 0 or step == training.max_steps:
            start = time.perf_counter()
            val, val_loss = evaluate_split_with_loss(model, val_split, device, vocab_size, training.eval_batch_size)
            eval_seconds += time.perf_counter() - start
            final_val, final_val_loss = val, val_loss
            history.append({
                "step": step,
                "train_loss_last_batch": loss,
                "train_loss_interval_mean": float(np.mean(interval_losses)),
                "gradient_norm_last": grad,
                "gradient_norm_interval_mean": float(np.mean(interval_grads)),
                "gradient_norm_interval_max": float(np.max(interval_grads)),
                "val_loss": val_loss,
                "val_exact_match": val.exact_match,
                "val_token_accuracy": val.token_accuracy,
                "val_chance_normalised_token_accuracy": val.chance_normalised_token_accuracy,
            })
            log(f"    step {step:>6d} | train loss {np.mean(interval_losses):.4f} | val loss {val_loss:.4f} | "
                f"val tok {val.token_accuracy:6.2f}% (norm {val.chance_normalised_token_accuracy:.3f}) | "
                f"val EM {val.exact_match:6.2f}% | grad max {np.max(interval_grads):.3f}")
            interval_losses, interval_grads = [], []
            if val.chance_normalised_token_accuracy > best_score:
                best_score, best_step, best_val, best_val_loss = val.chance_normalised_token_accuracy, step, val, val_loss
                best_state = copy.deepcopy(model.state_dict())
            if stop_condition is not None and stop_condition(val):
                stopped_reason = "stop_condition"
                break

    final_state = copy.deepcopy(model.state_dict())
    test, test_predictions = None, None
    if best_state is not None:
        model.load_state_dict(best_state)
        if test_split is not None:
            start = time.perf_counter()
            test_predictions = predict(model, test_split.inputs, device, training.eval_batch_size)
            test = compute_metrics(test_predictions, test_split.targets, vocab_size)
            eval_seconds += time.perf_counter() - start

    return TrainOutcome(
        steps_run=step,
        stopped_reason=stopped_reason,
        best_step=best_step,
        best_val=best_val,
        best_val_loss=best_val_loss,
        final_val=final_val,
        final_val_loss=final_val_loss,
        test=test,
        test_predictions=test_predictions,
        history=history,
        gradient_norm_stats={"mean": grad_sum / grad_count if grad_count else None,
                             "max": grad_max if grad_count else None,
                             "non_finite_steps": grad_non_finite},
        best_state=best_state,
        final_state=final_state,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        rejected_training_rows=stream.rejected_rows,
    )
