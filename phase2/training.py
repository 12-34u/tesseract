"""Fixed-step training protocol (PHASE2_BENCHMARK_DESIGN.md §4.12).

* Every grid cell runs exactly ``max_steps`` optimizer steps. There is no
  early stopping on training loss. The only run that may stop on a
  criterion is the pilot's calibration run, which is not a grid cell and uses
  a pre-registered ``stop_condition``.
* Every step draws a fresh batch from the training stream.
* Validation runs every ``eval_every`` steps and at the last step. The
  checkpoint with the best chance-normalised validation token accuracy
  (earliest on ties) is evaluated once on the test split.
* A non-finite loss stops the run, and the stop is reported.

Each update is Phase 1's ``training.trainer.train_step``, unmodified.
"""

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from phase2.config import TrainingConfig
from phase2.data import Split, TrainingStream
from phase2.metrics import SplitMetrics, compute_metrics
from training.trainer import build_optimizer, train_step


@torch.no_grad()
def predict(model: nn.Module, inputs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    was_training = model.training
    model.eval()
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch_size])).to(device)
        outputs.append(model(batch, return_states=False)[0].argmax(dim=-1).cpu())
    model.train(was_training)
    return torch.cat(outputs).numpy()


def evaluate_split(model: nn.Module, split: Split, device: torch.device, vocab_size: int, batch_size: int) -> SplitMetrics:
    return compute_metrics(predict(model, split.inputs, device, batch_size), split.targets, vocab_size)


@dataclass
class TrainOutcome:
    steps_run: int
    stopped_reason: str  # "budget" | "non_finite_loss" | "stop_condition"
    best_step: Optional[int]
    best_val: Optional[SplitMetrics]
    final_val: Optional[SplitMetrics]
    test: Optional[SplitMetrics]
    test_predictions: Optional[np.ndarray]
    history: List[Dict]
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
    best_state, best_score, best_step, best_val, final_val = None, -math.inf, None, None, None
    stopped_reason = "budget"
    train_seconds = eval_seconds = 0.0
    step = 0

    for step in range(1, training.max_steps + 1):
        inputs, targets = stream.next_batch()
        start = time.perf_counter()
        metrics = train_step(model, torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device), optimizer)
        train_seconds += time.perf_counter() - start

        if not math.isfinite(metrics["loss"]):
            history.append({"step": step, "train_loss": metrics["loss"]})
            stopped_reason = "non_finite_loss"
            log(f"    step {step}: non-finite loss — stopping")
            break

        if step % training.eval_every == 0 or step == training.max_steps:
            start = time.perf_counter()
            val = evaluate_split(model, val_split, device, vocab_size, training.eval_batch_size)
            eval_seconds += time.perf_counter() - start
            final_val = val
            history.append({
                "step": step,
                "train_loss": metrics["loss"],
                "train_batch_token_accuracy": metrics["token_accuracy"],
                "gradient_norm": metrics["gradient_norm"],
                "val_exact_match": val.exact_match,
                "val_token_accuracy": val.token_accuracy,
                "val_chance_normalised_token_accuracy": val.chance_normalised_token_accuracy,
            })
            log(f"    step {step:>6d} | loss {metrics['loss']:.4f} | val tok {val.token_accuracy:6.2f}% "
                f"(norm {val.chance_normalised_token_accuracy:.3f}) | val EM {val.exact_match:6.2f}%")
            if val.chance_normalised_token_accuracy > best_score:
                best_score, best_step, best_val = val.chance_normalised_token_accuracy, step, val
                best_state = copy.deepcopy(model.state_dict())
            if stop_condition is not None and stop_condition(val):
                stopped_reason = "stop_condition"
                break

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
        final_val=final_val,
        test=test,
        test_predictions=test_predictions,
        history=history,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        rejected_training_rows=stream.rejected_rows,
    )
