"""Training utilities for Tesseract models.

Deliberately explicit PyTorch loops (no Lightning / HF Trainer / Accelerate)
so that BPTT through the recursive core stays transparent. Every step is:

    zero_grad → forward (no detach, no no_grad) → loss → backward
              → [read gradient diagnostics] → optimizer.step
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from evaluation.metrics import exact_match_accuracy, token_accuracy
from models.tesseract import TesseractModel
from utils.config import OptimizerConfig

StateHistory = List[Dict[str, torch.Tensor]]


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Create the optimizer described by ``config`` (currently AdamW only)."""
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


# ============================================================================
# Gradient / state instrumentation
# ============================================================================


def retain_state_gradients(state_history: StateHistory) -> None:
    """Call retain_grad() on every intermediate z_L / z_H tensor.

    Intermediate tensors are non-leaf, so autograd would otherwise discard
    their gradients after backward. retain_grad() does not change any
    parameter gradient.
    """
    for states in state_history:
        states["z_L"].retain_grad()
        states["z_H"].retain_grad()


def _norm_or_none(tensor: Optional[torch.Tensor]) -> Optional[float]:
    return None if tensor is None else torch.linalg.vector_norm(tensor).item()


def collect_state_gradient_norms(state_history: StateHistory) -> Dict[str, Optional[float]]:
    """Map ``grad_zL_step_k`` / ``grad_zH_step_k`` to ‖∂L/∂z‖.

    A value is ``None`` when autograd produced no gradient for that tensor. This
    is expected for z_H^(K), which the output head never consumes; anywhere
    else it means the recursion is disconnected. None is deliberately not
    reported as 0.0.
    """
    norms: Dict[str, Optional[float]] = {}
    for k, states in enumerate(state_history, start=1):
        norms[f"grad_zL_step_{k}"] = _norm_or_none(states["z_L"].grad)
        norms[f"grad_zH_step_{k}"] = _norm_or_none(states["z_H"].grad)
    return norms


def collect_state_norms(state_history: StateHistory) -> Dict[str, float]:
    """Map ``zL_norm_step_k`` / ``zH_norm_step_k`` to activation norms."""
    norms: Dict[str, float] = {}
    for k, states in enumerate(state_history, start=1):
        norms[f"zL_norm_step_{k}"] = torch.linalg.vector_norm(states["z_L"].detach()).item()
        norms[f"zH_norm_step_{k}"] = torch.linalg.vector_norm(states["z_H"].detach()).item()
    return norms


def total_gradient_norm(model: nn.Module) -> float:
    """Global L2 norm of all parameter gradients (one host sync)."""
    norms = [torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(norms).double()).item()


# ============================================================================
# Training steps and loops
# ============================================================================


def train_step(
    model: TesseractModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    instrument_gradients: bool = False,
) -> Dict[str, Any]:
    """Execute one optimizer step with full BPTT.

    Returned loss/accuracies come from this step's forward pass, i.e. they
    describe the parameters *before* this step's update.

    Args:
        instrument_gradients: also record per-recursive-step ‖∂L/∂z‖ and ‖z‖.
    """
    model.train()
    optimizer.zero_grad()

    logits, state_history = model(inputs, return_states=instrument_gradients)
    if instrument_gradients:
        retain_state_gradients(state_history)

    loss = model.compute_loss(logits, targets)
    loss.backward()

    predictions = logits.detach().argmax(dim=-1)
    metrics: Dict[str, Any] = {
        "loss": loss.item(),
        "token_accuracy": token_accuracy(predictions, targets),
        "exact_match_accuracy": exact_match_accuracy(predictions, targets),
        "gradient_norm": total_gradient_norm(model),
    }
    if instrument_gradients:
        metrics.update(collect_state_gradient_norms(state_history))
        metrics.update(collect_state_norms(state_history))

    optimizer.step()
    return metrics


@dataclass(frozen=True)
class TrainResult:
    """Outcome of :func:`train_full_batch`.

    ``final_*`` values come from the last step's forward pass, before that
    step's optimizer update.
    """

    initial_loss: float
    final_loss: float
    final_token_accuracy: float
    final_exact_match_accuracy: float
    steps: int
    stopped_reason: str  # "early_stop" | "max_steps" | "non_finite_loss"


def train_full_batch(
    model: TesseractModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    early_stop_loss: float,
    instrument: Callable[[int], bool] = lambda step: False,
    on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> TrainResult:
    """Train on the full batch until the loss drops below ``early_stop_loss``,
    ``max_steps`` is reached, or the loss becomes non-finite."""
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    initial_loss: Optional[float] = None
    stopped_reason = "max_steps"
    for step in range(1, max_steps + 1):
        metrics = train_step(model, inputs, targets, optimizer, instrument_gradients=instrument(step))
        if initial_loss is None:
            initial_loss = metrics["loss"]
        if on_step is not None:
            on_step(step, metrics)
        if metrics["loss"] < early_stop_loss:
            stopped_reason = "early_stop"
            break
        if not math.isfinite(metrics["loss"]):
            stopped_reason = "non_finite_loss"
            break

    return TrainResult(
        initial_loss=initial_loss,
        final_loss=metrics["loss"],
        final_token_accuracy=metrics["token_accuracy"],
        final_exact_match_accuracy=metrics["exact_match_accuracy"],
        steps=step,
        stopped_reason=stopped_reason,
    )


def verify_parameter_update(
    model: TesseractModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tuple[bool, float]:
    """One-step sanity check: do parameters actually change after optimizer.step()?

    Returns:
        (any parameter changed, sum of per-parameter L2 change norms)
    """
    params_before = {
        name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
    }

    model.train()
    optimizer.zero_grad()
    logits, _ = model(inputs, return_states=False)
    loss = model.compute_loss(logits, targets)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        deltas = [
            torch.linalg.vector_norm(p - params_before[name])
            for name, p in model.named_parameters()
            if name in params_before
        ]
    total_delta = torch.stack(deltas).sum().item()
    return total_delta > 0.0, total_delta


# ============================================================================
# BPTT verification
# ============================================================================


@dataclass(frozen=True)
class BPTTStepReport:
    step: int
    grad_z_L_norm: Optional[float]
    grad_z_H_norm: Optional[float]
    z_L_norm: float
    z_H_norm: float


@dataclass(frozen=True)
class BPTTReport:
    loss: float
    steps: List[BPTTStepReport]
    failures: List[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def verify_bptt(model: TesseractModel, inputs: torch.Tensor, targets: torch.Tensor) -> BPTTReport:
    """Check that the loss gradient reaches every recursive state.

    Requirements (K = model's recursive depth):
        * ∂L/∂z_L^(k) exists, is finite and non-zero for every k = 1..K
        * ∂L/∂z_H^(k) exists, is finite and non-zero for every k = 1..K-1

    z_H^(K) feeds nothing downstream, so its gradient is expected to be absent
    and is not a failure. Parameter gradients are cleared before returning.
    """
    model.train()
    model.zero_grad(set_to_none=True)

    logits, state_history = model(inputs, return_states=True)
    retain_state_gradients(state_history)
    loss = model.compute_loss(logits, targets)
    loss.backward()

    num_steps = len(state_history)
    grad_norms = collect_state_gradient_norms(state_history)
    state_norms = collect_state_norms(state_history)

    def problem(value: Optional[float]) -> bool:
        return value is None or not math.isfinite(value) or value == 0.0

    steps: List[BPTTStepReport] = []
    failures: List[str] = []
    for k in range(1, num_steps + 1):
        g_l = grad_norms[f"grad_zL_step_{k}"]
        g_h = grad_norms[f"grad_zH_step_{k}"]
        if problem(g_l):
            failures.append(f"||dL/dz_L^({k})|| = {g_l}")
        if k < num_steps and problem(g_h):
            failures.append(f"||dL/dz_H^({k})|| = {g_h}")
        steps.append(
            BPTTStepReport(
                step=k,
                grad_z_L_norm=g_l,
                grad_z_H_norm=g_h,
                z_L_norm=state_norms[f"zL_norm_step_{k}"],
                z_H_norm=state_norms[f"zH_norm_step_{k}"],
            )
        )

    model.zero_grad(set_to_none=True)
    return BPTTReport(loss=loss.item(), steps=steps, failures=failures)
