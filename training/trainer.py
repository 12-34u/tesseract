"""Training loop for Tesseract models.

Implements a simple, explicit PyTorch training loop with:
    - Full BPTT through recursive computation (no detach, no no_grad)
    - Per-step gradient instrumentation on recursive z_L/z_H states
    - Token-level and exact-match accuracy tracking
    - One-step parameter update verification

No PyTorch Lightning, no HuggingFace Trainer, no Accelerate.
The loop is deliberately transparent for BPTT research.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.tesseract import TesseractModel
from training.logger import ExperimentLogger


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute per-token accuracy (%).

    Args:
        logits: [B, N, V] model output.
        targets: [B, N] integer targets.

    Returns:
        Token accuracy as a percentage (0-100).
    """
    preds = logits.argmax(dim=-1)  # [B, N]
    correct = (preds == targets).float()
    return correct.mean().item() * 100.0


def compute_exact_match_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute exact-match sequence accuracy (%).

    A sequence counts as correct only if ALL tokens match.

    Args:
        logits: [B, N, V] model output.
        targets: [B, N] integer targets.

    Returns:
        Exact-match accuracy as a percentage (0-100).
    """
    preds = logits.argmax(dim=-1)  # [B, N]
    seq_correct = (preds == targets).all(dim=-1).float()  # [B]
    return seq_correct.mean().item() * 100.0


def retain_state_gradients(state_history: List[Dict[str, torch.Tensor]]) -> None:
    """Call retain_grad() on all intermediate z_L and z_H tensors.

    This is necessary because intermediate tensors are non-leaf and their
    gradients would normally be discarded after backward.

    Args:
        state_history: List of state dicts from RecursiveReasoner.
    """
    if state_history is None:
        return
    for states in state_history:
        states["z_L"].retain_grad()
        states["z_H"].retain_grad()


def collect_state_gradient_norms(
    state_history: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Collect gradient norms from retained intermediate states.

    Args:
        state_history: List of state dicts (must have had retain_grad called).

    Returns:
        Dict mapping e.g. 'grad_zL_step_1' → gradient norm value.
    """
    norms = {}
    if state_history is None:
        return norms

    for k, states in enumerate(state_history, start=1):
        z_L = states["z_L"]
        z_H = states["z_H"]

        if z_L.grad is not None:
            norms[f"grad_zL_step_{k}"] = torch.norm(z_L.grad).item()
        else:
            norms[f"grad_zL_step_{k}"] = 0.0

        if z_H.grad is not None:
            norms[f"grad_zH_step_{k}"] = torch.norm(z_H.grad).item()
        else:
            norms[f"grad_zH_step_{k}"] = 0.0

    return norms


def collect_state_norms(
    state_history: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Collect activation norms from intermediate states.

    Args:
        state_history: List of state dicts.

    Returns:
        Dict mapping e.g. 'zL_norm_step_1' → norm value.
    """
    norms = {}
    if state_history is None:
        return norms

    for k, states in enumerate(state_history, start=1):
        norms[f"zL_norm_step_{k}"] = torch.norm(states["z_L"]).item()
        norms[f"zH_norm_step_{k}"] = torch.norm(states["z_H"]).item()

    return norms


def train_step(
    model: TesseractModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    instrument_gradients: bool = True,
) -> Dict[str, Any]:
    """Execute a single training step with full BPTT.

    Args:
        model: The TesseractModel instance.
        inputs: Token IDs [B, N].
        targets: Target IDs [B, N].
        optimizer: The optimizer.
        instrument_gradients: Whether to retain and inspect recursive state grads.

    Returns:
        Dict of metrics for this step.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass WITH state history for gradient instrumentation
    logits, state_history = model(inputs, return_states=instrument_gradients)

    # Retain gradients on intermediate z_L/z_H tensors BEFORE backward
    if instrument_gradients and state_history is not None:
        retain_state_gradients(state_history)

    # Compute loss
    loss = model.compute_loss(logits, targets)

    # Full BPTT — no detach, no torch.no_grad
    loss.backward()

    # Collect gradient norms AFTER backward, BEFORE optimizer.step
    metrics: Dict[str, Any] = {
        "loss": loss.item(),
        "token_accuracy": compute_token_accuracy(logits.detach(), targets),
        "exact_match_accuracy": compute_exact_match_accuracy(logits.detach(), targets),
    }

    # Total gradient norm across all model parameters
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    metrics["gradient_norm"] = total_grad_norm ** 0.5

    # Per-recursive-step gradient norms
    if instrument_gradients and state_history is not None:
        grad_norms = collect_state_gradient_norms(state_history)
        metrics.update(grad_norms)

        state_norms = collect_state_norms(state_history)
        metrics.update(state_norms)

    # Update parameters
    optimizer.step()

    return metrics


def verify_parameter_update(
    model: TesseractModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Tuple[bool, float]:
    """One-step sanity check: verify parameters actually change after optimizer.step().

    Args:
        model: TesseractModel.
        inputs: Token IDs [B, N].
        targets: Target IDs [B, N].
        optimizer: The optimizer.

    Returns:
        Tuple of (parameters_changed: bool, total_delta_norm: float).
    """
    # Snapshot parameters before
    params_before = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    # One training step
    model.train()
    optimizer.zero_grad()
    logits, _ = model(inputs, return_states=False)
    loss = model.compute_loss(logits, targets)
    loss.backward()
    optimizer.step()

    # Compare parameters after
    total_delta = 0.0
    any_changed = False
    for name, p in model.named_parameters():
        if p.requires_grad and name in params_before:
            delta = torch.norm(p.data - params_before[name]).item()
            total_delta += delta
            if delta > 0:
                any_changed = True

    return any_changed, total_delta
