"""Evaluation metrics for Tesseract sequence-to-sequence tasks.

Single implementation of the accuracy metrics used by every experiment:

* token accuracy: percentage of positions where prediction == target
* exact-match accuracy: percentage of sequences where EVERY position matches

For a sequence of length N, one wrong token makes that whole sequence an
exact-match failure, so exact match can be 0 % while token accuracy is high.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


def _validate(predictions: torch.Tensor, targets: torch.Tensor) -> None:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions shape {list(predictions.shape)} != targets shape {list(targets.shape)}"
        )
    if predictions.ndim != 2 or predictions.numel() == 0:
        raise ValueError(f"expected non-empty [B, N] tensors, got shape {list(predictions.shape)}")


def token_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-token accuracy in percent (0-100). Inputs: integer [B, N] tensors."""
    _validate(predictions, targets)
    return (predictions == targets).float().mean().item() * 100.0


def exact_match_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Sequence exact-match accuracy in percent (0-100). Inputs: integer [B, N] tensors."""
    _validate(predictions, targets)
    return (predictions == targets).all(dim=-1).float().mean().item() * 100.0


def token_errors_per_sequence(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Number of wrong tokens in each sequence, shape [B]."""
    _validate(predictions, targets)
    return (predictions != targets).sum(dim=-1)


@dataclass(frozen=True)
class EvalResult:
    token_accuracy: float
    exact_match_accuracy: float
    predictions: torch.Tensor  # [B, N], same device as inputs


@torch.no_grad()
def evaluate(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> EvalResult:
    """Evaluate ``model`` in eval mode (no dropout, no autograd).

    The model's previous train/eval mode is restored afterwards.
    """
    was_training = model.training
    model.eval()
    logits, _ = model(inputs, return_states=False)
    model.train(was_training)
    predictions = logits.argmax(dim=-1)
    return EvalResult(
        token_accuracy=token_accuracy(predictions, targets),
        exact_match_accuracy=exact_match_accuracy(predictions, targets),
        predictions=predictions,
    )
