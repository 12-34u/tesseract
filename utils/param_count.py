"""Parameter counting utility for Tesseract models."""

from typing import Dict
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        Dict[str, int]: A dictionary with keys 'trainable' and 'total'.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
    }
