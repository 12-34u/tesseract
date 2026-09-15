"""Centralised device selection for Tesseract."""

import torch


def resolve_device(requested: str) -> torch.device:
    """Turn a device request into a ``torch.device``.

    Args:
        requested: ``"auto"`` (CUDA if available, otherwise CPU), ``"cpu"``,
            ``"cuda"`` or ``"cuda:N"``.

    Raises:
        ValueError: If the string is not a supported device.
        RuntimeError: If CUDA is explicitly requested but unavailable. An
            explicit request is never silently downgraded to CPU.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except RuntimeError as exc:
        raise ValueError(
            f"Invalid device {requested!r}; expected 'auto', 'cpu', 'cuda' or 'cuda:N'."
        ) from exc

    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"Unsupported device type {device.type!r}; expected 'auto', 'cpu' or 'cuda'."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {requested!r} was requested but CUDA is not available "
            f"(torch {torch.__version__}, CUDA build: {torch.version.cuda}). "
            "Use --device cpu or --device auto."
        )
    return device


def describe_device(device: torch.device) -> dict[str, object]:
    """Return device facts that affect timing/numerics, for run metadata."""
    info: dict[str, object] = {
        "type": device.type,
        "torch_num_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_version"] = torch.version.cuda
    return info
