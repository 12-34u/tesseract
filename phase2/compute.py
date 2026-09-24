"""Compute-platform control for Phase 2: determinism, and metadata that records it.

Phase 1 ran on CPU. Phase 2 runs on GPU (Amendment 02a), which changes the
numerics, so two things must be true and must be recorded:

1. **Same-hardware determinism.** Re-running a cell on the same device with the
   same seed must reproduce it exactly. ``configure_determinism`` enables
   PyTorch's deterministic kernels, disables cuDNN autotuning, and turns off
   TF32. TF32 matters most: on Ampere it silently truncates float32 matmul
   inputs to 10 mantissa bits, which would make CPU and GPU disagree for a
   reason that has nothing to do with the experiment.

2. **Cross-device agreement within a stated tolerance.** Determinism is
   per-device; bitwise CPU/GPU equality is not achievable and is not claimed.
   ``phase2.crosscheck`` measures the disagreement against the tolerance
   Amendment 02a states in advance.

``utils/device.py`` and ``utils/run_artifacts.py`` are frozen Phase 1 files, so
the extra metadata is added here and attached to Phase 2 runs through
``run_cell(provenance=...)`` rather than by editing the recorder.
"""

import os
import platform
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import torch

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_cublas_workspace_config() -> Dict[str, object]:
    """Set CUBLAS_WORKSPACE_CONFIG, which cuBLAS reads when its handle is created.

    Deterministic cuBLAS reductions require this variable, and it is read once,
    at handle creation. Setting it after CUDA is already initialised may have no
    effect, so that case is reported rather than passed over in silence; call
    this before torch touches the GPU.
    """
    already = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return {"value": CUBLAS_WORKSPACE_CONFIG, "previous": already,
            "set_before_cuda_init": not torch.cuda.is_initialized()}


def configure_determinism(device: torch.device) -> Dict[str, object]:
    """Put the process into deterministic mode and return exactly what was set."""
    record = {"cublas_workspace_config": set_cublas_workspace_config()}

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Full float32 precision on every backend, so the GPU is not quietly less
    # precise than the CPU it is being compared against.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    record.update(
        use_deterministic_algorithms=True,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        device=str(device),
    )
    return record


def nvidia_driver_version() -> Optional[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0].strip() or None


def describe_compute(device: torch.device) -> Dict[str, object]:
    """Every platform fact that can change Phase 2 numerics or timings."""
    info: Dict[str, object] = {
        "device": str(device),
        "device_type": device.type,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "numpy": np.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            cuda_device_index=index,
            cuda_device_name=props.name,
            cuda_capability=f"{props.major}.{props.minor}",
            cuda_total_memory_bytes=props.total_memory,
            cuda_runtime_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            nvidia_driver_version=nvidia_driver_version(),
        )
    return info


def compute_provenance(device: torch.device, determinism: Dict[str, object]) -> Dict[str, object]:
    """The provenance block attached to every Phase 2 run under Amendment 02a."""
    return {"amendment": "amendment_02a", "compute": describe_compute(device), "determinism": determinism,
            "reproducibility": "same-hardware determinism; cross-device agreement only within the "
                               "Amendment 02a tolerance"}
