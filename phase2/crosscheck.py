"""CPU/GPU agreement check for Amendment 02a.

Phase 2 moves to GPU, so the platform must be shown not to change the science.
Two levels, because they fail for different reasons:

* **Level A, kernel agreement.** One forward and one backward from *identical*
  weights on an *identical* batch. No optimisation, so nothing is amplified:
  any disagreement is purely a difference between CPU and GPU kernels. This is
  the tight, blocking check.
* **Level B, end to end.** Train the same cell independently on each device and
  compare final metrics. Optimisation is chaotic, so a 1e-7 kernel difference
  becomes a visibly different trajectory over thousands of steps. Loose, and
  non-blocking by design: every comparison *inside* the experiment is made
  between cells on the same device.

Tolerances come from configs/phase2/amendment_02a_draft.yaml and were fixed
before this module was ever run.
"""

from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from phase2.compute import configure_determinism, describe_compute
from phase2.config import (
    Amendment02aConfig,
    BenchmarkConfig,
    CellRunConfig,
    CrossDeviceToleranceConfig,
    ModelVariant,
    load_amendment_02a,
    load_benchmark,
)
from phase2.data import SplitStore
from phase2.groups import get_group
from phase2.models import build_model
from utils.seed import set_seed

LEVEL_A_BATCH = 32


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0 else 0.0


def level_a_kernel_agreement(model_config, benchmark: BenchmarkConfig, cell, inputs: np.ndarray,
                             targets: np.ndarray, tolerance, devices=("cpu", "cuda")) -> Dict:
    """Compare one forward+backward across devices from a single shared initialisation."""
    set_seed(cell.seed)
    reference = build_model(cell.family, cell.depth, model_config, benchmark.n)
    state = {k: v.clone() for k, v in reference.model.state_dict().items()}

    x = torch.from_numpy(np.ascontiguousarray(inputs[:LEVEL_A_BATCH]))
    y = torch.from_numpy(np.ascontiguousarray(targets[:LEVEL_A_BATCH]))

    per_device = {}
    for name in devices:
        device = torch.device(name)
        built = build_model(cell.family, cell.depth, model_config, benchmark.n)
        model = built.model.to(device)
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        model.train()
        model.zero_grad(set_to_none=True)

        logits = model(x.to(device), return_states=False)[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1))
        loss.backward()

        grads = torch.cat([p.grad.detach().reshape(-1).double().cpu() for p in model.parameters()
                           if p.grad is not None])
        per_device[name] = {"logits": logits.detach().double().cpu(), "loss": float(loss.item()),
                            "grad_norm": float(grads.norm().item()), "grads": grads}

    a, b = devices
    da, db = per_device[a], per_device[b]
    logit_abs = float((da["logits"] - db["logits"]).abs().max().item())
    logit_scale = float(torch.maximum(da["logits"].abs(), db["logits"].abs()).max().item())
    grad_abs = float((da["grads"] - db["grads"]).abs().max().item())

    measured = {
        "max_abs_logit_difference": logit_abs,
        "max_rel_logit_difference": logit_abs / logit_scale if logit_scale > 0 else 0.0,
        "max_abs_loss_difference_nats": abs(da["loss"] - db["loss"]),
        "max_rel_gradient_norm_difference": _rel(da["grad_norm"], db["grad_norm"]),
        "max_abs_gradient_element_difference": grad_abs,
    }
    checks = {key: measured[key] <= getattr(tolerance, key)
              for key in ("max_abs_logit_difference", "max_rel_logit_difference",
                          "max_abs_loss_difference_nats", "max_rel_gradient_norm_difference")}
    return {
        "level": "A_kernel_agreement",
        "description": tolerance.description,
        "devices": list(devices),
        "batch": LEVEL_A_BATCH,
        "per_device": {name: {"loss": v["loss"], "grad_norm": v["grad_norm"]} for name, v in per_device.items()},
        "measured": measured,
        "tolerance": {k: getattr(tolerance, k) for k in checks},
        "checks": checks,
        "blocking": tolerance.blocking,
        "passed": all(checks.values()),
    }


def compare_level_b(cpu_result: Dict, gpu_result: Dict, tolerance) -> Dict:
    """Compare two completed training runs of the same cell on different devices."""
    def metric(result: Dict, block: Optional[str], key: str) -> Optional[float]:
        # Metrics live in a nested block; losses are recorded at the top level.
        section = result if block is None else (result.get(block) or {})
        return section.get(key)

    pairs = {
        "chance_normalised_token_accuracy": ("best_val", "chance_normalised_token_accuracy",
                                             "max_abs_chance_normalised_token_accuracy_difference"),
        "token_accuracy": ("best_val", "token_accuracy", "max_abs_token_accuracy_difference_points"),
        "final_val_loss": (None, "final_val_loss", "max_abs_final_val_loss_difference_nats"),
    }
    measured, checks, values = {}, {}, {}
    for label, (block, key, tol_name) in pairs.items():
        cpu_value, gpu_value = metric(cpu_result, block, key), metric(gpu_result, block, key)
        values[label] = {"cpu": cpu_value, "gpu": gpu_value}
        if cpu_value is None or gpu_value is None:
            measured[label], checks[label] = None, False
            continue
        measured[label] = abs(cpu_value - gpu_value)
        checks[label] = measured[label] <= getattr(tolerance, tol_name)

    return {
        "level": "B_end_to_end",
        "description": tolerance.description,
        "cell": cpu_result.get("cell"),
        "steps": cpu_result.get("steps_run"),
        "values": values,
        "measured_absolute_difference": measured,
        "tolerance": {label: getattr(tolerance, pairs[label][2]) for label in pairs},
        "checks": checks,
        "blocking": tolerance.blocking,
        "passed": all(checks.values()),
    }


def summarise(level_a: Dict, level_b: Optional[Dict], amendment: Amendment02aConfig) -> Dict:
    """Overall verdict. Only a Level A failure blocks GPU execution (Amendment 02a P4)."""
    blocking_failures = [r["level"] for r in (level_a, level_b) if r and r["blocking"] and not r["passed"]]
    advisory_failures = [r["level"] for r in (level_a, level_b) if r and not r["blocking"] and not r["passed"]]
    return {
        "amendment": amendment.id,
        "amendment_status": amendment.status,
        "level_a": level_a,
        "level_b": level_b,
        "blocking_failures": blocking_failures,
        "advisory_failures": advisory_failures,
        "gpu_execution_permitted": not blocking_failures,
        "trajectory_divergence_recorded": bool(advisory_failures),
        "passed": not blocking_failures and not advisory_failures,
    }
