"""Batch 01 diagnostics (PHASE2_BATCH_01_PREREG.md, commit 7dbd7b6).

Each arm changes exactly one thing against the reference configuration, which
is Probe A. Nothing here is part of the Phase 2 grid, and no number produced by
these arms may be used as evidence for the depth hypothesis.

The arms reuse the existing training loop and the existing model builder; what
they vary is the data stream, the read-out normalisation, or the optimiser
schedule. The test split is never read by anything in this module.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from phase2.data import Split, TrainingStream, digest, make_targets, sample_inputs
from phase2.groups import get_group

# ---------------------------------------------------------------------------
# Arm 1 — fixed-set fit
# ---------------------------------------------------------------------------


class FixedSetStream:
    """Cycles one frozen set of examples instead of sampling fresh ones.

    This is the single change Arm 1 makes against the reference: the reference
    draws a new batch every step, this one repeats the same `size` examples.
    The arm therefore measures whether the model can *fit* the function at this
    depth, not whether it generalises, and its numbers must never be reported
    as task performance.
    """

    def __init__(self, group_name: str, task: str, t: int, n: int, batch_size: int, size: int,
                 data_seed: int, model_seed: int, excluded_inputs=()) -> None:
        if size < batch_size:
            raise ValueError(f"fixed set of {size} is smaller than one batch of {batch_size}")
        # Drawn through the ordinary stream so the leakage guard still applies:
        # no validation or test input can enter the fixed set.
        source = TrainingStream(group_name, task, t, n, size, data_seed, model_seed,
                                excluded_inputs=excluded_inputs)
        self.inputs, self.targets = source.next_batch()
        self.rejected_rows = source.rejected_rows
        self.batch_size = batch_size
        self.size = size
        self.t, self.n, self.task = t, n, task
        self.group = source.group
        self.rng = np.random.default_rng(model_seed)
        self.batches_drawn = 0
        self._order = self.rng.permutation(size)
        self._cursor = 0

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._cursor + self.batch_size > self.size:
            self._order = self.rng.permutation(self.size)
            self._cursor = 0
        idx = self._order[self._cursor:self._cursor + self.batch_size]
        self._cursor += self.batch_size
        self.batches_drawn += 1
        return self.inputs[idx], self.targets[idx]

    def as_split(self) -> Split:
        """The fixed set as an evaluation split, so fit is measured on what was trained."""
        return Split(group=self.group.name if hasattr(self.group, "name") else "A5", task=self.task,
                     split="fixed_fit", t=self.t, n=self.n, inputs=self.inputs, targets=self.targets,
                     sha256=digest(self.inputs, self.targets))


# ---------------------------------------------------------------------------
# Arm 2 — residual-stream read-out normalisation
# ---------------------------------------------------------------------------


class FinalLayerNorm(nn.Module):
    """A LayerNorm on the state just before the output projection.

    Phase 1 established that without it the residual stream grows with the
    number of block executions, so step-0 loss rises steeply with K (measured:
    4.43 nats at K=1 against 33.97 at K=8). That is a difference in optimisation
    conditions sitting on exactly the axis the experiment attributes its effect
    to.

    This wraps an existing model's output head rather than altering the
    recursion, which is the point: the scale dependence enters at the read-out,
    so it is removed at the read-out and the function the block computes is left
    untouched.
    """

    def __init__(self, head: nn.Module, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = head

    def forward(self, x):
        return self.head(self.norm(x))


def find_output_head(model: nn.Module, vocab_size: int) -> Tuple[nn.Module, str, int]:
    """Locate the d_model -> vocab projection, so the norm can be inserted before it.

    Matched on `out_features == vocab_size` rather than on the widest layer: the
    feed-forward expansion is wider than the vocabulary (1024 against 60), so a
    "widest layer" heuristic silently patches the FFN instead and the norm
    becomes a no-op.
    """
    candidates = [(name, m) for name, m in model.named_modules()
                  if isinstance(m, nn.Linear) and m.out_features == vocab_size]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one Linear with out_features={vocab_size}, found {len(candidates)}: "
            f"{[n for n, _ in candidates]}")
    name, head = candidates[0]
    return head, name, head.in_features


def apply_final_layer_norm(model: nn.Module, vocab_size: int) -> str:
    """Insert a LayerNorm before the output projection. Returns the module path patched."""
    head, name, d_model = find_output_head(model, vocab_size)
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], FinalLayerNorm(head, d_model).to(next(head.parameters()).device))
    return name


@torch.no_grad()
def step_zero_loss(model: nn.Module, split: Split, device: torch.device, batch_size: int) -> float:
    """Mean per-token cross-entropy of an untrained model, in nats."""
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(split.inputs), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(split.inputs[start:start + batch_size])).to(device)
        y = torch.from_numpy(np.ascontiguousarray(split.targets[start:start + batch_size])).to(device)
        logits = model(x, return_states=False)[0]
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        tokens += y.numel()
    return total / tokens


# ---------------------------------------------------------------------------
# Arm 3 — curriculum over T
# ---------------------------------------------------------------------------


class CurriculumStream:
    """Trains on a rising sequence of depths, ending at the target depth.

    The schedule is fixed by Amendment 03 and is identical for every K and every
    baseline, so it cannot be tuned per arm after a result is seen. Only the
    depth of the generated targets changes between stages; the inputs are drawn
    the same way throughout.
    """

    def __init__(self, group_name: str, task: str, n: int, batch_size: int, data_seed: int, model_seed: int,
                 stages: List[int], steps_per_stage: int, target_t: int, target_steps: int,
                 excluded_inputs=()) -> None:
        self.schedule: List[Tuple[int, int]] = [(t, steps_per_stage) for t in stages]
        self.schedule.append((target_t, target_steps))
        self.streams = {
            t: TrainingStream(group_name, task, t, n, batch_size, data_seed, model_seed,
                              excluded_inputs=excluded_inputs)
            for t, _ in self.schedule
        }
        self.target_t = target_t
        self.batches_drawn = 0
        self._stage = 0
        self._in_stage = 0

    @property
    def current_t(self) -> int:
        return self.schedule[min(self._stage, len(self.schedule) - 1)][0]

    @property
    def rejected_rows(self) -> int:
        return sum(s.rejected_rows for s in self.streams.values())

    def stage_boundaries(self) -> List[Dict]:
        """Cumulative step at which each stage ends, for the per-stage diagnostics."""
        out, cumulative = [], 0
        for t, steps in self.schedule:
            cumulative += steps
            out.append({"t": t, "steps": steps, "ends_at_step": cumulative})
        return out

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        t, steps = self.schedule[min(self._stage, len(self.schedule) - 1)]
        if self._in_stage >= steps and self._stage < len(self.schedule) - 1:
            self._stage += 1
            self._in_stage = 0
            t, steps = self.schedule[self._stage]
        self._in_stage += 1
        self.batches_drawn += 1
        return self.streams[t].next_batch()


# ---------------------------------------------------------------------------
# Arm 4 — warmup and decay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmupCosine:
    """Linear warmup to `peak`, then cosine decay to `final` at `total_steps`.

    Pre-registered in PHASE2_BATCH_01_PREREG.md §7 and not tunable afterwards.
    """

    peak: float
    final: float
    warmup_steps: int
    total_steps: int

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak * (step + 1) / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        return self.final + 0.5 * (self.peak - self.final) * (1.0 + np.cos(np.pi * progress))

    def apply(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return float(lr)


def sha256_of(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@torch.no_grad()
def hidden_state_norms(model: nn.Module, inputs: np.ndarray, device: torch.device) -> List[Dict]:
    """RMS norm of z_L and z_H after each recursion step.

    This is the quantity the residual confound is actually about. Step-0 loss is
    only a symptom of it, and an output-side LayerNorm makes that symptom
    K-invariant by construction — so the growth of the stream itself has to be
    measured directly to say whether the confound is present.

    Reported as root-mean-square over the feature dimension, averaged over
    tokens and examples, so values are comparable across widths.
    """
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(inputs)).to(device)
    _, history = model(x, return_states=True)
    rows = []
    for step, state in enumerate(history, start=1):
        row = {"step": step}
        for key in ("z_L", "z_H"):
            value = state[key].float()
            row[f"{key}_rms"] = float(value.pow(2).mean(dim=-1).sqrt().mean().item())
            row[f"{key}_max_abs"] = float(value.abs().max().item())
        rows.append(row)
    return rows


def layer_norm_parameter_cost(d_model: int) -> int:
    """Parameters a final LayerNorm adds: one weight and one bias per feature."""
    return 2 * d_model


# ---------------------------------------------------------------------------
# Arm 2c — per-step normalisation inside the recursion
# ---------------------------------------------------------------------------


class PerStepNormalisedReasoner(nn.Module):
    """The frozen recursion, with z_L and z_H normalised after every block call.

    The residual stream grows because nothing bounds z_L between executions:
    measured RMS rises 1.43 -> 25.30 over eight steps. An output-side norm hides
    that in the loss without touching it. This bounds the state where the growth
    happens.

    ``models/`` is frozen, so this does not edit the reasoner. It wraps the
    existing one and reuses its parameters — the same shared TransformerBlock
    and the same learnable initial states — reproducing the recorded update rule

        m^(k)   = z_L^(k-1) + z_H^(k-1) + x_emb
        z_L^(k) = f_theta(m^(k))
        z_H^(k) = alpha * z_H^(k-1) + (1 - alpha) * z_L^(k)

    and adding a normalisation of each state at the end of the step. The two
    LayerNorms are shared across steps, exactly as the block is, so the
    parameter count stays independent of K.
    """

    def __init__(self, reasoner: nn.Module, elementwise_affine: bool = False) -> None:
        super().__init__()
        self.inner = reasoner
        self.d_model = reasoner.d_model
        self.num_recursive_steps = reasoner.num_recursive_steps
        self.alpha = reasoner.alpha
        self.elementwise_affine = elementwise_affine
        self.norm_L = nn.LayerNorm(self.d_model, elementwise_affine=elementwise_affine)
        self.norm_H = nn.LayerNorm(self.d_model, elementwise_affine=elementwise_affine)

    @property
    def shared_block(self) -> nn.Module:
        return self.inner.shared_block

    def forward(self, x_emb: torch.Tensor, return_states: bool = False):
        if x_emb.ndim != 3:
            raise ValueError(f"Expected 3D input [B, N, D], got shape {list(x_emb.shape)}.")
        b, n, d = x_emb.shape
        if d != self.d_model:
            raise ValueError(f"Input dimension D ({d}) does not match d_model ({self.d_model}).")

        z_H = self.inner.learnable_init_H.unsqueeze(0).unsqueeze(0).expand(b, n, d)
        z_L = self.inner.learnable_init_L.unsqueeze(0).unsqueeze(0).expand(b, n, d)
        history = [] if return_states else None

        for _ in range(self.num_recursive_steps):
            merged = z_L + z_H + x_emb
            z_L = self.shared_block(merged)
            z_H = self.alpha * z_H + (1 - self.alpha) * z_L
            # The single change: bound both states before the next execution.
            z_L = self.norm_L(z_L)
            z_H = self.norm_H(z_H)
            # No detach: full BPTT through every step, as in the frozen reasoner.
            if return_states:
                history.append({"z_H": z_H, "z_L": z_L})
        return z_L, history


def per_step_norm_parameter_cost(d_model: int, elementwise_affine: bool = False) -> int:
    """Parameters per-step normalisation adds.

    Zero when non-affine, which is what Addendum 02 adopts; two LayerNorms'
    worth when affine, as Addendum 01 originally specified.
    """
    return 2 * layer_norm_parameter_cost(d_model) if elementwise_affine else 0


def apply_per_step_norm(model: nn.Module, elementwise_affine: bool = False) -> str:
    """Replace the reasoner with the per-step-normalised one. Returns the path patched.

    Non-affine by default (Addendum 02). An affine normalisation bounds the
    state only at initialisation: its gain is a free parameter, and nothing in
    the objective stops it growing with depth during training, which would undo
    the bound while the fix is still nominally in place. Without a gain the
    output has unit RMS by construction at every step of training.
    """
    if not hasattr(model, "reasoner"):
        raise ValueError("model has no `reasoner` to normalise; per-step norm applies to the shared-block family")
    model.reasoner = PerStepNormalisedReasoner(model.reasoner, elementwise_affine=elementwise_affine)
    return "reasoner"



# ---------------------------------------------------------------------------
# Arm configuration, so isolation between arms can be asserted rather than hoped
# ---------------------------------------------------------------------------

#: What each Batch 01 arm resolves to. Kept as data so a change to one arm can
#: be shown not to touch the others: `tests/phase2/test_batch01.py` compares
#: this against committed expected values.
ARM_CONFIG = {
    "1": {"fix": "none", "curriculum": False, "steps": 20000, "seeds": 1, "fixed_set": 512},
    "2b": {"fix": "output", "curriculum": False, "steps": 20000, "seeds": 3, "fixed_set": None},
    "3": {"fix": "none", "curriculum": True, "steps": 35000, "seeds": 3, "fixed_set": None},
    "4": {"fix": "none", "curriculum": False, "steps": 40000, "seeds": 3, "fixed_set": None},
    "2c": {"fix": "per_step", "curriculum": False, "steps": 20000, "seeds": 3, "fixed_set": None},
}


def resolved_arm_config(arm: str) -> Dict:
    """The resolved settings for one arm, as a plain comparable mapping."""
    if arm not in ARM_CONFIG:
        raise ValueError(f"unknown arm {arm!r}; known arms are {sorted(ARM_CONFIG)}")
    return dict(ARM_CONFIG[arm])
