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
