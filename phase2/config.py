"""Typed configuration for Phase 2.

This module reuses Phase 1's strict YAML → dataclass loader (``utils.config``)
without modifying it. Benchmark constants live in
``configs/phase2/benchmark.yaml`` and are referenced by name from run configs.
"""

from __future__ import annotations

from dataclasses import dataclass

from phase2.groups import BENCHMARK_GROUPS
from phase2.tasks import TASKS
from utils.config import ConfigError, ModelConfig, OptimizerConfig, RunConfig, load_config

FAMILIES = ("tesseract", "unrolled", "param_matched", "width_scaled")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class GateConfig:
    seed: int
    uniformity_states: int
    uniformity_tv_max: float
    cycle_states: int
    cycle_steps: int
    sensitivity_states: int
    sensitivity_min: float
    shortcut_states: int
    shortcut_agreement_max: float


@dataclass(frozen=True)
class BenchmarkConfig:
    group: str
    ablation_groups: tuple[str, ...]
    tasks: tuple[str, ...]
    n: int
    t_values: tuple[int, ...]
    k_values: tuple[int, ...]
    interpolation_t_values: tuple[int, ...]
    length_n: int
    data_seed: int
    val_size: int
    test_size: int
    length_test_size: int
    model_seeds: tuple[int, ...]
    tau: float
    gates: GateConfig

    def __post_init__(self) -> None:
        for name in (self.group, *self.ablation_groups):
            _require(name in BENCHMARK_GROUPS, f"unknown benchmark group {name!r}")
        _require(set(self.tasks) <= set(TASKS), f"tasks must be a subset of {TASKS}")
        all_t = (*self.t_values, *self.interpolation_t_values)
        _require(all(t >= 1 for t in all_t), "T values must be >= 1")
        _require(self.n > max(all_t) + 1, "n must exceed max(T)+1 so the dependency window never wraps onto itself")
        _require(self.length_n > max(self.t_values) + 1, "length_n must exceed max(T)+1")
        _require(all(k >= 1 for k in self.k_values), "K values must be >= 1")
        _require(0.0 < self.tau <= 1.0, "tau must be in (0, 1]")


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: OptimizerConfig
    batch_size: int
    max_steps: int  # fixed step budget: every step is run, no early stopping
    eval_every: int
    eval_batch_size: int

    def __post_init__(self) -> None:
        _require(self.batch_size > 0 and self.max_steps > 0 and self.eval_every > 0, "training sizes must be positive")


@dataclass(frozen=True)
class CellSpec:
    family: str  # tesseract | unrolled | param_matched | width_scaled
    depth: int  # K for tesseract; L for non-shared; matched K for width_scaled (which executes 1 block)
    group: str
    task: str
    t: int
    seed: int

    def __post_init__(self) -> None:
        _require(self.family in FAMILIES, f"family must be one of {FAMILIES}")
        _require(self.task in TASKS, f"task must be one of {TASKS}")
        _require(self.group in BENCHMARK_GROUPS, f"group must be one of {BENCHMARK_GROUPS}")
        _require(self.depth >= 1 and self.t >= 1, "depth and T must be >= 1")

    @property
    def cell_id(self) -> str:
        return f"{self.family}_d{self.depth}_{self.group}_{self.task}_T{self.t}_s{self.seed}"


@dataclass(frozen=True)
class CellRunConfig:
    """Everything that defines one training run; saved as the run's config.yaml."""

    experiment: RunConfig
    model: ModelConfig
    benchmark: BenchmarkConfig
    training: TrainingConfig
    cell: CellSpec


@dataclass(frozen=True)
class GatesRunConfig:
    experiment: RunConfig
    benchmark: str  # config name, e.g. "phase2/benchmark"


@dataclass(frozen=True)
class CalibrationConfig:
    family: str
    depth: int
    task: str
    t: int
    max_steps: int
    converge_exact_match: float  # validation exact match (%) that counts as converged
    budget_multiplier: int  # fixed budget = multiplier × steps-to-converge (rounded up to eval_every)


@dataclass(frozen=True)
class PilotCell:
    family: str
    depth: int
    task: str
    t: int


@dataclass(frozen=True)
class PilotConfig:
    experiment: RunConfig  # experiment.seed is the model seed
    model: ModelConfig
    benchmark: str
    optimizer: OptimizerConfig
    batch_size: int
    eval_every: int
    eval_batch_size: int
    calibration: CalibrationConfig
    floor_cells: tuple[PilotCell, ...]
    floor_threshold: float  # chance-normalised token accuracy below which a cell is "at floor"


@dataclass(frozen=True)
class GridConfig:
    experiment: RunConfig
    model: ModelConfig
    benchmark: str
    optimizer: OptimizerConfig
    batch_size: int
    eval_every: int
    eval_batch_size: int
    pilot_output_dir: str  # fixed step budget is read from this pilot's report
    families: tuple[str, ...]
    include_ablation_groups: bool


def load_benchmark(name_or_path: str = "phase2/benchmark") -> BenchmarkConfig:
    return load_config(name_or_path, BenchmarkConfig)
