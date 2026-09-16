"""Typed configuration for Phase 2.

This module reuses Phase 1's strict YAML → dataclass loader (``utils.config``)
without modifying it. Benchmark constants live in
``configs/phase2/benchmark.yaml`` and are referenced by name from run configs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from phase2.groups import BENCHMARK_GROUPS
from phase2.tasks import TASKS
from utils.config import (
    ConfigError,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    from_dict,
    load_config,
    load_yaml,
    resolve_config_path,
)

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
    budget_multiplier: int  # fixed budget = exactly multiplier × steps-to-criterion


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
class ShortcutRecord:
    group: str
    task: str
    t: int
    identity: str
    candidate: str  # name as produced by phase2.gates.shallow_candidates
    condition: str
    expected_token_agreement: float
    tolerance: float


@dataclass(frozen=True)
class AmendmentConfig:
    id: str
    date: str
    recorded_before_training: bool
    original_gates_output_dir: str
    summary: str
    anchor_t_values: tuple[int, ...]
    diagnostic_t_values: tuple[int, ...]
    primary_depth_t_values: tuple[int, ...]
    documented_shortcuts: tuple[ShortcutRecord, ...]

    def __post_init__(self) -> None:
        _require(self.recorded_before_training, "an amendment must be recorded before training")
        roles = (set(self.anchor_t_values), set(self.diagnostic_t_values), set(self.primary_depth_t_values))
        _require(sum(len(r) for r in roles) == len(set().union(*roles)), "T roles must be disjoint")
        _require(all(s.t in self.diagnostic_t_values for s in self.documented_shortcuts),
                 "documented shortcuts must be at diagnostic T values")

    @property
    def gate_verdict(self) -> str:
        return f"PASS_WITH_{self.id.upper()}"


@dataclass(frozen=True)
class AmendedGatesRunConfig:
    experiment: RunConfig
    benchmark: str
    amendment: str


def load_amendment(name_or_path: str = "phase2/amendment_01") -> AmendmentConfig:
    return load_config(name_or_path, AmendmentConfig)


@dataclass(frozen=True)
class ModelVariant:
    name: str  # label used in artifact paths, e.g. "small"
    base: str  # model base config name or path, e.g. "prototype_small"


@dataclass(frozen=True)
class ModelFamily:
    """One selectable Tesseract width.

    Widths differ only in capacity. Every one of them is a single shared
    TransformerBlock executed K times, so the trainable parameter count is
    constant across K within a family. ``approx_parameters`` is documentation
    at vocab_size=60; the authoritative count is always measured from the
    instantiated model.
    """

    name: str
    base: str
    label: str
    approx_parameters: int


# The selectable model widths. ``model_base`` in a run config must name one of
# these bases; nothing defaults to any of them, and the Phase 2 experiment keeps
# model_base = PENDING_P1B until a width is approved after P1b.
MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily("small", "prototype_small", "Small Prototype (222K)", 222_140),
    ModelFamily("medium", "phase2/prototype_medium", "Medium Prototype (837K)", 837_436),
    ModelFamily("large", "phase2/prototype_large", "Large Research Model (~7M)", 7_230_780),
)

MODEL_BASES: tuple[str, ...] = tuple(f.base for f in MODEL_FAMILIES)


def model_family(base: str) -> ModelFamily:
    """The registered family for a model base config, or a loud failure."""
    for family in MODEL_FAMILIES:
        if family.base == base or family.name == base:
            return family
    raise ConfigError(f"unknown model base {base!r}; registered bases are {list(MODEL_BASES)}")


@dataclass(frozen=True)
class CapacityDiagnosticConfig:
    """P1b (PHASE2_BENCHMARK_DESIGN.md §9): model width × K at one T, fixed steps, validation only."""

    experiment: RunConfig
    benchmark: str
    vocab_size: int
    models: tuple[ModelVariant, ...]
    t: int
    k_values: tuple[int, ...]
    optimizer: OptimizerConfig
    batch_size: int
    eval_every: int
    eval_batch_size: int
    max_steps: int
    floor_threshold: float
    reference_pilot_output_dir: str

    def __post_init__(self) -> None:
        _require(len({m.name for m in self.models}) == len(self.models), "model variant names must be unique")
        _require(self.max_steps > 0 and self.eval_every > 0 and self.batch_size > 0, "sizes must be positive")


def resolve_model_variant(variant: ModelVariant, vocab_size: int) -> ModelConfig:
    """Load a model base config, overriding only the vocabulary."""
    base = load_yaml(resolve_config_path(variant.base)).get("model")
    _require(isinstance(base, dict), f"{variant.base}: no 'model' mapping")
    return from_dict(ModelConfig, {**base, "vocab_size": vocab_size}, f"{variant.base}:model")


def load_benchmark(name_or_path: str = "phase2/benchmark") -> BenchmarkConfig:
    return load_config(name_or_path, BenchmarkConfig)


# ============================================================================
# Phase 2 experiment (PHASE2_BENCHMARK_DESIGN.md §10; Amendment 02 draft)
# ============================================================================

PENDING_P1B = "PENDING_P1B"


class PendingP1BError(RuntimeError):
    """A value that may only be chosen after P1b completes is still pending."""


@dataclass(frozen=True)
class StageSpec:
    name: str
    purpose: str
    family: str
    tasks: tuple[str, ...]
    t_values: tuple[int, ...]
    depths: tuple[int, ...]  # K for tesseract, L for non-shared, matched K for width_scaled
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(self.family in FAMILIES, f"stage {self.name}: family must be one of {FAMILIES}")
        _require(set(self.tasks) <= set(TASKS), f"stage {self.name}: tasks must be a subset of {TASKS}")
        _require(all(t >= 1 for t in self.t_values) and all(d >= 1 for d in self.depths), f"stage {self.name}: T/depth >= 1")


@dataclass(frozen=True)
class Phase2ExperimentBase:
    experiment: RunConfig
    benchmark: str
    amendments: tuple[str, ...]
    vocab_size: int
    optimizer: OptimizerConfig
    batch_size: int
    eval_every: int
    eval_batch_size: int
    stages: tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        _require(len({s.name for s in self.stages}) == len(self.stages), "stage names must be unique")
        _require(self.batch_size > 0 and self.eval_every > 0 and self.eval_batch_size > 0, "sizes must be positive")


@dataclass(frozen=True)
class Phase2ExperimentConfig(Phase2ExperimentBase):
    model_base: Optional[str] = None  # None = PENDING_P1B
    max_steps: Optional[int] = None  # None = PENDING_P1B

    @property
    def pending(self) -> list:
        return [name for name in ("model_base", "max_steps") if getattr(self, name) is None]

    def require_frozen(self) -> None:
        if self.pending:
            raise PendingP1BError(f"{self.pending} are still {PENDING_P1B}; they are chosen only after P1b completes "
                                  "and is approved. No Phase 2 training can run.")

    def stage(self, name: str) -> StageSpec:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ConfigError(f"unknown stage {name!r}; available: {[s.name for s in self.stages]}")


def load_phase2_experiment(name_or_path: str = "phase2/phase2_experiment") -> Phase2ExperimentConfig:
    path = resolve_config_path(name_or_path)
    raw = load_yaml(path)
    values = {}
    for key, kind in (("model_base", str), ("max_steps", int)):
        _require(key in raw, f"{path.name}: missing {key!r} (write {PENDING_P1B!r} while pending)")
        value = raw.pop(key)
        if value == PENDING_P1B:
            values[key] = None
        else:
            _require(type(value) is kind and (kind is str or value > 0), f"{path.name}: invalid {key}: {value!r}")
            values[key] = value
    base = from_dict(Phase2ExperimentBase, raw, path.name)
    return Phase2ExperimentConfig(**{f.name: getattr(base, f.name) for f in fields(base)}, **values)


@dataclass(frozen=True)
class AmendmentDecision:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class DecisionRuleConfig:
    primary_t_values: tuple[int, ...]
    k_low: int
    k_high: int
    ci_level: float
    floor_threshold: float
    tau: float
    score: str

    def __post_init__(self) -> None:
        _require(len(self.primary_t_values) == 2, "the DiD rule needs exactly two primary T values")
        _require(self.k_low < self.k_high, "k_low must be < k_high")
        _require(self.ci_level == 0.95, "only 95 % t-intervals are implemented")


@dataclass(frozen=True)
class Stage1ReviewConfig:
    t: int
    floor_threshold: float
    ceiling_tau: float


@dataclass(frozen=True)
class Amendment02Config:
    id: str
    date: str
    status: str  # "draft" | "frozen"
    frozen: bool
    decisions: tuple[AmendmentDecision, ...]
    decision_rule: DecisionRuleConfig
    stage1_review: Stage1ReviewConfig
    deviations: tuple[str, ...]
    pending_p1b: str

    def __post_init__(self) -> None:
        _require(self.status in ("draft", "frozen"), "status must be 'draft' or 'frozen'")
        _require(self.frozen == (self.status == "frozen"), "frozen flag must match status")


def load_amendment_02(name_or_path: str = "phase2/amendment_02_draft") -> Amendment02Config:
    return load_config(name_or_path, Amendment02Config)
