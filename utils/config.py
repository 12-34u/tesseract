"""Typed, validated configuration loading for Tesseract experiments.

Every experiment reads one YAML file in ``configs/``. The model architecture
has a single source of truth (``configs/prototype_small.yaml``). Experiment
configs reference it through ``model.base`` and may override task-dependent
fields such as ``vocab_size``.

YAML is mapped onto frozen dataclasses with strict checking: missing keys,
unknown keys and wrong types raise :class:`ConfigError`. Nothing silently
falls back to a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints

import yaml

from utils.paths import CONFIG_DIR

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


# ============================================================================
# Schema
# ============================================================================


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    d_ff: int
    alpha: float
    dropout: float

    def __post_init__(self) -> None:
        for name in ("vocab_size", "max_seq_len", "d_model", "num_heads", "d_ff"):
            _require(getattr(self, name) > 0, f"model.{name} must be positive")
        _require(
            self.d_model % self.num_heads == 0,
            f"model.d_model ({self.d_model}) must be divisible by model.num_heads ({self.num_heads})",
        )
        _require(0.0 <= self.alpha <= 1.0, f"model.alpha must be in [0, 1], got {self.alpha}")
        _require(0.0 <= self.dropout < 1.0, f"model.dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class PrototypeConfig:
    model: ModelConfig
    default_k: int

    def __post_init__(self) -> None:
        _require(self.default_k >= 1, "default_k must be >= 1")


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str  # relative paths resolve under the runs root
    seed: int
    device: str  # "auto", "cpu", "cuda" or "cuda:N"


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float

    def __post_init__(self) -> None:
        _require(self.name == "adamw", f"optimizer.name must be 'adamw', got {self.name!r}")
        _require(self.learning_rate > 0, "optimizer.learning_rate must be positive")
        _require(self.weight_decay >= 0, "optimizer.weight_decay must be non-negative")


@dataclass(frozen=True)
class CopyDataConfig:
    task: str
    num_examples: int
    seq_len: int

    def __post_init__(self) -> None:
        _require(self.task in ("copy", "reverse"), f"data.task must be copy/reverse, got {self.task!r}")
        _require(self.num_examples > 0 and self.seq_len > 0, "data sizes must be positive")


def _require_k_values(k_values: tuple[int, ...], where: str) -> None:
    _require(all(k >= 1 for k in k_values), f"{where} must all be >= 1, got {k_values}")
    _require(len(set(k_values)) == len(k_values), f"{where} must be unique, got {k_values}")


@dataclass(frozen=True)
class CrucibleTrainingConfig:
    optimizer: OptimizerConfig
    max_steps: int
    early_stop_loss: float
    log_every: int
    grad_instrument_every: int
    grad_instrument_first_steps: int


@dataclass(frozen=True)
class CruciblePassCriteria:
    max_final_loss: float
    min_exact_match: float


@dataclass(frozen=True)
class CrucibleConfig:
    experiment: RunConfig
    model: ModelConfig
    k: int
    data: CopyDataConfig
    training: CrucibleTrainingConfig
    pass_criteria: CruciblePassCriteria

    def __post_init__(self) -> None:
        _require(self.k >= 1, f"k must be >= 1, got {self.k}")
        _require(self.data.seq_len <= self.model.max_seq_len, "data.seq_len exceeds model.max_seq_len")


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    global_warmup_k: int
    global_warmup_runs: int
    warmup_runs: int
    timing_runs: int

    def __post_init__(self) -> None:
        _require(self.timing_runs > 0, "benchmark.timing_runs must be positive")


@dataclass(frozen=True)
class KScalingTrainingConfig:
    enabled: bool
    data: CopyDataConfig
    optimizer: OptimizerConfig
    max_steps: int
    early_stop_loss: float
    log_every: int


@dataclass(frozen=True)
class KScalingConfig:
    experiment: RunConfig
    model: ModelConfig
    k_values: tuple[int, ...]
    benchmark: BenchmarkConfig
    training: KScalingTrainingConfig

    def __post_init__(self) -> None:
        _require_k_values(self.k_values, "k_values")
        _require(self.benchmark.seq_len <= self.model.max_seq_len, "benchmark.seq_len exceeds model.max_seq_len")
        _require(self.training.data.seq_len <= self.model.max_seq_len, "training.data.seq_len exceeds model.max_seq_len")


@dataclass(frozen=True)
class CADataConfig:
    rule_number: int
    seq_len: int
    t_values: tuple[int, ...]
    num_train: int
    num_val: int
    val_seed_offset: int

    def __post_init__(self) -> None:
        _require(0 <= self.rule_number <= 255, "data.rule_number must be in [0, 255]")
        _require(all(t >= 0 for t in self.t_values), "data.t_values must be non-negative")
        _require(self.val_seed_offset != 0, "data.val_seed_offset must be non-zero (train/val must differ)")


@dataclass(frozen=True)
class CATrainingConfig:
    optimizer: OptimizerConfig
    batch_size: int
    max_steps: int
    early_stop_loss: float
    eval_every: int
    eval_first_steps: int


@dataclass(frozen=True)
class CellularAutomatonConfig:
    experiment: RunConfig
    model: ModelConfig
    k_values: tuple[int, ...]
    data: CADataConfig
    training: CATrainingConfig

    def __post_init__(self) -> None:
        _require_k_values(self.k_values, "k_values")
        _require(self.model.vocab_size == 2, "cellular automaton states are binary: model.vocab_size must be 2")
        _require(self.data.seq_len <= self.model.max_seq_len, "data.seq_len exceeds model.max_seq_len")


# ============================================================================
# Loading
# ============================================================================


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value, where)
    if get_origin(tp) is tuple:
        item_tp = get_args(tp)[0]
        _require(isinstance(value, list) and len(value) > 0, f"{where}: expected a non-empty list, got {value!r}")
        return tuple(_coerce(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value))
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    # type() check (not isinstance) so that True is not accepted as an int.
    _require(type(value) is tp, f"{where}: expected {tp.__name__}, got {value!r} ({type(value).__name__})")
    return value


def from_dict(cls: type[T], data: Any, where: str) -> T:
    """Build dataclass ``cls`` from a mapping, rejecting missing/unknown keys."""
    _require(isinstance(data, Mapping), f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    missing = sorted(set(names) - set(data))
    unknown = sorted(set(data) - set(names))
    _require(not missing, f"{where}: missing keys {missing}")
    _require(not unknown, f"{where}: unknown keys {unknown}")
    return cls(**{name: _coerce(hints[name], data[name], f"{where}.{name}") for name in names})


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _require(isinstance(data, dict), f"{path}: top level must be a mapping")
    return data


def resolve_config_path(path_or_name: str | Path) -> Path:
    """Accept either a path to a YAML file or a bare name in ``configs/``."""
    path = Path(path_or_name)
    if path.suffix in (".yaml", ".yml"):
        return path.expanduser().resolve()
    return CONFIG_DIR / f"{path_or_name}.yaml"


def _resolve_model_section(section: Any, where: str) -> Any:
    if not isinstance(section, Mapping) or "base" not in section:
        return section  # Inline model definition; validated by from_dict.
    unknown = sorted(set(section) - {"base", "overrides"})
    _require(not unknown, f"{where}: unknown keys {unknown} (expected 'base' and optional 'overrides')")
    base = load_yaml(resolve_config_path(section["base"])).get("model")
    _require(isinstance(base, Mapping), f"{where}: base config {section['base']!r} has no 'model' mapping")
    overrides = section.get("overrides") or {}
    _require(isinstance(overrides, Mapping), f"{where}.overrides must be a mapping")
    bad = sorted(set(overrides) - set(base))
    _require(not bad, f"{where}.overrides: keys {bad} do not exist in base model config")
    return {**base, **overrides}


def load_config(path_or_name: str | Path, cls: type[T]) -> T:
    """Load and validate a YAML config into dataclass ``cls``."""
    path = resolve_config_path(path_or_name)
    raw = load_yaml(path)
    if "model" in raw:
        raw["model"] = _resolve_model_section(raw["model"], f"{path.name}:model")
    return from_dict(cls, raw, path.name)


def load_prototype_config(name: str = "prototype_small") -> PrototypeConfig:
    return load_config(name, PrototypeConfig)


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a config dataclass to plain YAML/JSON-serialisable data."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return convert(dataclasses.asdict(config))
