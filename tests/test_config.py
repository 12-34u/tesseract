"""Configuration loading, strict validation and device resolution."""

from dataclasses import replace

import pytest
import torch
import yaml

from models.tesseract import TesseractModel
from utils.config import (
    CellularAutomatonConfig,
    ConfigError,
    CrucibleConfig,
    KScalingConfig,
    ModelConfig,
    config_to_dict,
    load_config,
    load_prototype_config,
)
from utils.device import resolve_device
from utils.param_count import count_parameters

EXPERIMENT_CONFIGS = [
    ("crucible", CrucibleConfig),
    ("k_scaling", KScalingConfig),
    ("cellular_automaton", CellularAutomatonConfig),
]


def analytic_parameter_count(m: ModelConfig) -> int:
    d, v, f = m.d_model, m.vocab_size, m.d_ff
    embeddings = v * d + m.max_seq_len * d
    block = 4 * d + (3 * d * d + 3 * d) + (d * d + d) + (d * f + f) + (f * d + d)
    initial_states = 2 * d
    head = d * v + v
    return embeddings + block + initial_states + head


@pytest.mark.parametrize("name, cls", EXPERIMENT_CONFIGS)
def test_shipped_experiment_configs_load(name, cls):
    config = load_config(name, cls)
    assert config.experiment.name == name
    assert config.experiment.device == "auto"


@pytest.mark.parametrize("name, cls", EXPERIMENT_CONFIGS)
def test_experiments_use_the_prototype_architecture(name, cls):
    prototype = load_prototype_config().model
    model = load_config(name, cls).model
    assert replace(model, vocab_size=prototype.vocab_size) == prototype


@pytest.mark.parametrize("name, cls", EXPERIMENT_CONFIGS)
def test_measured_parameter_count_matches_formula(name, cls):
    model_config = load_config(name, cls).model
    expected = analytic_parameter_count(model_config)
    for k in (1, 2, 4, 8):
        assert count_parameters(TesseractModel.from_config(model_config, k))["trainable"] == expected


def _write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def crucible_data():
    return config_to_dict(load_config("crucible", CrucibleConfig))


def test_round_trip(tmp_path, crucible_data):
    assert load_config(_write(tmp_path, crucible_data), CrucibleConfig) == load_config("crucible", CrucibleConfig)


def test_unknown_key_rejected(tmp_path, crucible_data):
    crucible_data["training"]["learning_rat"] = 0.1
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, crucible_data), CrucibleConfig)


def test_missing_key_rejected(tmp_path, crucible_data):
    del crucible_data["k"]
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(_write(tmp_path, crucible_data), CrucibleConfig)


def test_yaml_scientific_notation_string_rejected(tmp_path, crucible_data):
    # PyYAML parses `3e-4` (no decimal point) as a string, not a float.
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(crucible_data).replace("learning_rate: 0.0003", "learning_rate: 3e-4"))
    with pytest.raises(ConfigError, match="expected float"):
        load_config(path, CrucibleConfig)


def test_bool_is_not_an_int(tmp_path, crucible_data):
    crucible_data["k"] = True
    with pytest.raises(ConfigError, match="expected int"):
        load_config(_write(tmp_path, crucible_data), CrucibleConfig)


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("model", "num_heads", 5, "divisible"),
        ("model", "alpha", 1.5, "alpha"),
        ("data", "seq_len", 128, "max_seq_len"),
        ("pass_criteria", "min_exact_match", "90", "expected float"),
    ],
)
def test_invalid_values_rejected(tmp_path, crucible_data, section, key, value, message):
    crucible_data[section][key] = value
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, crucible_data), CrucibleConfig)


def test_model_override_must_exist_in_base(tmp_path, crucible_data):
    crucible_data["model"] = {"base": "prototype_small", "overrides": {"d_modle": 64}}
    with pytest.raises(ConfigError, match="do not exist in base"):
        load_config(_write(tmp_path, crucible_data), CrucibleConfig)


def test_ca_config_requires_binary_vocab(tmp_path):
    data = config_to_dict(load_config("cellular_automaton", CellularAutomatonConfig))
    data["model"]["vocab_size"] = 16
    with pytest.raises(ConfigError, match="vocab_size must be 2"):
        load_config(_write(tmp_path, data), CellularAutomatonConfig)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", CrucibleConfig)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def test_auto_device_matches_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


def test_cpu_device():
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("bad", ["mps", "not_a_device"])
def test_unsupported_device_rejected(bad):
    with pytest.raises(ValueError):
        resolve_device(bad)


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without CUDA")
def test_explicit_cuda_without_cuda_is_an_error():
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")
