"""P1b capacity diagnostic: width-only model change, settings identical to P1, no overwrites, no test data."""

import json
from dataclasses import asdict, replace

import pytest
import torch
import yaml

from phase2.config import (
    CapacityDiagnosticConfig,
    GatesRunConfig,
    ModelVariant,
    PilotConfig,
    load_benchmark,
    resolve_model_variant,
)
from phase2.models import build_model
from phase2.runner import run_capacity_diagnostic, run_verification_and_gates
from utils.config import OptimizerConfig, RunConfig, load_config

CPU = torch.device("cpu")


def test_prototype_medium_differs_from_small_only_in_width():
    small = resolve_model_variant(ModelVariant("small", "prototype_small"), 60)
    medium = resolve_model_variant(ModelVariant("medium", "phase2/prototype_medium"), 60)
    diff = {k: (v, asdict(medium)[k]) for k, v in asdict(small).items() if asdict(medium)[k] != v}
    assert diff == {"d_model": (128, 256), "num_heads": (4, 8), "d_ff": (512, 1024)}
    assert small.d_model // small.num_heads == medium.d_model // medium.num_heads == 32
    for k in (1, 8):
        assert build_model("tesseract", k, small, 17).parameter_count == 222_140
        assert build_model("tesseract", k, medium, 17).parameter_count == 837_436


def test_p1b_settings_match_pilot_p1():
    p1b = load_config("phase2/capacity_diagnostic_p1b", CapacityDiagnosticConfig)
    p1 = load_config("phase2/pilot_p1", PilotConfig)
    assert (p1b.optimizer, p1b.batch_size, p1b.eval_every, p1b.eval_batch_size, p1b.floor_threshold,
            p1b.experiment.seed, p1b.benchmark) == (p1.optimizer, p1.batch_size, p1.eval_every, p1.eval_batch_size,
                                                   p1.floor_threshold, p1.experiment.seed, p1.benchmark)
    assert resolve_model_variant(p1b.models[0], p1b.vocab_size) == p1.model  # small == the P1 model
    assert (p1b.t, p1b.k_values, p1b.max_steps) == (8, (1, 8), 20000)
    assert [m.name for m in p1b.models] == ["small", "medium"]
    assert p1b.experiment.output_dir != p1.experiment.output_dir


def test_duplicate_variant_names_rejected():
    p1b = load_config("phase2/capacity_diagnostic_p1b", CapacityDiagnosticConfig)
    with pytest.raises(ValueError, match="unique"):
        replace(p1b, models=(p1b.models[0], p1b.models[0]))


@pytest.fixture
def tiny_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERACT_RUNS_DIR", str(tmp_path))
    b = load_benchmark()
    gates = replace(b.gates, uniformity_states=500, uniformity_tv_max=0.2, cycle_states=4, cycle_steps=16,
                    sensitivity_states=300, shortcut_states=300)
    benchmark = replace(b, t_values=(1,), interpolation_t_values=(3,), ablation_groups=(), tasks=("main",),
                        val_size=32, test_size=48, length_test_size=8, gates=gates)
    gates_dir = tmp_path / "phase2" / "gates"
    run_verification_and_gates(GatesRunConfig(RunConfig("g", str(gates_dir), 7, "cpu"), "phase2/benchmark"),
                               tmp_path / "g.yaml", gates_dir, benchmark, CPU, log=lambda s: None)
    for name, d_model in (("tiny_a", 16), ("tiny_b", 32)):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({"model": {
            "vocab_size": 16, "max_seq_len": 64, "d_model": d_model, "num_heads": 2, "d_ff": 2 * d_model,
            "alpha": 0.9, "dropout": 0.0}}))
    config = CapacityDiagnosticConfig(
        experiment=RunConfig("diag", str(tmp_path / "diag"), 0, "cpu"), benchmark="phase2/benchmark", vocab_size=60,
        models=(ModelVariant("small", str(tmp_path / "tiny_a.yaml")), ModelVariant("medium", str(tmp_path / "tiny_b.yaml"))),
        t=1, k_values=(1, 2), optimizer=OptimizerConfig("adamw", 1e-3, 0.01), batch_size=8, eval_every=2,
        eval_batch_size=64, max_steps=4, floor_threshold=0.02, reference_pilot_output_dir=str(tmp_path / "p1"))
    return tmp_path, benchmark, gates_dir, config


def test_capacity_diagnostic_end_to_end(tiny_setup):
    tmp_path, benchmark, gates_dir, config = tiny_setup
    report = run_capacity_diagnostic(config, tmp_path / "c.yaml", CPU, tmp_path / "diag", benchmark=benchmark,
                                     gates_run_dir=gates_dir, log=lambda s: None)
    assert set(report["runs"]) == {"small", "medium"}
    for name in ("small", "medium"):
        for k in ("1", "2"):
            r = report["runs"][name][k]
            assert r["steps_run"] == 4 and r["stopped_reason"] == "budget" and r["test_evaluated"] is False
            assert r["block_calls_per_forward"] == int(k) and len(r["curve"]) == 2
            assert {"at_floor_best", "at_floor_final"} <= r.keys()
            result = json.loads((tmp_path / "diag" / "cells" / name / r["cell_id"] / "result.json").read_text())
            assert result["test"] is None and result["checkpoints"]["final"]["sha256"]
    within = report["comparisons"]["within_width_K_high_minus_K_low"]
    assert within["small"]["parameter_count_equal"] and within["medium"]["parameter_count_equal"]
    assert report["settings"]["test_split_evaluated"] is False
    with pytest.raises(RuntimeError, match="preserved"):
        run_capacity_diagnostic(config, tmp_path / "c.yaml", CPU, tmp_path / "diag", benchmark=benchmark,
                                gates_run_dir=gates_dir, log=lambda s: None)
