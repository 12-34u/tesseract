"""Fixed-step protocol, sanity baselines, cell/pilot/grid runners (tiny sizes, temp runs root)."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from phase2.config import (
    CalibrationConfig,
    CellRunConfig,
    CellSpec,
    GatesRunConfig,
    PilotCell,
    PilotConfig,
    TrainingConfig,
    load_benchmark,
)
from phase2.data import SplitStore, TrainingStream, generate_split
from phase2.models import build_model
from phase2.runner import (
    require_gates_passed,
    run_cell,
    run_pilot,
    run_verification_and_gates,
)
from phase2.sanity import sanity_baselines
from phase2.training import evaluate_split, train_fixed_steps
from utils.config import ModelConfig, OptimizerConfig, RunConfig, load_config

CPU = torch.device("cpu")
TINY_MODEL = ModelConfig(vocab_size=60, max_seq_len=64, d_model=16, num_heads=2, d_ff=32, alpha=0.9, dropout=0.0)
OPT = OptimizerConfig("adamw", 1e-3, 0.01)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERACT_RUNS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def tiny_benchmark():
    b = load_benchmark()
    gates = replace(b.gates, uniformity_states=500, uniformity_tv_max=0.2, cycle_states=4, cycle_steps=16,
                    sensitivity_states=300, shortcut_states=300)
    # T=1 only: G2 does not apply there, so the gate run passes and the pipeline machinery can be exercised.
    # (The real benchmark's gate verdict is decided by scripts/phase2_gates.py, not by these tests.)
    return replace(b, t_values=(1,), k_values=(1, 2), interpolation_t_values=(3,), ablation_groups=(),
                   tasks=("main",), val_size=32, test_size=48, length_test_size=8, model_seeds=(0,), gates=gates)


def experiment(tmp_path, name="t"):
    return RunConfig(name=name, output_dir=str(tmp_path / name), seed=0, device="cpu")


# ----------------------------------------------------------------------------
# Training protocol
# ----------------------------------------------------------------------------


def small_splits():
    return (generate_split("A5", "main", "val", 1, 17, 32, 1729), generate_split("A5", "main", "test", 1, 17, 48, 1729))


def test_fixed_budget_runs_every_step_and_evaluates_best_checkpoint():
    val, test = small_splits()
    torch.manual_seed(0)
    model = build_model("tesseract", 1, TINY_MODEL, 17).model
    stream = TrainingStream("A5", "main", 1, 17, 16, 1729, 0, excluded_inputs=[val.inputs, test.inputs])
    outcome = train_fixed_steps(model, stream, val, test, TrainingConfig(OPT, 16, 7, 3, 64), CPU, 60, log=lambda s: None)
    assert outcome.steps_run == 7 and outcome.stopped_reason == "budget" and stream.batches_drawn == 7
    assert [row["step"] for row in outcome.history] == [3, 6, 7]
    assert outcome.best_step in (3, 6, 7)
    # The model left behind is the best checkpoint, and test was evaluated on it.
    assert evaluate_split(model, val, CPU, 60, 64) == outcome.best_val
    assert outcome.test.num_examples == 48 and outcome.test_predictions.shape == (48, 17)
    assert outcome.best_val_loss > 0 and all("val_loss" in row for row in outcome.history)
    assert outcome.gradient_norm_stats["non_finite_steps"] == 0 and outcome.gradient_norm_stats["max"] > 0
    assert outcome.best_state is not None and outcome.final_state is not None


def test_test_split_is_optional_and_never_used_for_selection():
    val, _ = small_splits()
    model = build_model("tesseract", 1, TINY_MODEL, 17).model
    outcome = train_fixed_steps(model, TrainingStream("A5", "main", 1, 17, 8, 1729, 0), val, None,
                                TrainingConfig(OPT, 8, 4, 2, 64), CPU, 60, log=lambda s: None)
    assert outcome.test is None and outcome.test_predictions is None and outcome.best_step in (2, 4)


def test_stop_condition_is_only_used_when_given():
    val, test = small_splits()
    model = build_model("tesseract", 1, TINY_MODEL, 17).model
    stream = TrainingStream("A5", "main", 1, 17, 8, 1729, 0)
    outcome = train_fixed_steps(model, stream, val, test, TrainingConfig(OPT, 8, 50, 2, 64), CPU, 60,
                                stop_condition=lambda v: True, log=lambda s: None)
    assert outcome.stopped_reason == "stop_condition" and outcome.steps_run == 2


def test_non_finite_loss_is_reported():
    val, test = small_splits()
    model = build_model("tesseract", 1, TINY_MODEL, 17).model
    with torch.no_grad():
        model.output_head.bias.fill_(float("nan"))
    outcome = train_fixed_steps(model, TrainingStream("A5", "main", 1, 17, 8, 1729, 0), val, test,
                                TrainingConfig(OPT, 8, 10, 5, 64), CPU, 60, log=lambda s: None)
    assert outcome.stopped_reason == "non_finite_loss" and outcome.steps_run == 1 and outcome.test is None


# ----------------------------------------------------------------------------
# Sanity baselines
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("group, task, t", [("A5", "main", 8), ("A5", "c2_word", 4), ("Z60", "c1_span", 2)])
def test_sanity_baselines(group, task, t):
    split = generate_split(group, task, "test", t, 17, 200, 1729)
    report = sanity_baselines(split, seed=0)
    assert report["oracle_independent_simulator"]["exact_match"] == 100.0
    assert report["chance_expected"]["token_accuracy"] == pytest.approx(100 / 60)
    copy_expected = (split.inputs == split.targets).mean() * 100
    assert report["copy_input"]["token_accuracy"] == pytest.approx(copy_expected)
    assert abs(report["chance_sampled"]["token_accuracy"] - 100 / 60) < 1.0


# ----------------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------------


def test_gates_run_then_cell_run(runs, tiny_benchmark):
    gates_dir = runs / "phase2" / "gates"
    gates_cfg = GatesRunConfig(experiment=RunConfig("phase2_gates", str(gates_dir), 7, "cpu"), benchmark="phase2/benchmark")
    report = run_verification_and_gates(gates_cfg, runs / "g.yaml", gates_dir, tiny_benchmark, CPU, log=lambda s: None)
    assert report["passed"]
    assert require_gates_passed(gates_dir)["verdict"] == "PASS"
    assert (runs / "phase2" / "splits" / "manifest.json").is_file()

    cell = CellSpec("unrolled", 2, "A5", "main", 2, 0)
    cell_cfg = CellRunConfig(experiment(runs, "cell"), TINY_MODEL, tiny_benchmark, TrainingConfig(OPT, 8, 4, 2, 64), cell)
    result = run_cell(cell_cfg, runs / "c.yaml", CPU, runs / "cell", log=lambda s: None)
    assert result["steps_run"] == 4 and result["stopped_reason"] == "budget"
    assert 0.0 <= result["window_coverage"] <= 1.0 and result["window_coverage_reference"] == "test inputs"
    assert result["block_calls_per_forward"] == 2 and result["transformer_block_instances"] == 2
    assert result["best_val_loss"] > 0 and result["non_finite"]["final_parameters_finite"]
    assert result["bptt_on_best_checkpoint"]["passed"] and len(result["bptt_on_best_checkpoint"]["steps"]) == 2
    assert result["bptt_on_final_checkpoint"]["passed"]
    assert result["test"]["num_examples"] == 48 and result["test_final"]["num_examples"] == 48  # after training only
    assert result["test_final_bootstrap_ci95"] is not None and result["model"]["head_dim"] == 8
    for name in ("best", "final"):
        assert (runs / "cell" / result["checkpoints"][name]["file"]).is_file()
    assert json.loads((runs / "cell" / "run_metadata.json").read_text())["verdict"] == "COMPLETED"
    # the cell used the frozen, checksummed splits
    store = SplitStore(runs / "phase2" / "splits", tiny_benchmark.data_seed)
    assert result["split_sha256"]["test"] == store.get("A5", "main", "test", 2, 17, 48).sha256


def test_training_refuses_without_passed_gates(runs, tiny_benchmark):
    with pytest.raises(RuntimeError, match="No gate run"):
        require_gates_passed(runs / "phase2" / "gates")
    gates_dir = runs / "phase2" / "gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / "run_metadata.json").write_text(json.dumps({"status": "completed", "verdict": "FAIL"}))
    with pytest.raises(RuntimeError, match="have not passed"):
        require_gates_passed(gates_dir)


def test_pilot_end_to_end(runs, tiny_benchmark):
    gates_dir = runs / "phase2" / "gates"
    run_verification_and_gates(GatesRunConfig(RunConfig("g", str(gates_dir), 7, "cpu"), "phase2/benchmark"),
                               runs / "g.yaml", gates_dir, tiny_benchmark, CPU, log=lambda s: None)
    pilot = PilotConfig(
        experiment=experiment(runs, "pilot"), model=TINY_MODEL, benchmark="phase2/benchmark", optimizer=OPT,
        batch_size=8, eval_every=2, eval_batch_size=64,
        calibration=CalibrationConfig("tesseract", 1, "main", 1, max_steps=6, converge_exact_match=0.0, budget_multiplier=2),
        floor_cells=(PilotCell("tesseract", 2, "main", 2),), floor_threshold=0.02,
    )
    report = run_pilot(pilot, runs / "p.yaml", CPU, runs / "pilot", benchmark=tiny_benchmark, gates_run_dir=gates_dir,
                       log=lambda s: None)
    assert report["stage_a"]["converged"] and report["stage_a"]["steps_to_criterion"] == 2
    assert report["fixed_step_budget"] == 4  # exactly 2 × steps-to-criterion
    stage_b = report["stage_b"]
    assert [r["steps_run"] for r in stage_b] == [4] and stage_b[0]["stopped_reason"] == "budget"  # no early stopping
    assert stage_b[0]["block_calls_per_forward"] == 2 and stage_b[0]["test_evaluated"] is False
    assert "at_floor" in stage_b[0] and stage_b[0]["best_val"]["per_position_accuracy"]
    cell_result = json.loads((runs / "pilot" / "cells" / stage_b[0]["cell_id"] / "result.json").read_text())
    assert cell_result["test"] is None and cell_result["test_final"] is None  # P1 never evaluates the test split
    assert report["compute_environment"]["cpu_only"] == (not torch.cuda.is_available())
    assert report["compute_environment"]["git"] is not None
    assert (runs / "pilot" / "pilot_report.json").is_file() and (runs / "pilot" / "summary.txt").is_file()
    with pytest.raises(RuntimeError, match="already exists"):  # earlier pilot runs are preserved
        run_pilot(pilot, runs / "p.yaml", CPU, runs / "pilot", benchmark=tiny_benchmark, gates_run_dir=gates_dir,
                  log=lambda s: None)


def test_shipped_phase2_configs_load():
    pilot = load_config("phase2/pilot_p1", PilotConfig)
    assert pilot.model.vocab_size == 60 and pilot.calibration.t == 1 and pilot.calibration.depth == 1
    assert [(c.depth, c.t) for c in pilot.floor_cells] == [(8, 1), (1, 8), (8, 8)]
    assert load_config("phase2/gates", GatesRunConfig).experiment.output_dir == "phase2/gates"
