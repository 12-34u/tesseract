"""Phase 2 experiment infrastructure: pending-value guards, frozen matrix, stage/cell execution, no overwrites."""

import json
from dataclasses import replace

import pytest
import torch
import yaml

from phase2.config import (
    PENDING_P1B,
    PendingP1BError,
    Phase2ExperimentConfig,
    StageSpec,
    load_amendment_02,
    load_benchmark,
    load_phase2_experiment,
)
from phase2.experiment import (
    analyze_experiment,
    cell_directory,
    check_preconditions,
    describe_matrix,
    enumerate_matrix,
    run_one_cell,
    run_stage,
)
from utils.config import OptimizerConfig, RunConfig

CPU = torch.device("cpu")


# ----------------------------------------------------------------------------
# Shipped configuration: pending values, frozen matrix, draft amendment
# ----------------------------------------------------------------------------


def test_shipped_config_is_pending_p1b():
    config = load_phase2_experiment()
    assert config.model_base is None and config.max_steps is None
    assert config.pending == ["model_base", "max_steps"]
    with pytest.raises(PendingP1BError):
        config.require_frozen()
    raw = yaml.safe_load(open("configs/phase2/phase2_experiment.yaml"))
    assert raw["model_base"] == PENDING_P1B and raw["max_steps"] == PENDING_P1B


def test_frozen_matrix_matches_approved_plan():
    config = load_phase2_experiment()
    matrix = enumerate_matrix(config, "A5")
    assert {k: len(v) for k, v in matrix.items()} == {
        "t4": 12, "t8": 12, "anchors": 24, "controls": 48, "unrolled": 18, "param_matched": 18, "width_scaled": 6}
    cells = [c for v in matrix.values() for c in v]
    assert len(cells) == 138 and len({c.cell_id for c in cells}) == 138
    assert {c.seed for c in cells} == {0, 1, 2}
    main = {(c.t, c.depth) for c in cells if c.family == "tesseract" and c.task == "main"}
    assert main == {(t, k) for t in (1, 2, 4, 8) for k in (1, 2, 4, 8)}
    controls = {(c.task, c.t, c.depth) for c in cells if c.task != "main"}
    assert controls == {(task, t, k) for task in ("c1_span", "c2_word") for t in (4, 8) for k in (1, 2, 4, 8)}
    for family in ("unrolled", "param_matched"):
        assert {(c.t, c.depth) for c in cells if c.family == family} == {(t, l) for t in (4, 8) for l in (2, 4, 8)}
    assert {(c.t, c.depth) for c in cells if c.family == "width_scaled"} == {(4, 8), (8, 8)}
    assert all(c.task == "main" for c in cells if c.family != "tesseract")
    settings = (config.optimizer, config.batch_size, config.eval_every, config.eval_batch_size, config.vocab_size)
    assert settings == (OptimizerConfig("adamw", 0.001, 0.01), 128, 250, 2048, 60)
    assert "PENDING_P1B" in describe_matrix(config, "A5") and "total: 138 runs" in describe_matrix(config, "A5")


def test_benchmark_unchanged_for_phase2():
    b = load_benchmark()
    assert (b.group, b.n, b.t_values, b.k_values, b.model_seeds, b.data_seed, b.val_size, b.test_size) == (
        "A5", 17, (1, 2, 4, 8), (1, 2, 4, 8), (0, 1, 2), 1729, 2048, 10000)


def test_amendment_02_is_a_draft():
    a = load_amendment_02()
    assert a.status == "draft" and a.frozen is False
    assert [d.id for d in a.decisions] == ["D1", "D2", "D3", "D4", "D5"]
    assert (a.decision_rule.primary_t_values, a.decision_rule.k_low, a.decision_rule.k_high) == ((4, 8), 1, 8)
    assert (a.decision_rule.tau, a.decision_rule.floor_threshold) == (0.9, 0.02)
    assert (a.stage1_review.t, a.stage1_review.floor_threshold, a.stage1_review.ceiling_tau) == (4, 0.02, 0.9)
    with pytest.raises(ValueError, match="frozen flag"):
        replace(a, frozen=True)  # status must change too


@pytest.mark.parametrize("key, value, ok", [("max_steps", 0, False), ("max_steps", "20000", False),
                                            ("max_steps", 20000, True), ("model_base", 7, False),
                                            ("model_base", "phase2/prototype_medium", True)])
def test_pending_value_validation(tmp_path, key, value, ok):
    raw = yaml.safe_load(open("configs/phase2/phase2_experiment.yaml"))
    raw[key] = value
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(raw))
    if ok:
        assert getattr(load_phase2_experiment(str(path)), key) == value
    else:
        with pytest.raises(ValueError):
            load_phase2_experiment(str(path))


# ----------------------------------------------------------------------------
# Guards and execution (tiny model, tiny splits, temporary runs root)
# ----------------------------------------------------------------------------


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERACT_RUNS_DIR", str(tmp_path))
    (tmp_path / "tiny_width.yaml").write_text(yaml.safe_dump({"model": {
        "vocab_size": 16, "max_seq_len": 64, "d_model": 16, "num_heads": 2, "d_ff": 32, "alpha": 0.9, "dropout": 0.0}}))
    config = Phase2ExperimentConfig(
        experiment=RunConfig("exp", str(tmp_path / "exp"), 0, "cpu"), benchmark="phase2/benchmark",
        amendments=("phase2/amendment_01", "phase2/amendment_02_draft"), vocab_size=60,
        optimizer=OptimizerConfig("adamw", 1e-3, 0.01), batch_size=8, eval_every=2, eval_batch_size=64,
        stages=(StageSpec("tiny", "test stage", "tesseract", ("main",), (4,), (1, 2), (0, 1)),),
        model_base=str(tmp_path / "tiny_width.yaml"), max_steps=4)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text("frozen test config\n")
    benchmark = replace(load_benchmark(), val_size=32, test_size=48)
    p1b = tmp_path / "p1b"
    p1b.mkdir()
    (p1b / "run_metadata.json").write_text(json.dumps({"status": "completed", "run_id": "p1b-test"}))
    frozen_02 = replace(load_amendment_02(), status="frozen", frozen=True)
    integrity = {"passed": True, "checks": {"all": True}, "details": {"amended_gates_run_id": "gates-test"}}
    kwargs = dict(benchmark=benchmark, amendment02=frozen_02, integrity=integrity, p1b_run_dir=p1b, log=lambda s: None)
    return tmp_path, config, config_path, kwargs


def test_preconditions_block_training(tiny):
    tmp_path, config, _, kw = tiny
    ok = dict(amendment02=kw["amendment02"], integrity=kw["integrity"], p1b_run_dir=kw["p1b_run_dir"], confirm=True)
    check_preconditions(config, **ok)  # all satisfied
    with pytest.raises(PendingP1BError):
        check_preconditions(replace(config, max_steps=None), **ok)
    with pytest.raises(RuntimeError, match="DRAFT"):
        check_preconditions(config, **{**ok, "amendment02": load_amendment_02()})
    with pytest.raises(RuntimeError, match="integrity"):
        check_preconditions(config, **{**ok, "integrity": {"passed": False, "checks": {"x": False}}})
    (kw["p1b_run_dir"] / "run_metadata.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(RuntimeError, match="P1b has not completed"):
        check_preconditions(config, **ok)
    (kw["p1b_run_dir"] / "run_metadata.json").write_text(json.dumps({"status": "completed"}))
    with pytest.raises(RuntimeError, match="confirmation"):
        check_preconditions(config, **{**ok, "confirm": False})


def test_stage_runs_resumes_and_never_overwrites(tiny):
    tmp_path, config, config_path, kw = tiny
    run_dir = tmp_path / "exp"
    summary = run_stage(config, config_path, CPU, run_dir, "tiny", confirm=True, **kw)
    assert set(summary["cells"].values()) == {"ran"} and len(summary["cells"]) == 4
    assert "stage1_review" in summary and summary["stage1_review"]["basis"].startswith("best-checkpoint validation")
    assert all("test" not in key for row in summary["validation_only"] for key in row)  # validation only
    for cell_id in summary["cells"]:
        width_dir = run_dir / "cells" / "tiny_width" / cell_id
        result = json.loads((width_dir / "result.json").read_text())
        assert result["steps_run"] == 4 and result["stopped_reason"] == "budget"
        assert result["test"] is not None and result["test_final"] is not None  # after training only
        assert result["bptt_on_best_checkpoint"]["passed"] and result["bptt_on_final_checkpoint"]["passed"]
        assert result["block_calls_per_forward"] == result["cell"]["depth"]
        assert result["provenance"]["model_base"] == config.model_base and result["provenance"]["max_steps"] == 4
        assert result["provenance"]["experiment_config_sha256"] and result["provenance"]["p1b_run_id"] == "p1b-test"
        assert (width_dir / "best_checkpoint.pt").is_file() and (width_dir / "final_checkpoint.pt").is_file()

    again = run_stage(config, config_path, CPU, run_dir, "tiny", confirm=True, **kw)
    assert set(again["cells"].values()) == {"skipped_completed"}

    cell = next(c for c in enumerate_matrix(config, "A5")["tiny"])
    meta = cell_directory(run_dir, config.model_base, cell) / "run_metadata.json"
    meta.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(RuntimeError, match="incomplete run"):
        run_stage(config, config_path, CPU, run_dir, "tiny", confirm=True, **kw)


def test_single_cell_is_deterministic(tiny):
    tmp_path, config, config_path, kw = tiny
    cell_id = "tesseract_d2_A5_main_T4_s1"
    runs = []
    for name in ("a", "b"):
        run_one_cell(config, config_path, CPU, tmp_path / name, cell_id, confirm=True, **kw)
        runs.append(tmp_path / name / "cells" / "tiny_width" / cell_id)
    assert (runs[0] / "val_curve.csv").read_text() == (runs[1] / "val_curve.csv").read_text()
    first, second = (torch.load(r / "final_checkpoint.pt", weights_only=True) for r in runs)
    assert all(torch.equal(first[k], second[k]) for k in first)
    with pytest.raises(ValueError, match="not a planned cell"):
        run_one_cell(config, config_path, CPU, tmp_path / "c", "tesseract_d3_A5_main_T4_s0", confirm=True, **kw)


def test_analysis_requires_complete_matrix(tiny):
    tmp_path, config, config_path, kw = tiny
    run_dir = tmp_path / "exp"
    with pytest.raises(RuntimeError, match="not complete"):
        analyze_experiment(config, config_path, CPU, run_dir, benchmark=kw["benchmark"], amendment02=kw["amendment02"],
                           log=lambda s: None)
    run_stage(config, config_path, CPU, run_dir, "tiny", confirm=True, **kw)
    analysis = analyze_experiment(config, config_path, CPU, run_dir, benchmark=kw["benchmark"],
                                  amendment02=kw["amendment02"], log=lambda s: None)
    assert analysis["n_results"] == 4
    assert analysis["checkpoints"]["best"]["decision_rule"]["outcome"] == "incomplete_data"  # no K=8, no controls
    for name in ("analysis.json", "report.md", "fig1_delta_vs_t.png", "fig4_learning_curves.png"):
        assert (run_dir / "analysis" / name).is_file()


def test_cli_list_does_not_train(capsys):
    import scripts.phase2_experiment as cli
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "total: 138 runs" in out and "PENDING_P1B" in out
