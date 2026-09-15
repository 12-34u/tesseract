"""End-to-end smoke tests: each experiment runs on a tiny model and produces
consistent, non-stale artifacts with honest verdicts and exit codes."""

import csv
import json
from dataclasses import replace

import pytest
import torch
import yaml

from experiments import cellular_automaton, crucible, k_scaling
from utils.config import CellularAutomatonConfig, CrucibleConfig, KScalingConfig, config_to_dict, load_config
from utils.run_artifacts import METADATA_FILENAME, RunRecorder

CPU = torch.device("cpu")


def tiny(model_config):
    return replace(model_config, d_model=16, num_heads=2, d_ff=32)


def metadata(run_dir):
    return json.loads((run_dir / METADATA_FILENAME).read_text())


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def tiny_crucible(output_dir):
    config = load_config("crucible", CrucibleConfig)
    return replace(
        config,
        experiment=replace(config.experiment, output_dir=str(output_dir), device="cpu"),
        model=tiny(config.model),
        training=replace(config.training, max_steps=3, log_every=1),
    )


def test_crucible_reports_failure_through_exit_code(tmp_path):
    run_dir = tmp_path / "crucible"
    config_path = tmp_path / "crucible.yaml"
    config_path.write_text(yaml.safe_dump(config_to_dict(tiny_crucible(run_dir))))

    assert crucible.main(["--config", str(config_path)]) == 1  # 3 steps cannot converge

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["verdict"] == "FAIL"
    assert summary["checks"]["loss_converged"] is False
    assert summary["checks"]["bptt_gradients_ok"] is True
    assert summary["training_steps"] == 3
    for name in crucible.ARTIFACTS:
        assert (run_dir / name).exists(), name

    meta = metadata(run_dir)
    assert (meta["status"], meta["verdict"], meta["seed"]) == ("completed", "FAIL", 42)
    assert meta["device"]["type"] == "cpu"

    bptt = json.loads((run_dir / "bptt_verification.json").read_text())
    assert len(bptt["steps"]) == summary["k"]
    assert bptt["steps"][-1]["grad_z_H_norm"] is None


def test_crucible_is_reproducible(tmp_path):
    for name in ("a", "b"):
        crucible.run_crucible(tiny_crucible(tmp_path / name), tmp_path / "unused.yaml", CPU, tmp_path / name)
    assert (tmp_path / "a" / "metrics.csv").read_text() == (tmp_path / "b" / "metrics.csv").read_text()


def test_k_scaling_measures_calls_and_invariance(tmp_path):
    config = load_config("k_scaling", KScalingConfig)
    config = replace(
        config,
        model=tiny(config.model),
        k_values=(1, 3),
        benchmark=replace(config.benchmark, global_warmup_k=1, global_warmup_runs=1, warmup_runs=1, timing_runs=2),
        training=replace(config.training, max_steps=2, log_every=1),
    )
    results = k_scaling.run_k_scaling(config, tmp_path / "unused.yaml", CPU, tmp_path / "k")
    assert results["verdict"] == "PASS"

    rows = read_csv(tmp_path / "k" / "results.csv")
    assert [int(r["recursive_calls"]) for r in rows] == [1, 3]
    assert len({r["parameter_count"] for r in rows}) == 1
    assert [r["training_steps"] for r in rows] == ["2", "2"]


def test_cellular_automaton_smoke(tmp_path):
    config = load_config("cellular_automaton", CellularAutomatonConfig)
    config = replace(
        config,
        model=tiny(config.model),
        k_values=(1, 2),
        data=replace(config.data, seq_len=16, t_values=(1, 3), num_train=16, num_val=8),
        training=replace(config.training, batch_size=8, max_steps=3, eval_every=2, eval_first_steps=1),
    )
    run_dir = tmp_path / "ca"
    results = cellular_automaton.run_cellular_automaton(config, tmp_path / "unused.yaml", CPU, run_dir)
    assert results["verdict"] == "COMPLETED"

    rows = read_csv(run_dir / "results.csv")
    assert [(r["T"], r["K"]) for r in rows] == [("1", "1"), ("1", "2"), ("3", "1"), ("3", "2")]
    for row in rows:
        assert row["val_exact_match"] != "" and row["eval_step"] == row["training_steps"]
        assert (run_dir / "predictions" / f"T{row['T']}_K{row['K']}.txt").is_file()

    summary = (run_dir / "summary.txt").read_text()
    assert "T=1: yes" in summary and "T=3: no" in summary
    assert metadata(run_dir)["status"] == "completed"


def test_stale_artifacts_removed_and_other_files_kept(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "predictions" / "old.txt").write_text("stale")
    (run_dir / "summary.json").write_text("stale")
    (run_dir / "notes.txt").write_text("keep")

    config = load_config("crucible", CrucibleConfig)
    with RunRecorder(run_dir, "demo", config, tmp_path / "c.yaml", CPU, ["summary.json", "predictions"]) as run:
        assert not (run_dir / "summary.json").exists()
        assert not (run_dir / "predictions").exists()
        assert metadata(run_dir)["status"] == "running"
        with pytest.raises(ValueError):
            run.path("undeclared.txt")
        run.finish("COMPLETED", {"value": float("nan")})

    assert (run_dir / "notes.txt").read_text() == "keep"
    meta = metadata(run_dir)
    assert meta["status"] == "completed" and meta["results"] == {"value": None}
    assert (run_dir / "config.yaml").is_file()


def test_run_recorder_marks_crashed_runs_failed(tmp_path):
    config = load_config("crucible", CrucibleConfig)
    with pytest.raises(RuntimeError, match="boom"):
        with RunRecorder(tmp_path, "demo", config, tmp_path / "c.yaml", CPU, []):
            raise RuntimeError("boom")
    meta = metadata(tmp_path)
    assert meta["status"] == "failed" and "boom" in meta["error"]
