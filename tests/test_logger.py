"""ExperimentLogger: CSV output and plots must reflect all logged records."""

import csv

import pytest

from training.logger import ExperimentLogger


def test_gradient_plot_uses_keys_from_all_records(tmp_path):
    logger = ExperimentLogger(tmp_path, "test")
    logger.log({"loss": 1.0, "grad_zL_step_1": 0.5, "grad_zH_step_1": None}, step=1)
    logger.log({"loss": 0.5}, step=2)  # last record not instrumented (as in Crucible)
    assert logger.plot_gradient_norms(tmp_path / "grads.png").is_file()


def test_gradient_plot_without_gradient_data_raises(tmp_path):
    logger = ExperimentLogger(tmp_path, "test")
    logger.log({"loss": 1.0}, step=1)
    with pytest.raises(ValueError):
        logger.plot_gradient_norms(tmp_path / "grads.png")


def test_empty_csv_refused(tmp_path):
    with pytest.raises(ValueError):
        ExperimentLogger(tmp_path, "test").save_csv(tmp_path / "metrics.csv")


def test_csv_keeps_missing_values_missing(tmp_path):
    logger = ExperimentLogger(tmp_path, "test")
    logger.log({"loss": 0.0, "grad_zL_step_2": 0.1, "grad_zL_step_10": None}, step=1)
    logger.log({"loss": 0.25}, step=2)
    path = logger.save_csv(tmp_path / "metrics.csv")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["step", "grad_zL_step_2", "grad_zL_step_10", "loss"]
    assert rows[0]["loss"] == "0.0"  # zero stays zero
    assert rows[0]["grad_zL_step_10"] == ""  # None → empty, not 0
    assert rows[1]["grad_zL_step_2"] == ""
