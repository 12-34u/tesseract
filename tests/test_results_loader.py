"""Dashboard backend loader: values come only from artifacts; missing stays missing."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "results_loader_under_test", REPO_ROOT / "app" / "backend" / "services" / "results_loader.py"
)
results_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(results_loader)
ResultsLoader = results_loader.ResultsLoader

K_HEADER = "K,parameter_count,recursive_calls,latency_mean_ms,latency_median_ms,latency_std_ms\n"
CA_HEADER = ("T,K,parameter_count,initial_loss,final_loss,train_token_accuracy,train_exact_match,"
             "val_token_accuracy,val_exact_match,best_val_exact_match,training_steps\n")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def runs(tmp_path):
    return tmp_path


def test_empty_runs_directory_reports_missing_not_defaults(runs):
    loader = ResultsLoader(runs_root=runs)
    for payload in (loader.get_crucible(), loader.get_k_scaling(), loader.get_cellular_automaton()):
        assert payload["status"] == "missing"
    summary = loader.get_summary()
    assert summary["parameter_count"] is None and summary["model"] is None
    assert summary["k_tested"] == [] and summary["device"] is None
    assert summary["findings"] == []


def test_zero_is_preserved_and_empty_is_null(runs):
    write(runs / "cellular_automaton" / "results.csv", CA_HEADER + "1,4,207234,2.7,0.004,99.8,96.0,79.4,0.0,,1085\n")
    payload = ResultsLoader(runs_root=runs).get_cellular_automaton()
    assert payload["status"] == "available"
    row = payload["results"][0]
    assert row["val_exact_match"] == 0.0
    assert row["best_val_exact_match"] is None
    assert row["parameter_count"] == 207234 and isinstance(row["parameter_count"], int)
    assert payload["run"] is None and "predate run metadata" in payload["provenance_note"]


@pytest.mark.parametrize(
    "body, message",
    [
        ("1,210832,1,abc,0.8,0.1\n", "not a number"),
        ("1,210832,1,0.9\n", "wrong number of fields"),
        ("", "no data rows"),
    ],
)
def test_malformed_csv_is_an_explicit_error(runs, body, message):
    write(runs / "k_scaling" / "results.csv", K_HEADER + body)
    payload = ResultsLoader(runs_root=runs).get_k_scaling()
    assert payload["status"] == "error" and message in payload["message"]


def test_missing_required_column_is_an_error(runs):
    write(runs / "k_scaling" / "results.csv", "K,parameter_count\n1,5\n")
    payload = ResultsLoader(runs_root=runs).get_k_scaling()
    assert payload["status"] == "error" and "missing required columns" in payload["message"]


def test_summary_is_derived_from_artifacts(runs):
    write(runs / "k_scaling" / "results.csv", K_HEADER + "1,12345,1,1.0,1.1,0.1\n2,12345,2,2.0,2.2,0.1\n")
    write(runs / "k_scaling" / "run_metadata.json", json.dumps({"status": "completed", "verdict": "PASS",
                                                                "device": {"type": "cpu"}}))
    summary = ResultsLoader(runs_root=runs).get_summary()
    assert summary["parameter_count"] == 12345 and summary["k_tested"] == [1, 2] and summary["device"] == "cpu"
    texts = " ".join(f["text"] for f in summary["findings"])
    assert "12,345" in texts and "1.10 ms at K=1" in texts and "2.20 ms at K=2" in texts


def test_non_finite_json_values_become_null(runs):
    write(runs / "crucible_k04" / "metrics.csv", "step,loss,token_accuracy,exact_match_accuracy\n1,NaN,0.0,0.0\n")
    write(runs / "crucible_k04" / "summary.json", '{"final_loss": NaN}')
    payload = ResultsLoader(runs_root=runs).get_crucible()
    assert payload["status"] == "available"
    assert payload["metrics"][0]["loss"] is None
    assert payload["summary"] == {"final_loss": None}


def test_raw_results_only_serve_known_files(runs):
    write(runs / "k_scaling" / "results.csv", K_HEADER)
    raw = ResultsLoader(runs_root=runs).get_raw_results()
    assert set(raw) == {"crucible", "k_scaling", "cellular_automaton"}
    assert raw["k_scaling"]["results.csv"] == K_HEADER
    assert raw["crucible"]["metrics.csv"] is None
