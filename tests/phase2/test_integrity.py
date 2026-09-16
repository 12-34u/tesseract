"""Benchmark integrity check: the recorded gate history and frozen splits must be intact."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from phase2.config import AmendedGatesRunConfig, GatesRunConfig, load_amendment, load_benchmark
from phase2.integrity import check_benchmark_integrity
from phase2.runner import run_amended_gates, run_verification_and_gates
from utils.config import RunConfig

CPU = torch.device("cpu")


@pytest.fixture
def gate_history(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERACT_RUNS_DIR", str(tmp_path))
    b = load_benchmark()
    gates = replace(b.gates, uniformity_states=500, uniformity_tv_max=0.2, cycle_states=4, cycle_steps=16,
                    sensitivity_states=400, shortcut_states=2000)
    benchmark = replace(b, t_values=(1, 2, 4), interpolation_t_values=(3,), ablation_groups=(), tasks=("main",),
                        val_size=16, test_size=400, length_test_size=8, gates=gates)
    amendment = replace(load_amendment(), primary_depth_t_values=(4,))
    original = tmp_path / "phase2" / "gates"
    run_verification_and_gates(GatesRunConfig(RunConfig("g", str(original), 7, "cpu"), "phase2/benchmark"),
                               tmp_path / "g.yaml", original, benchmark, CPU, log=lambda s: None)
    amended = tmp_path / "phase2" / "gates_amendment01"
    run_amended_gates(AmendedGatesRunConfig(RunConfig("a", str(amended), 7, "cpu"), "phase2/benchmark", "phase2/amendment_01"),
                      tmp_path / "a.yaml", amended, benchmark, amendment, CPU, training_dirs=[], log=lambda s: None)
    return tmp_path, benchmark, amendment, original, amended


def check(ctx):
    tmp_path, benchmark, amendment, _, amended = ctx
    return check_benchmark_integrity(benchmark, amendment, amended, tmp_path / "phase2" / "splits")


def test_intact_history_passes(gate_history):
    report = check(gate_history)
    assert report["passed"], {k: v for k, v in report["checks"].items() if not v}
    assert report["checks"]["original_gate_run_FAIL_preserved"]
    assert report["checks"]["original_G2_failures_exactly_the_documented_diagnostic_T"]
    assert report["details"]["splits_checked"] > 0


def test_rewritten_original_verdict_is_detected(gate_history):
    original = gate_history[3]
    meta = json.loads((original / "run_metadata.json").read_text())
    (original / "run_metadata.json").write_text(json.dumps({**meta, "verdict": "PASS"}))
    report = check(gate_history)
    assert not report["passed"] and not report["checks"]["original_gate_run_FAIL_preserved"]


def test_modified_original_report_is_detected(gate_history):
    original = gate_history[3]
    report_path = original / "gates_report.json"
    report_path.write_text(report_path.read_text() + " ")
    assert not check(gate_history)["checks"]["amended_run_references_unchanged_original_report"]


def test_tampered_split_is_detected(gate_history):
    splits = gate_history[0] / "phase2" / "splits"
    path = next(p for p in sorted(splits.glob("*.npz")) if "test" in p.name)
    with np.load(path) as f:
        inputs, targets = f["inputs"], f["targets"].copy()
    targets[0, 0] = (targets[0, 0] + 1) % 60
    np.savez_compressed(path, inputs=inputs, targets=targets)
    report = check(gate_history)
    assert not report["checks"]["frozen_split_checksums_verified"] and report["details"]["split_mismatches"]


def test_missing_gate_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERACT_RUNS_DIR", str(tmp_path))
    report = check_benchmark_integrity(load_benchmark(), load_amendment(), tmp_path / "nope", tmp_path / "splits")
    assert report["passed"] is False and report["details"]["missing"]
