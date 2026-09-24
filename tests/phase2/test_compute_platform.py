"""Amendment 02a: determinism settings, cross-device tolerance, and the Phase 1 freeze."""

import json
from pathlib import Path

import pytest
import torch

from phase2.compute import compute_provenance, configure_determinism, describe_compute
from phase2.config import load_amendment_02a
from phase2.crosscheck import compare_level_b, summarise
from phase2.integrity import check_cpu_gpu_crosscheck, check_phase1_frozen

AMENDMENT = load_amendment_02a()
TOLERANCE = AMENDMENT.cross_device_tolerance


# ---------------------------------------------------------------------------
# The amendment itself
# ---------------------------------------------------------------------------


def test_amendment_02a_is_a_draft_and_leaves_amendment_02_pending():
    assert AMENDMENT.status == "draft" and AMENDMENT.frozen is False
    assert AMENDMENT.recorded_before_any_phase2_training_on_gpu is True
    assert {d.id for d in AMENDMENT.decisions} == {"P1", "P2", "P3", "P4", "P5"}


def test_level_a_is_blocking_and_level_b_is_not():
    """The two levels fail for different reasons, so they cannot share a severity."""
    assert TOLERANCE.level_a_kernel_agreement.blocking is True
    assert TOLERANCE.level_b_end_to_end.blocking is False


def test_level_a_tolerance_is_tighter_than_level_b():
    assert TOLERANCE.level_a_kernel_agreement.max_abs_loss_difference_nats < \
        TOLERANCE.level_b_end_to_end.max_abs_final_val_loss_difference_nats


# ---------------------------------------------------------------------------
# Determinism configuration
# ---------------------------------------------------------------------------


def test_configure_determinism_disables_tf32_and_reports_what_it_set():
    """TF32 would make the GPU quietly less precise than the CPU it is compared to."""
    record = configure_determinism(torch.device("cpu"))
    assert record["use_deterministic_algorithms"] is True
    assert record["matmul_allow_tf32"] is False and record["cudnn_allow_tf32"] is False
    assert record["float32_matmul_precision"] == "highest"
    assert record["cudnn_benchmark"] is False
    assert record["cublas_workspace_config"]["value"] == ":4096:8"


def test_provenance_records_the_versions_a_reader_would_need():
    provenance = compute_provenance(torch.device("cpu"), configure_determinism(torch.device("cpu")))
    for key in ("torch", "numpy", "python", "platform", "device_type"):
        assert provenance["compute"][key]
    assert provenance["amendment"] == "amendment_02a"
    assert "same-hardware" in provenance["reproducibility"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_metadata_includes_driver_and_capability():
    info = describe_compute(torch.device("cuda"))
    for key in ("cuda_device_name", "cuda_capability", "cuda_runtime_version", "cudnn_version"):
        assert info[key]


# ---------------------------------------------------------------------------
# The comparison logic
# ---------------------------------------------------------------------------


def _result(accuracy=100.0, normalised=1.0, loss=1e-4):
    return {"cell": {"t": 1}, "steps_run": 2000,
            "best_val": {"token_accuracy": accuracy, "chance_normalised_token_accuracy": normalised},
            "final_val_loss": loss}


def test_identical_runs_agree():
    report = compare_level_b(_result(), _result(), TOLERANCE.level_b_end_to_end)
    assert report["passed"] and all(v == 0 for v in report["measured_absolute_difference"].values())


def test_a_difference_beyond_tolerance_fails():
    limit = TOLERANCE.level_b_end_to_end.max_abs_token_accuracy_difference_points
    report = compare_level_b(_result(), _result(accuracy=100.0 - limit - 1.0), TOLERANCE.level_b_end_to_end)
    assert not report["passed"] and report["checks"]["token_accuracy"] is False


def test_a_missing_metric_fails_rather_than_passing_vacuously():
    report = compare_level_b(_result(), {**_result(), "final_val_loss": None}, TOLERANCE.level_b_end_to_end)
    assert report["checks"]["final_val_loss"] is False
    assert report["measured_absolute_difference"]["final_val_loss"] is None


def test_only_a_level_a_failure_blocks_gpu_execution():
    passing_a = {"level": "A_kernel_agreement", "blocking": True, "passed": True}
    failing_b = {"level": "B_end_to_end", "blocking": False, "passed": False}
    report = summarise(passing_a, failing_b, AMENDMENT)
    assert report["gpu_execution_permitted"] is True
    assert report["trajectory_divergence_recorded"] is True and report["passed"] is False

    failing_a = {**passing_a, "passed": False}
    assert summarise(failing_a, None, AMENDMENT)["gpu_execution_permitted"] is False


# ---------------------------------------------------------------------------
# The recorded cross-check and the Phase 1 freeze
# ---------------------------------------------------------------------------


def test_integrity_check_reapplies_the_tolerance_instead_of_trusting_the_verdict(tmp_path):
    """A report whose thresholds were loosened after the fact must not pass."""
    limits = TOLERANCE.level_a_kernel_agreement
    forged = {
        "level_a": {"devices": ["cpu", "cuda"], "passed": True,
                    "measured": {"max_abs_logit_difference": limits.max_abs_logit_difference * 1000,
                                 "max_rel_logit_difference": 0.0,
                                 "max_abs_loss_difference_nats": 0.0,
                                 "max_rel_gradient_norm_difference": 0.0}},
        "level_b": None,
        "determinism": {"shared": {"use_deterministic_algorithms": True, "matmul_allow_tf32": False}},
        "environments": {"cpu": {}, "cuda": {}},
        "gpu_execution_permitted": True,
    }
    (tmp_path / "crosscheck_report.json").write_text(json.dumps(forged), encoding="utf-8")
    report = check_cpu_gpu_crosscheck(tmp_path)
    assert report["checks"]["level_a_within_tolerance"] is False
    assert report["checks"]["level_a_verdict_matches_recomputation"] is False
    assert report["passed"] is False


def test_missing_crosscheck_is_reported_not_assumed_fine(tmp_path):
    assert check_cpu_gpu_crosscheck(tmp_path)["passed"] is False


def test_phase1_files_are_still_byte_identical():
    report = check_phase1_frozen()
    assert report["passed"], {"modified": report["modified"], "missing": report["missing"]}
    assert report["files_verified"] == report["files_total"] > 0


def test_freeze_manifest_detects_a_modified_phase1_file(tmp_path):
    """The manifest is only worth having if a changed byte actually fails it."""
    manifest = json.loads(Path("phase2/phase1_freeze.json").read_text(encoding="utf-8"))
    name, entry = next(iter(manifest["files"].items()))
    fake_root = tmp_path / "repo"
    (fake_root / name).parent.mkdir(parents=True, exist_ok=True)
    (fake_root / name).write_text("tampered", encoding="utf-8")
    trimmed = {**manifest, "files": {name: entry}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(trimmed), encoding="utf-8")
    report = check_phase1_frozen(path, fake_root)
    assert report["modified"] == [name] and report["passed"] is False
