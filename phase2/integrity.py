"""Read-only benchmark integrity checks, run before any Phase 2 training stage.

Confirms that the recorded gate history is intact and honest, and that the frozen
evaluation data is unchanged:
* the original gate run still records verdict FAIL, with its G2 failures exactly
  at the documented diagnostic T (Amendment 01);
* the amended gate run records PASS_WITH_AMENDMENT_01 with raw verdict FAIL, and
  references the unchanged original report (SHA-256);
* the benchmark definition equals the one the gates evaluated;
* independent verification passed;
* every frozen split still matches its recorded SHA-256.

Nothing here writes to the gate directories.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from phase2.config import AmendmentConfig, BenchmarkConfig
from phase2.data import ChecksumError, SplitStore
from utils.paths import resolve_run_dir
from utils.run_artifacts import METADATA_FILENAME, to_json_safe


def _load(path: Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _roundtrip(value):
    return json.loads(json.dumps(to_json_safe(value)))


def check_benchmark_integrity(benchmark: BenchmarkConfig, amendment01: AmendmentConfig, amended_gates_dir: Path,
                              splits_root: Path, verify_splits: bool = True) -> Dict:
    original_dir = resolve_run_dir(amendment01.original_gates_output_dir)
    amended_dir = Path(amended_gates_dir)
    required = {
        "original_metadata": original_dir / METADATA_FILENAME,
        "original_gates_report": original_dir / "gates_report.json",
        "amended_metadata": amended_dir / METADATA_FILENAME,
        "amendment_evaluation": amended_dir / "amendment_evaluation.json",
        "amended_splits_report": amended_dir / "splits_report.json",
        "amended_verification_report": amended_dir / "verification_report.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"checks": {"gate_artifacts_present": False}, "details": {"missing": missing}, "passed": False}

    original_meta = _load(required["original_metadata"])
    original_report = _load(required["original_gates_report"])
    amended_meta = _load(required["amended_metadata"])
    evaluation = _load(required["amendment_evaluation"])
    splits_report = _load(required["amended_splits_report"])
    verification = _load(required["amended_verification_report"])

    per_t = original_report["primary"]["per_T"]
    documented = {str(s.t): s.candidate for s in amendment01.documented_shortcuts}
    g2_failures = {t for t, r in per_t.items() if r["G2_passed"] is False}
    original_sha = hashlib.sha256(required["original_gates_report"].read_bytes()).hexdigest()

    checks = {
        "gate_artifacts_present": True,
        "original_gate_run_FAIL_preserved": original_meta.get("status") == "completed"
        and original_meta.get("verdict") == "FAIL" and original_report.get("passed") is False,
        "original_G2_failures_exactly_the_documented_diagnostic_T": g2_failures == set(documented)
        and set(documented) <= {str(t) for t in amendment01.diagnostic_t_values}
        and all(per_t[t]["G2"]["max_token_agreement_candidate"] == c for t, c in documented.items()),
        "original_G2_passed_on_primary_T": all(per_t.get(str(t), {}).get("G2_passed") is True
                                               for t in amendment01.primary_depth_t_values),
        "amended_verdict_recorded_with_raw_FAIL": amended_meta.get("status") == "completed"
        and amended_meta.get("verdict") == amendment01.gate_verdict
        and (amended_meta.get("results") or {}).get("raw_gate_verdict") == "FAIL",
        "amendment_evaluation_all_checks_passed": evaluation.get("amended_verdict") == amendment01.gate_verdict
        and all((evaluation.get("checks") or {}).values()),
        "amended_run_references_unchanged_original_report": (evaluation.get("original_gate_run") or {}).get("run_id")
        == original_meta.get("run_id")
        and (evaluation.get("original_gate_run") or {}).get("gates_report_sha256") == original_sha,
        "benchmark_matches_gated_definition": _roundtrip(asdict(benchmark)) == original_report.get("benchmark"),
        "independent_verification_passed": verification.get("passed") is True,
        "data_seed_matches": splits_report.get("data_seed") == benchmark.data_seed,
    }
    details = {"original_gates_run_id": original_meta.get("run_id"), "amended_gates_run_id": amended_meta.get("run_id"),
               "original_gates_report_sha256": original_sha}

    if verify_splits:
        store = SplitStore(Path(splits_root), splits_report["data_seed"])
        mismatched = []
        for key, recorded in splits_report["splits"].items():
            group, task, split, t, n, size = key.split("/")
            try:
                regenerated = store.get(group, task, split, int(t[1:]), int(n[1:]), int(size[1:])).sha256
            except ChecksumError as exc:  # recorded as a failed check, not suppressed
                mismatched.append(f"{key}: {exc}")
                continue
            if regenerated != recorded:
                mismatched.append(key)
        checks["frozen_split_checksums_verified"] = not mismatched
        details.update(splits_checked=len(splits_report["splits"]), split_mismatches=mismatched)

    return {"checks": checks, "details": details, "passed": all(checks.values())}


# ---------------------------------------------------------------------------
# Phase 1 freeze: the files no Phase 2 step may change
# ---------------------------------------------------------------------------
#
# Phase 1 finished at commit PHASE1_FREEZE_COMMIT. Every file tracked there is
# frozen except Phase 2's own work product, which simply had not been separated
# into its own directories yet when that commit was made:
#
#   phase2/, tests/phase2/, configs/phase2/, scripts/phase2_*  Phase 2 code
#   PHASE2_*.md                                                Phase 2 documents
#   app/                                                       the dashboard, a read-only
#                                                              viewer Phase 2 extended with
#                                                              new panels under explicit approval
#
# Everything else - the model, training loop, evaluation metrics, Phase 1
# experiments, their configs, their tests, and the recorded Phase 1 results -
# must stay byte-identical. That is the claim this check enforces.

PHASE1_FREEZE_COMMIT = "a69054f"
PHASE1_EXCLUDED_PREFIXES = ("phase2/", "tests/phase2/", "configs/phase2/", "scripts/phase2_", "PHASE2_", "app/")
DEFAULT_MANIFEST_PATH = Path("phase2/phase1_freeze.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _git(root: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def _is_phase1_path(path: str) -> bool:
    return not path.startswith(PHASE1_EXCLUDED_PREFIXES)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_phase1_manifest(root: Path) -> Dict:
    """Hash every frozen file, refusing to re-baseline onto files that already drifted.

    The hashes come from the working tree, but each one is required to equal the
    blob recorded at the freeze commit. A file that has changed therefore fails
    the build instead of being silently written into the manifest as the new
    truth - the failure mode a regenerable manifest would otherwise invite.
    """
    commit = _git(root, "rev-parse", PHASE1_FREEZE_COMMIT).strip()
    tracked = [p for p in _git(root, "ls-tree", "-r", "--name-only", commit).splitlines() if _is_phase1_path(p)]

    files, drifted, missing = {}, [], []
    for rel in sorted(tracked):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        frozen = _git(root, "rev-parse", f"{commit}:{rel}").strip()
        if _git(root, "hash-object", str(path)).strip() != frozen:
            drifted.append(rel)
            continue
        files[rel] = {"sha256": _sha256_file(path), "git_blob": frozen}

    if drifted or missing:
        raise RuntimeError(
            f"refusing to build a manifest over drifted Phase 1 files: modified={drifted} missing={missing}")
    return {"freeze_commit": commit, "freeze_commit_short": PHASE1_FREEZE_COMMIT,
            "excluded_prefixes": list(PHASE1_EXCLUDED_PREFIXES), "files": files}


def write_phase1_manifest(manifest: Dict, path) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_phase1_frozen(manifest_path=DEFAULT_MANIFEST_PATH, root: Path = None) -> Dict:
    """Verify every file in the manifest still hashes to its recorded SHA-256."""
    root = root or repo_root()
    manifest = _load(Path(manifest_path))
    modified, missing = [], []
    for rel, recorded in sorted(manifest["files"].items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif _sha256_file(path) != recorded["sha256"]:
            modified.append(rel)

    total = len(manifest["files"])
    return {"freeze_commit": manifest["freeze_commit"], "files_total": total,
            "files_verified": total - len(modified) - len(missing),
            "modified": modified, "missing": missing,
            "passed": not modified and not missing}


# ---------------------------------------------------------------------------
# Compute platform: the recorded CPU/GPU cross-check (Amendment 02a)
# ---------------------------------------------------------------------------


def check_cpu_gpu_crosscheck(crosscheck_dir, tolerance=None) -> Dict:
    """Confirm a cross-check was actually run, and still meets the stated tolerance.

    The tolerance is re-read from Amendment 02a and re-applied to the recorded
    measurements rather than trusting the verdict stored in the report, so a
    report whose thresholds were loosened after the fact fails here.
    """
    path = Path(crosscheck_dir) / "crosscheck_report.json"
    if not path.is_file():
        return {"checks": {"crosscheck_present": False}, "details": {"missing": str(path)}, "passed": False}

    if tolerance is None:
        from phase2.config import load_amendment_02a
        tolerance = load_amendment_02a().cross_device_tolerance

    report = _load(path)
    level_a = report.get("level_a") or {}
    level_b = report.get("level_b") or {}
    measured_a = level_a.get("measured") or {}
    limits_a = tolerance.level_a_kernel_agreement

    recomputed_a = {
        key: (measured_a.get(key) is not None and measured_a[key] <= getattr(limits_a, key))
        for key in ("max_abs_logit_difference", "max_rel_logit_difference",
                    "max_abs_loss_difference_nats", "max_rel_gradient_norm_difference")
    }
    limits_b = tolerance.level_b_end_to_end
    measured_b = level_b.get("measured_absolute_difference") or {}
    recomputed_b = {
        "chance_normalised_token_accuracy": limits_b.max_abs_chance_normalised_token_accuracy_difference,
        "token_accuracy": limits_b.max_abs_token_accuracy_difference_points,
        "final_val_loss": limits_b.max_abs_final_val_loss_difference_nats,
    }
    recomputed_b = {key: (measured_b.get(key) is not None and measured_b[key] <= limit)
                    for key, limit in recomputed_b.items()}

    devices = {name: (report.get("environments") or {}).get(name) or {} for name in ("cpu", "cuda")}
    checks = {
        "crosscheck_present": True,
        "level_a_compared_both_devices": set(level_a.get("devices") or []) == {"cpu", "cuda"},
        "level_a_within_tolerance": all(recomputed_a.values()),
        "level_a_verdict_matches_recomputation": level_a.get("passed") == all(recomputed_a.values()),
        "determinism_enforced": bool((report.get("determinism") or {}).get("shared", {})
                                     .get("use_deterministic_algorithms")),
        "tf32_disabled": (report.get("determinism") or {}).get("shared", {}).get("matmul_allow_tf32") is False,
        "gpu_metadata_recorded": all(devices["cuda"].get(key) for key in
                                     ("cuda_device_name", "cuda_capability", "cuda_runtime_version",
                                      "nvidia_driver_version", "torch")),
        "gpu_execution_permitted": report.get("gpu_execution_permitted") is True,
    }
    details = {
        "level_a_measured": measured_a,
        "level_a_recomputed_checks": recomputed_a,
        "level_b_measured": measured_b,
        "level_b_recomputed_checks": recomputed_b,
        "level_b_within_tolerance": all(recomputed_b.values()) if level_b else None,
        "level_b_blocking": limits_b.blocking,
        "trajectory_divergence_recorded": report.get("trajectory_divergence_recorded"),
        "torch_versions": {name: devices[name].get("torch") for name in devices},
        "gpu": devices["cuda"].get("cuda_device_name"),
        "driver": devices["cuda"].get("nvidia_driver_version"),
    }
    # Level B is advisory by Amendment 02a P4: divergence is recorded, not fatal.
    return {"checks": checks, "details": details, "passed": all(checks.values())}
