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
