#!/usr/bin/env python3
"""Run every Phase 2 integrity check and report. Read-only: trains nothing, writes nothing.

Checks, in order:
  1. Phase 1 freeze      - the 104 Phase 1 files are still byte-identical
  2. Benchmark integrity - gate history intact, splits match their checksums
  3. CPU/GPU cross-check - Amendment 02a tolerances re-applied to the recording

Usage:
    python scripts/phase2_integrity.py [--skip-splits]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.config import load_amendment, load_benchmark  # noqa: E402
from phase2.integrity import (  # noqa: E402
    check_benchmark_integrity,
    check_cpu_gpu_crosscheck,
    check_phase1_frozen,
)
from phase2.runner import DEFAULT_GATES_DIR, splits_dir  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402

CROSSCHECK_RUN_DIR = "phase2/crosscheck_cpu_gpu"


def report(title: str, checks: dict) -> bool:
    print(f"\n{title}")
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-splits", action="store_true", help="skip regenerating split checksums (slow)")
    args = parser.parse_args()

    frozen = check_phase1_frozen()
    print(f"\n1. Phase 1 freeze (commit {frozen['freeze_commit'][:7]})")
    print(f"  {'OK  ' if frozen['passed'] else 'FAIL'} "
          f"{frozen['files_verified']}/{frozen['files_total']} files byte-identical")
    for path in frozen["modified"]:
        print(f"       MODIFIED: {path}")
    for path in frozen["missing"]:
        print(f"       MISSING:  {path}")

    benchmark = check_benchmark_integrity(load_benchmark(), load_amendment(), resolve_run_dir(DEFAULT_GATES_DIR),
                                          splits_dir(), verify_splits=not args.skip_splits)
    benchmark_ok = report("2. Benchmark integrity", benchmark["checks"])

    crosscheck = check_cpu_gpu_crosscheck(resolve_run_dir(CROSSCHECK_RUN_DIR))
    crosscheck_ok = report("3. CPU/GPU cross-check (Amendment 02a)", crosscheck["checks"])
    if crosscheck["checks"].get("crosscheck_present"):
        details = crosscheck["details"]
        print(f"       gpu={details['gpu']} driver={details['driver']} torch={details['torch_versions']}")
        for label, value in (details["level_a_measured"] or {}).items():
            print(f"       level A {label}: {value:.3e}")
        for label, value in (details["level_b_measured"] or {}).items():
            print(f"       level B {label}: {value:.3e}" if value is not None else f"       level B {label}: n/a")

    passed = frozen["passed"] and benchmark_ok and crosscheck_ok
    print(f"\n{'ALL INTEGRITY CHECKS PASS' if passed else 'INTEGRITY CHECKS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
