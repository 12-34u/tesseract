#!/usr/bin/env python3
"""Single reproducibility entry point for the Phase 1 experiments.

Runs Crucible (GO/NO-GO gate), K-scaling and, unless skipped, the cellular
automaton experiment, each in its own process. The reported status of each
experiment is its real exit code, and this script exits non-zero if any of
them failed.

Usage:
    python scripts/run_all_experiments.py [--skip-ca] [--device auto|cpu|cuda]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    ("Crucible", "experiments/crucible.py"),
    ("K-Scaling", "experiments/k_scaling.py"),
    ("Cellular Automaton", "experiments/cellular_automaton.py"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Tesseract Phase 1 experiments.")
    parser.add_argument("--skip-ca", action="store_true", help="Skip the cellular automaton experiment.")
    parser.add_argument(
        "--device", default=None, help="Passed to every experiment (auto|cpu|cuda[:N]); default: each config's value."
    )
    args = parser.parse_args()

    print("=" * 56)
    print("  TESSERACT REPRODUCIBILITY SUITE")
    print("=" * 56)
    print(f"  Python:   {sys.executable}")
    print(f"  Device:   {args.device or 'from each config'}")
    print(f"  Skip CA:  {args.skip_ca}")
    print("=" * 56)

    statuses: dict[str, str] = {}
    for index, (name, script) in enumerate(EXPERIMENTS, start=1):
        if name == "Cellular Automaton" and args.skip_ca:
            statuses[name] = "SKIPPED"
            continue

        cmd = [sys.executable, str(REPO_ROOT / script)]
        if args.device:
            cmd += ["--device", args.device]
        print(f"\n>>> {index}/{len(EXPERIMENTS)} {name}: {' '.join(cmd)}\n")
        start = time.perf_counter()
        returncode = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        elapsed = time.perf_counter() - start
        statuses[name] = "PASS" if returncode == 0 else f"FAIL (exit {returncode})"
        print(f"\n<<< {name}: {statuses[name]} in {elapsed:.1f}s")

        if returncode != 0 and name == "Crucible":
            print("Crucible is the GO/NO-GO gate; not running the remaining experiments.")
            break

    print("\n" + "=" * 56)
    print("  RESULTS")
    print("=" * 56)
    for name, _ in EXPERIMENTS:
        print(f"  {name + ':':<22}{statuses.get(name, 'NOT RUN')}")
    print("=" * 56)

    ok = all(s in ("PASS", "SKIPPED") for s in statuses.values()) and len(statuses) == len(EXPERIMENTS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
