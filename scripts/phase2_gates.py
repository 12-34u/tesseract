#!/usr/bin/env python3
"""Phase 2: independent verification, gates G1/G2, frozen splits and sanity baselines.

Trains no model. Exits 1 unless everything passes.

Usage:
    python scripts/phase2_gates.py [--config configs/phase2/gates.yaml]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import parse_setup
from phase2.config import GatesRunConfig, load_benchmark
from phase2.runner import run_verification_and_gates


def main(argv=None) -> int:
    setup = parse_setup(argv, "Phase 2 verification and gates", "phase2/gates", GatesRunConfig)
    report = run_verification_and_gates(setup.config, setup.config_path, setup.run_dir,
                                        load_benchmark(setup.config.benchmark), setup.device)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
