#!/usr/bin/env python3
"""Phase 2: re-run verification and the unchanged gates under a pre-training amendment.

The raw gate verdict is reported unchanged. The amended verdict is evaluated
separately (PHASE2_BENCHMARK_DESIGN.md §8). The original gate run is never
overwritten. Exits 1 unless the amended verdict passes.

Usage:
    python scripts/phase2_gates_amended.py [--config configs/phase2/gates_amendment01.yaml]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import parse_setup
from phase2.config import AmendedGatesRunConfig, load_amendment, load_benchmark
from phase2.runner import run_amended_gates


def main(argv=None) -> int:
    setup = parse_setup(argv, "Phase 2 amended verification and gates", "phase2/gates_amendment01", AmendedGatesRunConfig)
    amendment = load_amendment(setup.config.amendment)
    evaluation = run_amended_gates(setup.config, setup.config_path, setup.run_dir,
                                   load_benchmark(setup.config.benchmark), amendment, setup.device)
    return 0 if evaluation["amended_verdict"] == amendment.gate_verdict else 1


if __name__ == "__main__":
    sys.exit(main())
