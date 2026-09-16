#!/usr/bin/env python3
"""Phase 2 P1b: balanced width × K capacity/learnability diagnostic (PHASE2_BENCHMARK_DESIGN.md §9).

Validation only, with a fixed step budget. Refuses to write into a directory
that already contains results.

Usage:
    python scripts/phase2_capacity_diagnostic.py [--config configs/phase2/capacity_diagnostic_p1b.yaml]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import parse_setup
from phase2.config import CapacityDiagnosticConfig
from phase2.runner import run_capacity_diagnostic


def main(argv=None) -> int:
    setup = parse_setup(argv, "Phase 2 P1b capacity diagnostic", "phase2/capacity_diagnostic_p1b", CapacityDiagnosticConfig)
    run_capacity_diagnostic(setup.config, setup.config_path, setup.device, setup.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
