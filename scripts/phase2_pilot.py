#!/usr/bin/env python3
"""Phase 2 pilot P1: step-budget calibration and floor check.

Refuses to run unless the gate run (runs/phase2/gates) has verdict PASS.

Usage:
    python scripts/phase2_pilot.py [--config configs/phase2/pilot_p1.yaml] [--device auto|cpu|cuda]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import parse_setup
from phase2.config import PilotConfig
from phase2.runner import run_pilot


def main(argv=None) -> int:
    setup = parse_setup(argv, "Phase 2 pilot P1", "phase2/pilot_p1", PilotConfig)
    report = run_pilot(setup.config, setup.config_path, setup.device, setup.run_dir)
    return 0 if report["fixed_step_budget"] else 1


if __name__ == "__main__":
    sys.exit(main())
