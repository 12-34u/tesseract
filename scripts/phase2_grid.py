#!/usr/bin/env python3
"""Phase 2 full grid (resumable). NOT to be run before the pilot has been reviewed.

Usage:
    python scripts/phase2_grid.py --confirm-full-grid [--config configs/phase2/grid.yaml] [--device ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import parse_setup
from phase2.config import GridConfig
from phase2.runner import run_grid

CONFIRM_FLAG = "--confirm-full-grid"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    confirm = CONFIRM_FLAG in argv
    if confirm:
        argv.remove(CONFIRM_FLAG)
    setup = parse_setup(argv, "Phase 2 full grid", "phase2/grid", GridConfig)
    run_grid(setup.config, setup.config_path, setup.device, setup.run_dir, confirm=confirm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
