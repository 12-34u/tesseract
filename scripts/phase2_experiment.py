#!/usr/bin/env python3
"""Phase 2 experiment runner (guarded). PHASE2_BENCHMARK_DESIGN.md §10.

Training refuses to start while the model width or max_steps is PENDING_P1B,
while Amendment 02 is a draft, when the benchmark integrity check fails,
before P1b completes, or without --confirm-stage.

Usage:
    python scripts/phase2_experiment.py --list
    python scripts/phase2_experiment.py --stage t4 --confirm-stage
    python scripts/phase2_experiment.py --cell tesseract_d8_A5_main_T4_s0 --confirm-stage
    python scripts/phase2_experiment.py --analyze        # only once every planned cell is complete
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.config import load_benchmark, load_phase2_experiment
from phase2.experiment import analyze_experiment, describe_matrix, run_one_cell, run_stage
from utils.config import resolve_config_path
from utils.device import resolve_device
from utils.paths import resolve_run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 experiment runner")
    parser.add_argument("--config", default="phase2/phase2_experiment")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda[:N] (default: experiment.device)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="print the planned matrix (no training)")
    action.add_argument("--stage", help="run one stage (requires --confirm-stage)")
    action.add_argument("--cell", help="run one planned cell by id (requires --confirm-stage)")
    action.add_argument("--analyze", action="store_true", help="pre-registered analysis of the complete matrix")
    parser.add_argument("--confirm-stage", action="store_true")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    config = load_phase2_experiment(config_path)
    benchmark = load_benchmark(config.benchmark)
    if args.list:
        print(describe_matrix(config, benchmark.group))
        return 0

    device = resolve_device(args.device or config.experiment.device)
    run_dir = resolve_run_dir(config.experiment.output_dir)
    if args.stage:
        run_stage(config, config_path, device, run_dir, args.stage, confirm=args.confirm_stage, benchmark=benchmark)
    elif args.cell:
        run_one_cell(config, config_path, device, run_dir, args.cell, confirm=args.confirm_stage, benchmark=benchmark)
    else:
        analyze_experiment(config, config_path, device, run_dir, benchmark=benchmark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
