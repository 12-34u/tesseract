#!/usr/bin/env python3
"""P1c step 1: low-order signal structure of the A5 CA task. Analysis only; trains nothing.

Usage:
    python scripts/phase2_signal_analysis.py [--samples 2000] [--t-values 1 2 4 8]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from phase2.config import load_benchmark
from phase2.signal_analysis import run_signal_analysis
from utils.config import RunConfig
from utils.paths import resolve_run_dir
from utils.run_artifacts import RunRecorder, write_json

ARTIFACTS = ["signal_analysis.json", "summary.txt"]


@dataclass(frozen=True)
class SignalAnalysisConfig:
    experiment: RunConfig
    group: str
    n: int
    t_values: tuple
    samples: int


def summary_text(report: dict, config: SignalAnalysisConfig) -> str:
    lines = ["PHASE 2 — P1c STEP 1: LOW-ORDER SIGNAL STRUCTURE (analysis only, no training)", "=" * 78,
             f"group {config.group}, n={config.n}, {config.samples} sampled states per T; "
             "excluded cell enumerated exactly over all 60 values", ""]
    lines.append(f"{'T':>3} {'window':>7} {'letters':>8} {'best proper-subset acc':>24} {'chance':>8} {'ratio':>7}  "
                 f"{'excluded offset':>15}  occurrences per offset")
    for t, e in report["per_T"].items():
        lines.append(f"{t:>3} {e['window_cells']:>7} {e['word_letters']:>8} "
                     f"{e['max_proper_subset_predictability'] * 100:>23.3f}% {e['chance'] * 100:>7.3f}% "
                     f"{e['ratio_to_chance']:>7.2f}  {e['most_informative_excluded_offset']:>15}  "
                     f"{e['occurrences_per_offset']}")
    lines += ["", "Reading: 'best proper-subset acc' upper-bounds what ANY strict subset of the window can achieve.",
              "At chance it means no lower-order feature carries signal, so SGD gets no partial credit and the",
              "full (T+1)-way interaction must be acquired at once — a property of the task, not of model width."]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--t-values", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--output-dir", default="phase2/signal_analysis")
    args = parser.parse_args(argv)

    benchmark = load_benchmark()
    t_values = tuple(args.t_values or benchmark.t_values)
    config = SignalAnalysisConfig(
        experiment=RunConfig("phase2_signal_analysis", args.output_dir, args.seed, "cpu"),
        group=benchmark.group, n=benchmark.n, t_values=t_values, samples=args.samples)
    run_dir = resolve_run_dir(args.output_dir)
    with RunRecorder(run_dir, config.experiment.name, config, Path(__file__), torch.device("cpu"), ARTIFACTS) as run:
        report = run_signal_analysis(config.group, t_values, config.n, config.samples, args.seed)
        write_json(run.path("signal_analysis.json"), report)
        text = summary_text(report, config)
        run.path("summary.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
        run.finish("COMPLETED", report["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
