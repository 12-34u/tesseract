#!/usr/bin/env python3
"""Amendment 02a: verify CPU/GPU agreement before Phase 2 runs on GPU.

Level A compares one forward+backward from identical weights (blocking).
Level B trains the same cell on each device and compares final metrics
(advisory). Tolerances come from the amendment and are not read from results.

Usage:
    python scripts/phase2_crosscheck.py [--output-dir phase2/crosscheck_cpu_gpu]
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.compute import compute_provenance, configure_determinism, describe_compute  # noqa: E402
from phase2.config import (  # noqa: E402
    TrainingConfig,
    load_amendment_02a,
    load_benchmark,
    resolve_model_variant,
    ModelVariant,
)
from phase2.crosscheck import compare_level_b, level_a_kernel_agreement, summarise  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.runner import _cell_config, splits_dir  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.config import OptimizerConfig, RunConfig  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="phase2/crosscheck_cpu_gpu")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-build-leg", action="store_true",
                        help="train only the CPU leg, using whichever torch build this environment has. "
                             "Run it in the Phase 1 CPU environment to compare BUILDS (+cpu vs +cu129), "
                             "not just devices.")
    parser.add_argument("--recompare", action="store_true",
                        help="rebuild the report from completed Level B runs; never retrains or overwrites them")
    args = parser.parse_args()

    import torch

    from phase2.runner import run_cell

    if not torch.cuda.is_available() and not args.cpu_build_leg:
        print("CUDA is not available in this environment; run this script in the CUDA environment.", file=sys.stderr)
        return 2

    amendment = load_amendment_02a()
    tolerance = amendment.cross_device_tolerance
    benchmark = load_benchmark()
    cell = tolerance.level_b_end_to_end.cell
    model_cfg = resolve_model_variant(ModelVariant(name="medium", base="phase2/prototype_medium"), 60)

    run_dir = resolve_run_dir(args.output_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not (args.recompare or args.cpu_build_leg):
        raise RuntimeError(f"{run_dir} already contains results; it is preserved. "
                           f"Use a new output directory, or --recompare to rebuild the report "
                           f"from the completed runs without retraining.")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu_build_leg:
        from phase2.runner import run_cell
        from utils.config import OptimizerConfig as _Opt
        device = torch.device("cpu")
        det = configure_determinism(device)
        steps = tolerance.level_b_end_to_end.steps
        cell_dir = run_dir / "level_b_build" / f"cpu_{torch.__version__}"
        if (cell_dir / "result.json").is_file():
            print(f"{cell_dir} already holds a completed run; it is preserved.")
            return 0
        training = TrainingConfig(optimizer=_Opt("adamw", 1e-3, 0.01), batch_size=args.batch_size,
                                  max_steps=steps, eval_every=args.eval_every, eval_batch_size=benchmark.val_size)
        experiment = RunConfig(name="phase2_crosscheck_cpu_build", output_dir=str(cell_dir), seed=cell.seed,
                               device="cpu")
        print(f"=== Level B (build leg): training {cell.cell_id} with torch {torch.__version__} ===")
        run_cell(_cell_config(experiment, experiment.name, cell_dir, model_cfg, benchmark, training, cell),
                 Path("configs/phase2/amendment_02a_draft.yaml"), device, cell_dir,
                 evaluate_test=False, compute_coverage=False, provenance=compute_provenance(device, det))
        print(f"build leg written to {cell_dir}")
        return 0

    determinism = {name: None for name in ("cpu", "cuda")}
    determinism["shared"] = configure_determinism(torch.device("cuda"))

    store = SplitStore(splits_dir(), benchmark.data_seed)
    val = store.get(cell.group, cell.task, "val", cell.t, benchmark.n, benchmark.val_size)

    print("=== Level A: kernel agreement (one forward + backward, identical weights) ===")
    level_a = level_a_kernel_agreement(model_cfg, benchmark, cell, val.inputs, val.targets,
                                       tolerance.level_a_kernel_agreement)
    for key, value in level_a["measured"].items():
        print(f"  {key}: {value:.3e}")
    print(f"  Level A: {'PASS' if level_a['passed'] else 'FAIL'}")

    level_b = None
    if not level_a["passed"] and tolerance.level_a_kernel_agreement.blocking:
        print("  Level A is blocking and failed; Level B is not run.", file=sys.stderr)
    else:
        steps = tolerance.level_b_end_to_end.steps
        training = TrainingConfig(
            optimizer=OptimizerConfig("adamw", 1e-3, 0.01), batch_size=args.batch_size, max_steps=steps, eval_every=args.eval_every,
            eval_batch_size=benchmark.val_size)
        results = {}
        for name in ("cpu", "cuda"):
            device = torch.device(name)
            determinism[name] = configure_determinism(device)
            cell_dir = run_dir / "level_b" / name
            experiment = RunConfig(name=f"phase2_crosscheck_{name}", output_dir=str(cell_dir), seed=cell.seed,
                                   device=name)
            existing = cell_dir / "result.json"
            if existing.is_file():
                print(f"=== Level B: reusing the completed {name} run at {cell_dir} (not retrained) ===")
                results[name] = json.loads(existing.read_text(encoding="utf-8"))
                continue
            print(f"=== Level B: training {cell.cell_id} on {name} for {steps} steps ===")
            results[name] = run_cell(
                _cell_config(experiment, f"phase2_crosscheck_{name}", cell_dir, model_cfg, benchmark, training, cell),
                Path("configs/phase2/amendment_02a_draft.yaml"), device, cell_dir,
                evaluate_test=False, compute_coverage=False,
                provenance=compute_provenance(device, determinism[name]))
        level_b = compare_level_b(results["cpu"], results["cuda"], tolerance.level_b_end_to_end)
        for label, diff in level_b["measured_absolute_difference"].items():
            values = level_b["values"][label]
            print(f"  {label}: cpu={values['cpu']} cuda={values['gpu']} |diff|={diff}")
        print(f"  Level B: {'PASS' if level_b['passed'] else 'FAIL (advisory)'}")

    report = summarise(level_a, level_b, amendment)
    report["environments"] = {name: describe_compute(torch.device(name)) for name in ("cpu", "cuda")}
    report["determinism"] = determinism
    write_json(run_dir / "crosscheck_report.json", report)

    print(f"\nGPU execution permitted: {report['gpu_execution_permitted']}")
    print(f"Trajectory divergence recorded: {report['trajectory_divergence_recorded']}")
    print(f"report: {run_dir / 'crosscheck_report.json'}")
    return 0 if report["gpu_execution_permitted"] else 1


if __name__ == "__main__":
    sys.exit(main())
