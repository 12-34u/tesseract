"""Phase 2 orchestration.

1. ``run_verification_and_gates``: implementation-B verification, gates G1/G2,
   frozen checksummed splits, and sanity baselines. Nothing trains until this
   passes.
2. ``run_cell``: one fixed-step training run with full artifacts.
3. ``run_pilot``: pilot P1. Calibrates the fixed step budget on
   T=1 / K=1, then runs the floor-check cells. Requires passed gates.
4. ``run_grid``: the full grid. It refuses to start without explicit
   confirmation, passed gates and a completed pilot.

All artifacts go under ``<runs root>/phase2/``; Phase 1 run directories are
never touched.
"""

import csv
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from phase2.analysis import aggregate
from phase2.config import (
    BenchmarkConfig,
    CellRunConfig,
    CellSpec,
    GatesRunConfig,
    GridConfig,
    PilotConfig,
    TrainingConfig,
    load_benchmark,
)
from phase2.data import SplitStore, TrainingStream, seen_window_codes, window_coverage
from phase2.gates import run_gates
from phase2.groups import get_group
from phase2.metrics import bootstrap_ci
from phase2.models import build_model
from phase2.sanity import sanity_baselines
from phase2.training import train_fixed_steps
from phase2.verification import run_verification
from utils.paths import resolve_run_dir, runs_root
from utils.run_artifacts import METADATA_FILENAME, RunRecorder, write_json
from utils.seed import set_seed

DEFAULT_GATES_DIR = "phase2/gates"


def splits_dir() -> Path:
    return runs_root() / "phase2" / "splits"


# ============================================================================
# 1. Verification, gates, splits, sanity baselines
# ============================================================================

GATES_ARTIFACTS = ["verification_report.json", "gates_report.json", "splits_report.json",
                   "sanity_baselines.json", "summary.txt"]


def _gates_summary(verification: Dict, gates: Dict, sanity: Dict, splits: Dict, passed: bool) -> str:
    lines = ["PHASE 2 — VERIFICATION AND PRE-TRAINING GATES", "=" * 60]
    lines.append(f"Independent verification: {'PASS' if verification['passed'] else 'FAIL'}")
    for name in ("A5", "Z60"):
        failed = [k for k, v in verification[name]["checks"].items() if not v]
        lines.append(f"  {name}: {'all checks pass' if not failed else 'FAILED: ' + ', '.join(failed)}")
    for name, words in verification["word_expansion"].items():
        lines.append(f"  {name} F^T == word expansion for T<=8: {words['passed']}")
    lines.append(f"  control tasks vs pure-Python reference: {all(t['passed'] for t in verification['tasks'].values())}")
    lines.append(f"  Z60 closed-form shortcut holds (expected): {verification['z60_closed_form_shortcut']['passed']}")

    primary = gates["primary"]
    lines += ["", f"Gates on {primary['group']} (n={primary['n']}): {'PASS' if primary['passed'] else 'FAIL'}"]
    cycles = primary["G1_cycles"]
    lines.append(f"  G1 cycles: {cycles['rows_with_repeated_state']} of {cycles['states']} trajectories repeat a state "
                 f"within {cycles['steps']} steps")
    for t, r in primary["per_T"].items():
        g2 = r["G2"]
        g2_text = "n/a (T=1 is the primitive)" if not g2["applicable"] else (
            f"max token agreement {g2['max_token_agreement']:.4f} ({g2['max_token_agreement_candidate']}), "
            f"max sequence agreement {g2['max_sequence_agreement']:.4f} → {'PASS' if g2['passed'] else 'FAIL'}")
        lines.append(f"  T={t}: G1 TV={r['G1_uniformity']['tv_distance']:.4f}, "
                     f"min in-window sensitivity={r['G1_sensitivity']['min_inside_fraction']:.4f}, "
                     f"out-of-window={r['G1_sensitivity']['max_outside_fraction']:.4f} → {'PASS' if r['G1_passed'] else 'FAIL'}; "
                     f"G2 {g2_text}")
    st = gates["self_test"]
    lines.append(f"  Self-test on Z2 (T=8): G2 detects collapse={st['G2_detects_collapse']} "
                 f"({st['G2_max_token_agreement_candidate']} agreement {st['G2_max_token_agreement']:.3f}); "
                 f"G1 sensitivity detects collapse={st['G1_sensitivity_detects_collapse']}")
    for name, ab in gates["ablations_informational"].items():
        lines.append(f"  [informational] {name}: gates {'pass' if ab['passed'] else 'fail'}")

    oracle_ok = all(s["oracle_independent_simulator"]["exact_match"] == 100.0 for s in sanity.values())
    lines += ["", f"Frozen splits: {len(splits)} (SHA-256 recorded in splits_report.json)",
              f"Oracle (independent simulator) = 100% exact match on every test split: {oracle_ok}",
              "", f"OVERALL: {'PASS' if passed else 'FAIL'}"]
    return "\n".join(lines)


def run_verification_and_gates(config: GatesRunConfig, config_path: Path, run_dir: Path, benchmark: BenchmarkConfig,
                               device: torch.device, log: Callable[[str], None] = print) -> Dict:
    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, GATES_ARTIFACTS) as run:
        log("Running independent verification…")
        verification = run_verification(t_max=8, n=benchmark.n)
        write_json(run.path("verification_report.json"), verification)

        log("Running gates G1/G2…")
        gates = run_gates(benchmark)
        write_json(run.path("gates_report.json"), {"benchmark": asdict(benchmark), **gates})

        log("Materialising frozen splits and sanity baselines…")
        store = SplitStore(splits_dir(), benchmark.data_seed)
        splits, sanity = {}, {}
        for group in (benchmark.group, *benchmark.ablation_groups):
            for task in benchmark.tasks:
                for t in benchmark.t_values:
                    for split_name, n, size in (("val", benchmark.n, benchmark.val_size),
                                                ("test", benchmark.n, benchmark.test_size),
                                                ("length", benchmark.length_n, benchmark.length_test_size)):
                        split = store.get(group, task, split_name, t, n, size)
                        splits[split.key] = split.sha256
                        if split_name == "test":
                            sanity[split.key] = sanity_baselines(split, seed=benchmark.gates.seed)
                for t in benchmark.interpolation_t_values:
                    split = store.get(group, task, "test", t, benchmark.n, benchmark.test_size)
                    splits[split.key] = split.sha256
        write_json(run.path("splits_report.json"), {"data_seed": benchmark.data_seed, "splits": splits})
        write_json(run.path("sanity_baselines.json"), sanity)

        oracle_ok = all(s["oracle_independent_simulator"]["exact_match"] == 100.0 for s in sanity.values())
        passed = verification["passed"] and gates["passed"] and oracle_ok
        summary = _gates_summary(verification, gates, sanity, splits, passed)
        run.path("summary.txt").write_text(summary + "\n", encoding="utf-8")
        log(summary)
        run.finish("PASS" if passed else "FAIL", {"verification_passed": verification["passed"],
                                                   "gates_passed": gates["passed"], "oracle_ok": oracle_ok,
                                                   "num_splits": len(splits)})
    return {"passed": passed, "verification": verification, "gates": gates}


def require_gates_passed(gates_run_dir: Path) -> Dict:
    meta_path = Path(gates_run_dir) / METADATA_FILENAME
    if not meta_path.is_file():
        raise RuntimeError(f"No gate run found at {gates_run_dir}; run scripts/phase2_gates.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "completed" or meta.get("verdict") != "PASS":
        raise RuntimeError(f"Gates have not passed (status={meta.get('status')}, verdict={meta.get('verdict')}); "
                           "no model training is allowed.")
    return meta


# ============================================================================
# 2. One training cell
# ============================================================================

CELL_ARTIFACTS = ["result.json", "val_curve.csv"]


def run_cell(config: CellRunConfig, config_path: Path, device: torch.device, run_dir: Path,
             stop_condition=None, compute_coverage: bool = True, log: Callable[[str], None] = print) -> Dict:
    cell, benchmark, training = config.cell, config.benchmark, config.training
    group = get_group(cell.group)
    if config.model.vocab_size != group.order:
        raise ValueError(f"model vocab_size {config.model.vocab_size} != |{cell.group}| = {group.order}")
    if config.model.max_seq_len < benchmark.n:
        raise ValueError("model max_seq_len is smaller than the ring size n")

    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, CELL_ARTIFACTS) as run:
        store = SplitStore(splits_dir(), benchmark.data_seed)
        val = store.get(cell.group, cell.task, "val", cell.t, benchmark.n, benchmark.val_size)
        test = store.get(cell.group, cell.task, "test", cell.t, benchmark.n, benchmark.test_size)

        set_seed(cell.seed)
        built = build_model(cell.family, cell.depth, config.model, benchmark.n)
        model = built.model.to(device)

        def make_stream() -> TrainingStream:
            return TrainingStream(cell.group, cell.task, cell.t, benchmark.n, training.batch_size, benchmark.data_seed,
                                  cell.seed, excluded_inputs=[val.inputs, test.inputs])

        log(f"[{cell.cell_id}] params={built.parameter_count:,} block_executions={built.block_executions} "
            f"flops/seq≈{built.forward_flops_per_sequence:,} max_steps={training.max_steps}")
        outcome = train_fixed_steps(model, make_stream(), val, test, training, device, group.order,
                                    stop_condition=stop_condition, log=log)

        coverage = None
        if compute_coverage and outcome.steps_run > 0:
            coverage = window_coverage(test.inputs, cell.t, group.order, seen_window_codes(make_stream, outcome.steps_run))

        test_ci = None
        if outcome.test_predictions is not None:
            equal = outcome.test_predictions == test.targets
            test_ci = {"exact_match": bootstrap_ci(equal.all(axis=1) * 100.0, seed=cell.seed),
                       "token_accuracy": bootstrap_ci(equal.mean(axis=1) * 100.0, seed=cell.seed)}

        result = {
            "cell": asdict(cell),
            "cell_id": cell.cell_id,
            "model": built.describe(),
            "max_steps": training.max_steps,
            "steps_run": outcome.steps_run,
            "stopped_reason": outcome.stopped_reason,
            "best_step": outcome.best_step,
            "best_val": outcome.best_val.to_dict() if outcome.best_val else None,
            "final_val": outcome.final_val.to_dict() if outcome.final_val else None,
            "test": outcome.test.to_dict() if outcome.test else None,
            "test_bootstrap_ci95": test_ci,
            "train_seconds": outcome.train_seconds,
            "eval_seconds": outcome.eval_seconds,
            "seconds_per_train_step": outcome.seconds_per_train_step,
            "rejected_training_rows": outcome.rejected_training_rows,
            "test_window_coverage": coverage,
            "split_sha256": {"val": val.sha256, "test": test.sha256},
        }

        fieldnames = sorted({key for row in outcome.history for key in row}, key=lambda k: (k != "step", k))
        with open(run.path("val_curve.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(outcome.history)
        write_json(run.path("result.json"), result)

        verdict = "FAIL" if outcome.stopped_reason == "non_finite_loss" else "COMPLETED"
        run.finish(verdict, {key: result[key] for key in ("cell_id", "steps_run", "stopped_reason", "best_step", "test")})
    return result


def _cell_config(base_experiment, name: str, run_dir: Path, model, benchmark, training, cell) -> CellRunConfig:
    return CellRunConfig(experiment=replace(base_experiment, name=name, output_dir=str(run_dir), seed=cell.seed),
                         model=model, benchmark=benchmark, training=training, cell=cell)


# ============================================================================
# 3. Pilot P1
# ============================================================================

PILOT_ARTIFACTS = ["pilot_report.json", "summary.txt", "calibration", "cells"]


def _fit_step_time(points: List[Dict]) -> Optional[Dict]:
    ks = np.array([p["block_executions"] for p in points], dtype=float)
    secs = np.array([p["seconds_per_train_step"] for p in points], dtype=float)
    if len(set(ks)) < 2:
        return None
    slope, intercept = np.polyfit(ks, secs, 1)
    return {"intercept_s": float(intercept), "per_block_execution_s": float(slope)}


def run_pilot(config: PilotConfig, config_path: Path, device: torch.device, run_dir: Path,
              benchmark: Optional[BenchmarkConfig] = None, gates_run_dir: Optional[Path] = None,
              log: Callable[[str], None] = print) -> Dict:
    benchmark = benchmark or load_benchmark(config.benchmark)
    gates_meta = require_gates_passed(gates_run_dir or resolve_run_dir(DEFAULT_GATES_DIR))
    seed = config.experiment.seed
    cal = config.calibration

    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, PILOT_ARTIFACTS) as run:
        log("=== Pilot P1 stage A: step-budget calibration ===")
        cal_cell = CellSpec(cal.family, cal.depth, benchmark.group, cal.task, cal.t, seed)
        cal_training = TrainingConfig(config.optimizer, config.batch_size, cal.max_steps, config.eval_every,
                                      config.eval_batch_size)
        cal_dir = run.path("calibration")
        cal_result = run_cell(
            _cell_config(config.experiment, f"{config.experiment.name}_calibration", cal_dir, config.model, benchmark,
                         cal_training, cal_cell),
            config_path, device, cal_dir,
            stop_condition=lambda val: val.exact_match >= cal.converge_exact_match, log=log,
        )
        converged = cal_result["stopped_reason"] == "stop_condition"
        budget = (math.ceil(cal.budget_multiplier * cal_result["steps_run"] / config.eval_every) * config.eval_every
                  if converged else None)
        log(f"Calibration converged={converged} after {cal_result['steps_run']} steps → fixed budget {budget}")

        floor_results = []
        if converged:
            log(f"=== Pilot P1 stage B: floor check at the fixed budget of {budget} steps ===")
            training = replace(cal_training, max_steps=budget)
            for pilot_cell in config.floor_cells:
                cell = CellSpec(pilot_cell.family, pilot_cell.depth, benchmark.group, pilot_cell.task, pilot_cell.t, seed)
                cell_dir = run.path("cells") / cell.cell_id
                result = run_cell(_cell_config(config.experiment, f"{config.experiment.name}_{cell.cell_id}", cell_dir,
                                               config.model, benchmark, training, cell),
                                  config_path, device, cell_dir, log=log)
                score = result["test"]["chance_normalised_token_accuracy"] if result["test"] else None
                floor_results.append({**result, "at_floor": score is None or score < config.floor_threshold})

        timing_points = [{"block_executions": r["model"]["block_executions"], "seconds_per_train_step": r["seconds_per_train_step"]}
                         for r in [cal_result, *floor_results]]
        step_time = _fit_step_time(timing_points)
        grid_estimate = None
        if step_time and budget:
            per_run = {k: (step_time["intercept_s"] + step_time["per_block_execution_s"] * k) * budget
                       for k in benchmark.k_values}
            runs_per_k = len(benchmark.tasks) * len(benchmark.t_values) * len(benchmark.model_seeds)
            grid_estimate = {
                "scope": "Tesseract main+controls grid only (3 tasks × 4 T × 4 K × 3 seeds); training steps only, "
                         "excluding evaluation, baselines and the Z60 ladder",
                "training_hours_per_run_by_k": {k: v / 3600 for k, v in per_run.items()},
                "total_training_hours": sum(v * runs_per_k for v in per_run.values()) / 3600,
                "device": str(device),
            }

        t8_cells = [r for r in floor_results if r["cell"]["t"] == max(benchmark.t_values)]
        report = {
            "gates_run_id": gates_meta.get("run_id"),
            "pre_registered": {
                "calibration_cell": asdict(cal),
                "convergence": f"validation exact match >= {cal.converge_exact_match}%",
                "budget_rule": f"{cal.budget_multiplier} × steps-to-converge, rounded up to eval_every={config.eval_every}",
                "floor_threshold_chance_normalised": config.floor_threshold,
                "tau": benchmark.tau,
            },
            "calibration": {key: cal_result[key] for key in ("cell_id", "steps_run", "stopped_reason", "best_step",
                                                             "best_val", "test", "seconds_per_train_step", "model",
                                                             "test_window_coverage")} | {"converged": converged},
            "fixed_step_budget": budget,
            "floor_check": [{key: r[key] for key in ("cell_id", "cell", "model", "steps_run", "best_step", "best_val",
                                                     "test", "test_bootstrap_ci95", "seconds_per_train_step",
                                                     "test_window_coverage", "at_floor")} for r in floor_results],
            "floor_effect_at_max_t": bool(t8_cells) and all(r["at_floor"] for r in t8_cells),
            "step_time_fit": step_time,
            "grid_compute_estimate": grid_estimate,
        }
        write_json(run.path("pilot_report.json"), report)
        summary = _pilot_summary(report)
        run.path("summary.txt").write_text(summary + "\n", encoding="utf-8")
        log(summary)
        run.finish("COMPLETED" if converged else "FAIL", {"fixed_step_budget": budget, "converged": converged,
                                                          "floor_effect_at_max_t": report["floor_effect_at_max_t"]})
    return report


def _pilot_summary(report: Dict) -> str:
    cal = report["calibration"]
    lines = ["PHASE 2 — PILOT P1", "=" * 60,
             f"Calibration {cal['cell_id']}: converged={cal['converged']} after {cal['steps_run']} steps "
             f"({cal['seconds_per_train_step'] * 1000:.1f} ms/step)"]
    if cal["best_val"]:
        lines.append(f"  best val: EM {cal['best_val']['exact_match']:.2f}%, token {cal['best_val']['token_accuracy']:.2f}%")
    if cal["test"]:
        lines.append(f"  test:     EM {cal['test']['exact_match']:.2f}%, token {cal['test']['token_accuracy']:.2f}%")
    lines.append(f"Fixed step budget: {report['fixed_step_budget']}")
    for r in report["floor_check"]:
        test = r["test"]
        lines.append(f"  {r['cell_id']}: test EM {test['exact_match']:.2f}%, token {test['token_accuracy']:.2f}%, "
                     f"chance-normalised {test['chance_normalised_token_accuracy']:.4f}, at_floor={r['at_floor']}, "
                     f"window coverage {r['test_window_coverage']}, {r['seconds_per_train_step'] * 1000:.1f} ms/step")
    lines.append(f"Floor effect at max T: {report['floor_effect_at_max_t']}")
    if report["grid_compute_estimate"]:
        lines.append(f"Estimated Tesseract-grid training time: {report['grid_compute_estimate']['total_training_hours']:.1f} h "
                     f"({report['grid_compute_estimate']['device']})")
    return "\n".join(lines)


# ============================================================================
# 4. Full grid (guarded)
# ============================================================================


def enumerate_grid_cells(grid: GridConfig, benchmark: BenchmarkConfig) -> List[CellSpec]:
    cells: List[CellSpec] = []
    for seed in benchmark.model_seeds:
        for t in benchmark.t_values:
            if "tesseract" in grid.families:
                for task in benchmark.tasks:
                    cells += [CellSpec("tesseract", k, benchmark.group, task, t, seed) for k in benchmark.k_values]
                if grid.include_ablation_groups:
                    for group in benchmark.ablation_groups:
                        cells += [CellSpec("tesseract", k, group, "main", t, seed) for k in benchmark.k_values]
            for family in ("unrolled", "param_matched"):
                if family in grid.families:
                    cells += [CellSpec(family, k, benchmark.group, "main", t, seed) for k in benchmark.k_values]
            if "width_scaled" in grid.families:
                cells.append(CellSpec("width_scaled", max(benchmark.k_values), benchmark.group, "main", t, seed))
    return cells


def run_grid(grid: GridConfig, config_path: Path, device: torch.device, run_dir: Path, confirm: bool,
             benchmark: Optional[BenchmarkConfig] = None, gates_run_dir: Optional[Path] = None,
             log: Callable[[str], None] = print) -> Dict:
    if not confirm:
        raise RuntimeError("The full Phase 2 grid requires explicit confirmation after pilot review (--confirm-full-grid).")
    benchmark = benchmark or load_benchmark(grid.benchmark)
    require_gates_passed(gates_run_dir or resolve_run_dir(DEFAULT_GATES_DIR))
    pilot_report_path = resolve_run_dir(grid.pilot_output_dir) / "pilot_report.json"
    if not pilot_report_path.is_file():
        raise RuntimeError(f"No pilot report at {pilot_report_path}; run and review pilot P1 first.")
    budget = json.loads(pilot_report_path.read_text(encoding="utf-8"))["fixed_step_budget"]
    if not budget:
        raise RuntimeError("Pilot P1 did not produce a fixed step budget.")

    training = TrainingConfig(grid.optimizer, grid.batch_size, budget, grid.eval_every, grid.eval_batch_size)
    results = []
    cells_dir = Path(run_dir) / "cells"
    for index, cell in enumerate(enumerate_grid_cells(grid, benchmark), start=1):
        cell_dir = cells_dir / cell.cell_id
        meta_path = cell_dir / METADATA_FILENAME
        if meta_path.is_file() and json.loads(meta_path.read_text())["status"] == "completed":
            results.append(json.loads((cell_dir / "result.json").read_text()))
            continue  # resume: completed cells are not re-run
        log(f"--- grid cell {index}: {cell.cell_id} ---")
        results.append(run_cell(_cell_config(grid.experiment, f"{grid.experiment.name}_{cell.cell_id}", cell_dir,
                                             grid.model, benchmark, training, cell), config_path, device, cell_dir, log=log))

    summary = aggregate(results, benchmark.tau, benchmark.k_values)
    with RunRecorder(run_dir, grid.experiment.name, grid, config_path, device, ["grid_summary.json"]) as run:
        write_json(run.path("grid_summary.json"), {"fixed_step_budget": budget, **summary})
        run.finish("COMPLETED", {"cells": len(results)})
    return summary
