"""Phase 2 orchestration.

1. ``run_verification_and_gates`` / ``run_amended_gates``: implementation-B
   verification, gates G1/G2, frozen checksummed splits, and sanity baselines.
   Nothing trains until the gates pass (or pass under a recorded amendment).
2. ``run_cell``: one fixed-step training run with full artifacts
   (checkpoints, curves, diagnostics).
3. ``run_pilot``: pilot P1. Calibrates the fixed step budget on T=1 / K=1,
   then runs the floor-check cells. Validation data only, never test.
4. The Phase 2 experiment stages (matrix, guards, analysis) live in
   ``phase2/experiment.py`` and use ``run_cell``.

All artifacts go under ``<runs root>/phase2/``; Phase 1 run directories are
never touched.
"""

import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import yaml

from models.block import TransformerBlock
from phase2.amendments import accepted_gate_verdicts, evaluate_amendment, verify_involution_shortcut
from phase2.config import (
    AmendedGatesRunConfig,
    AmendmentConfig,
    BenchmarkConfig,
    CapacityDiagnosticConfig,
    CellRunConfig,
    CellSpec,
    GatesRunConfig,
    PilotConfig,
    TrainingConfig,
    load_amendment,
    load_benchmark,
    resolve_model_variant,
)
from phase2.data import SplitStore, TrainingStream, seen_window_codes, window_coverage
from phase2.gates import run_gates
from phase2.groups import get_group
from phase2.metrics import bootstrap_ci, compute_metrics
from phase2.models import build_model
from phase2.sanity import sanity_baselines
from phase2.training import predict, train_fixed_steps
from phase2.verification import run_verification
from training.trainer import verify_bptt
from utils.paths import resolve_run_dir, runs_root
from utils.run_artifacts import METADATA_FILENAME, RunRecorder, write_json
from utils.seed import set_seed

DEFAULT_GATES_DIR = "phase2/gates_amendment01"  # the original FAIL run stays in phase2/gates
ACCEPTED_GATE_VERDICTS = accepted_gate_verdicts([load_amendment("phase2/amendment_01")])


def splits_dir() -> Path:
    return runs_root() / "phase2" / "splits"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _verify_gates_splits_sanity(benchmark: BenchmarkConfig, run: RunRecorder, log: Callable[[str], None]):
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
    return verification, gates, splits, sanity, oracle_ok


def run_verification_and_gates(config: GatesRunConfig, config_path: Path, run_dir: Path, benchmark: BenchmarkConfig,
                               device: torch.device, log: Callable[[str], None] = print) -> Dict:
    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, GATES_ARTIFACTS) as run:
        verification, gates, splits, sanity, oracle_ok = _verify_gates_splits_sanity(benchmark, run, log)
        passed = verification["passed"] and gates["passed"] and oracle_ok
        summary = _gates_summary(verification, gates, sanity, splits, passed)
        run.path("summary.txt").write_text(summary + "\n", encoding="utf-8")
        log(summary)
        run.finish("PASS" if passed else "FAIL", {"verification_passed": verification["passed"],
                                                   "gates_passed": gates["passed"], "oracle_ok": oracle_ok,
                                                   "num_splits": len(splits)})
    return {"passed": passed, "verification": verification, "gates": gates}


AMENDED_GATES_ARTIFACTS = GATES_ARTIFACTS + ["amendment_evaluation.json"]


def run_amended_gates(config: AmendedGatesRunConfig, config_path: Path, run_dir: Path, benchmark: BenchmarkConfig,
                      amendment: AmendmentConfig, device: torch.device,
                      training_dirs: Optional[List[Path]] = None, log: Callable[[str], None] = print) -> Dict:
    """Re-run verification and the unchanged gates, then evaluate the amendment separately.

    The original gate run directory is read, never written.
    """
    original_dir = resolve_run_dir(amendment.original_gates_output_dir)
    if Path(run_dir).resolve() == original_dir.resolve():
        raise RuntimeError("The amended gate run must not write into the original gate run directory.")
    original_report_path = original_dir / "gates_report.json"
    if not original_report_path.is_file():
        raise RuntimeError(f"Original gate report not found at {original_report_path}.")
    original_bytes = original_report_path.read_bytes()
    original_report = json.loads(original_bytes)
    original_meta = json.loads((original_dir / METADATA_FILENAME).read_text(encoding="utf-8"))

    if training_dirs is None:
        training_dirs = [resolve_run_dir("phase2/pilot_p1"), resolve_run_dir("phase2/grid")]
    training_present = [str(d) for d in training_dirs if (Path(d) / METADATA_FILENAME).is_file()]

    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, AMENDED_GATES_ARTIFACTS) as run:
        verification, gates, splits, sanity, oracle_ok = _verify_gates_splits_sanity(benchmark, run, log)
        shortcut_check = verify_involution_shortcut(benchmark.group, benchmark.n, 20000, benchmark.gates.seed)
        evaluation = evaluate_amendment(amendment, benchmark, verification, gates, sanity, original_report,
                                        shortcut_check, training_present)
        evaluation["original_gate_run"] = {
            "output_dir": amendment.original_gates_output_dir,
            "run_id": original_meta.get("run_id"),
            "verdict": original_meta.get("verdict"),
            "gates_report_sha256": hashlib.sha256(original_bytes).hexdigest(),
        }
        write_json(run.path("amendment_evaluation.json"), evaluation)

        raw_passed = verification["passed"] and gates["passed"] and oracle_ok
        lines = [_gates_summary(verification, gates, sanity, splits, raw_passed), "",
                 f"AMENDMENT {amendment.id} ({amendment.date}; recorded before training)",
                 "=" * 60,
                 f"Original gate run preserved: {evaluation['original_gate_run']['run_id']} "
                 f"(verdict {evaluation['original_gate_run']['verdict']}, "
                 f"gates_report sha256 {evaluation['original_gate_run']['gates_report_sha256'][:16]}…)",
                 f"T roles: anchor {list(amendment.anchor_t_values)}, diagnostic {list(amendment.diagnostic_t_values)}, "
                 f"primary depth {list(amendment.primary_depth_t_values)}",
                 f"Shortcut mechanism ({shortcut_check['elements_squaring_to_identity']}/60 elements square to e): "
                 f"token agreement {shortcut_check['token_agreement']:.4f}, "
                 f"agreement ⇔ s[i+1]²=e exactly: {shortcut_check['agreement_iff_condition']}"]
        for key, baseline in evaluation["shortcut_baselines"].items():
            if baseline:
                lines.append(f"T=2 shortcut baseline s[i]·s[i+2] on {key}: token {baseline['token_accuracy']:.2f}%, "
                             f"EM {baseline['exact_match']:.2f}%, chance-normalised "
                             f"{baseline['chance_normalised_token_accuracy']:.4f}, condition rate "
                             f"{baseline['condition_token_rate_percent']:.2f}%, exact iff condition: "
                             f"{baseline['agreement_iff_condition']}")
        for name, ok in evaluation["checks"].items():
            lines.append(f"  [{'✓' if ok else '✗'}] {name}")
        lines.append(f"RAW GATE VERDICT (unchanged): {evaluation['raw_gate_verdict']}")
        lines.append(f"AMENDED GATE VERDICT: {evaluation['amended_verdict']}")
        summary = "\n".join(lines)
        run.path("summary.txt").write_text(summary + "\n", encoding="utf-8")
        log(summary)
        run.finish(evaluation["amended_verdict"], {"raw_gate_verdict": evaluation["raw_gate_verdict"],
                                                    "amended_verdict": evaluation["amended_verdict"],
                                                    "amendment": amendment.id, "checks": evaluation["checks"]})
    return evaluation


def require_gates_passed(gates_run_dir: Path) -> Dict:
    meta_path = Path(gates_run_dir) / METADATA_FILENAME
    if not meta_path.is_file():
        raise RuntimeError(f"No gate run found at {gates_run_dir}; run the Phase 2 gate script first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "completed" or meta.get("verdict") not in ACCEPTED_GATE_VERDICTS:
        raise RuntimeError(f"Gates have not passed (status={meta.get('status')}, verdict={meta.get('verdict')}); "
                           "no model training is allowed.")
    return meta


# ============================================================================
# 2. One training cell
# ============================================================================

CELL_ARTIFACTS = ["result.json", "val_curve.csv", "best_checkpoint.pt", "final_checkpoint.pt"]
BPTT_DIAGNOSTIC_EXAMPLES = 64  # validation examples used for the post-training BPTT gradient check


def measure_block_calls(model: torch.nn.Module, n: int, device: torch.device) -> Dict[str, int]:
    """Count TransformerBlock instances and block executions in one forward pass."""
    blocks = [m for m in model.modules() if isinstance(m, TransformerBlock)]
    calls: List[int] = []
    handles = [block.register_forward_hook(lambda *_: calls.append(1)) for block in blocks]
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, n, dtype=torch.long, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {"transformer_block_instances": len(blocks), "block_calls_per_forward": len(calls)}


def _all_finite(state: Optional[Dict]) -> Optional[bool]:
    if state is None:
        return None
    return all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_floating_point(v))


def run_cell(config: CellRunConfig, config_path: Path, device: torch.device, run_dir: Path,
             stop_condition=None, evaluate_test: bool = True, compute_coverage: bool = True,
             provenance: Optional[Dict] = None, log: Callable[[str], None] = print) -> Dict:
    """Train one cell with the fixed-step protocol and record everything.

    Test policy: the test split is never touched during training. When
    ``evaluate_test`` is set, it is evaluated once after training on the
    best-validation checkpoint (``test``, primary) and once on the final
    checkpoint (``test_final``, sensitivity).
    """
    cell, benchmark, training = config.cell, config.benchmark, config.training
    group = get_group(cell.group)
    if config.model.vocab_size != group.order:
        raise ValueError(f"model vocab_size {config.model.vocab_size} != |{cell.group}| = {group.order}")
    if config.model.max_seq_len < benchmark.n:
        raise ValueError("model max_seq_len is smaller than the ring size n")

    wall_start = time.perf_counter()
    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, CELL_ARTIFACTS) as run:
        store = SplitStore(splits_dir(), benchmark.data_seed)
        val = store.get(cell.group, cell.task, "val", cell.t, benchmark.n, benchmark.val_size)
        test = store.get(cell.group, cell.task, "test", cell.t, benchmark.n, benchmark.test_size)

        set_seed(cell.seed)
        built = build_model(cell.family, cell.depth, config.model, benchmark.n)
        model = built.model.to(device)
        structure = measure_block_calls(model, benchmark.n, device)

        def make_stream() -> TrainingStream:
            # Test inputs are excluded from training (leakage guard); their labels are never read here.
            return TrainingStream(cell.group, cell.task, cell.t, benchmark.n, training.batch_size, benchmark.data_seed,
                                  cell.seed, excluded_inputs=[val.inputs, test.inputs])

        log(f"[{cell.cell_id}] params={built.parameter_count:,} blocks={structure['transformer_block_instances']} "
            f"block_calls/forward={structure['block_calls_per_forward']} max_steps={training.max_steps} "
            f"test_evaluated={evaluate_test}")
        outcome = train_fixed_steps(model, make_stream(), val, test if evaluate_test else None, training, device,
                                    group.order, stop_condition=stop_condition, log=log)

        torch.save(outcome.final_state, run.path("final_checkpoint.pt"))
        checkpoints = {"final": {"file": "final_checkpoint.pt", "step": outcome.steps_run,
                                 "sha256": _sha256_file(run.path("final_checkpoint.pt"))}}
        if outcome.best_state is not None:
            torch.save(outcome.best_state, run.path("best_checkpoint.pt"))
            checkpoints["best"] = {"file": "best_checkpoint.pt", "step": outcome.best_step,
                                   "selected_by": "max validation chance-normalised token accuracy (earliest on ties)",
                                   "sha256": _sha256_file(run.path("best_checkpoint.pt"))}

        bptt_x = torch.from_numpy(val.inputs[:BPTT_DIAGNOSTIC_EXAMPLES]).to(device)
        bptt_y = torch.from_numpy(val.targets[:BPTT_DIAGNOSTIC_EXAMPLES]).to(device)

        def bptt_report(label: str) -> Dict:
            report = verify_bptt(model, bptt_x, bptt_y)
            return {"evaluated_on": f"{label} checkpoint, first {BPTT_DIAGNOSTIC_EXAMPLES} validation examples",
                    "passed": report.passed, "failures": report.failures, "loss": report.loss,
                    "steps": [asdict(s) for s in report.steps]}

        bptt = bptt_report("best") if outcome.best_state is not None else None  # model holds the best checkpoint

        bptt_final, test_final, test_final_predictions = None, None, None
        if outcome.stopped_reason != "non_finite_loss":
            model.load_state_dict(outcome.final_state)
            bptt_final = bptt_report("final")
            if evaluate_test:
                test_final_predictions = predict(model, test.inputs, device, training.eval_batch_size)
                test_final = compute_metrics(test_final_predictions, test.targets, group.order)

        coverage, coverage_reference = None, None
        if compute_coverage and outcome.steps_run > 0:
            reference = test if evaluate_test else val
            coverage_reference = "test inputs" if evaluate_test else "validation inputs"
            coverage = window_coverage(reference.inputs, cell.t, group.order, seen_window_codes(make_stream, outcome.steps_run))

        def bootstrap(predictions) -> Optional[Dict]:
            if predictions is None:
                return None
            equal = predictions == test.targets
            return {"exact_match": bootstrap_ci(equal.all(axis=1) * 100.0, seed=cell.seed),
                    "token_accuracy": bootstrap_ci(equal.mean(axis=1) * 100.0, seed=cell.seed)}

        test_ci = bootstrap(outcome.test_predictions)
        test_final_ci = bootstrap(test_final_predictions)

        last_eval = next((row for row in reversed(outcome.history) if "val_loss" in row), None)
        result = {
            "cell": asdict(cell),
            "cell_id": cell.cell_id,
            "model": built.describe(),
            **structure,
            "max_steps": training.max_steps,
            "steps_run": outcome.steps_run,
            "stopped_reason": outcome.stopped_reason,
            "non_finite": {
                "loss": outcome.stopped_reason == "non_finite_loss",
                "gradient_norm_steps": outcome.gradient_norm_stats["non_finite_steps"],
                "final_parameters_finite": _all_finite(outcome.final_state),
                "best_parameters_finite": _all_finite(outcome.best_state),
            },
            "gradient_norm_stats": outcome.gradient_norm_stats,
            "final_train_loss_interval_mean": last_eval["train_loss_interval_mean"] if last_eval else None,
            "best_step": outcome.best_step,
            "best_val": outcome.best_val.to_dict() if outcome.best_val else None,
            "best_val_loss": outcome.best_val_loss,
            "final_val": outcome.final_val.to_dict() if outcome.final_val else None,
            "final_val_loss": outcome.final_val_loss,
            "test_evaluated": evaluate_test,
            "test": outcome.test.to_dict() if outcome.test else None,
            "test_bootstrap_ci95": test_ci,
            "test_final": test_final.to_dict() if test_final else None,
            "test_final_bootstrap_ci95": test_final_ci,
            "test_checkpoint_policy": ("after training only: best-validation checkpoint (primary) and final checkpoint "
                                       "(sensitivity)" if evaluate_test else "test split not evaluated"),
            "bptt_on_best_checkpoint": bptt,
            "bptt_on_final_checkpoint": bptt_final,
            "checkpoints": checkpoints,
            "train_seconds": outcome.train_seconds,
            "eval_seconds": outcome.eval_seconds,
            "seconds_per_train_step": outcome.seconds_per_train_step,
            "rejected_training_rows": outcome.rejected_training_rows,
            "window_coverage": coverage,
            "window_coverage_reference": coverage_reference,
            "split_sha256": {"val": val.sha256, **({"test": test.sha256} if evaluate_test else {})},
            "provenance": provenance,
        }

        fieldnames = sorted({key for row in outcome.history for key in row}, key=lambda k: (k != "step", k))
        with open(run.path("val_curve.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(outcome.history)
        result["total_wall_seconds"] = time.perf_counter() - wall_start
        write_json(run.path("result.json"), result)

        verdict = "FAIL" if outcome.stopped_reason == "non_finite_loss" else "COMPLETED"
        run.finish(verdict, {key: result[key] for key in ("cell_id", "steps_run", "stopped_reason", "best_step",
                                                          "best_val", "non_finite", "total_wall_seconds")})
    return result


def _cell_config(base_experiment, name: str, run_dir: Path, model, benchmark, training, cell) -> CellRunConfig:
    return CellRunConfig(experiment=replace(base_experiment, name=name, output_dir=str(run_dir), seed=cell.seed),
                         model=model, benchmark=benchmark, training=training, cell=cell)


# ============================================================================
# 3. Pilot P1
# ============================================================================

PILOT_ARTIFACTS = ["pilot_report.json", "summary.txt", "calibration", "cells"]


def _fit_step_time(points: List[Dict]) -> Optional[Dict]:
    ks = np.array([p["block_calls_per_forward"] for p in points], dtype=float)
    secs = np.array([p["seconds_per_train_step"] for p in points], dtype=float)
    if len(set(ks)) < 2:
        return None
    slope, intercept = np.polyfit(ks, secs, 1)
    return {"intercept_s": float(intercept), "per_block_execution_s": float(slope)}


def _cell_report(result: Dict, floor_threshold: Optional[float] = None) -> Dict:
    best_val = result["best_val"]
    report = {key: result[key] for key in (
        "cell_id", "cell", "model", "transformer_block_instances", "block_calls_per_forward", "max_steps", "steps_run",
        "stopped_reason", "non_finite", "gradient_norm_stats", "final_train_loss_interval_mean", "best_step",
        "best_val", "best_val_loss", "final_val", "final_val_loss", "bptt_on_best_checkpoint", "checkpoints",
        "seconds_per_train_step", "train_seconds", "eval_seconds", "total_wall_seconds", "window_coverage",
        "window_coverage_reference", "test_evaluated")}
    if floor_threshold is not None:
        score = best_val["chance_normalised_token_accuracy"] if best_val else None
        report["at_floor"] = score is None or score < floor_threshold
    return report


def run_pilot(config: PilotConfig, config_path: Path, device: torch.device, run_dir: Path,
              benchmark: Optional[BenchmarkConfig] = None, gates_run_dir: Optional[Path] = None,
              log: Callable[[str], None] = print) -> Dict:
    """Pilot P1. Uses validation data only (no test evaluation) and never overwrites an earlier pilot run."""
    if (Path(run_dir) / METADATA_FILENAME).exists():
        raise RuntimeError(f"A pilot run already exists at {run_dir}; it is preserved. Use a new --output-dir.")
    benchmark = benchmark or load_benchmark(config.benchmark)
    gates_meta = require_gates_passed(gates_run_dir or resolve_run_dir(DEFAULT_GATES_DIR))
    seed = config.experiment.seed
    cal = config.calibration

    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, PILOT_ARTIFACTS) as run:
        run_meta = json.loads((Path(run_dir) / METADATA_FILENAME).read_text(encoding="utf-8"))
        log("=== Pilot P1 stage A: step-budget calibration ===")
        cal_cell = CellSpec(cal.family, cal.depth, benchmark.group, cal.task, cal.t, seed)
        cal_training = TrainingConfig(config.optimizer, config.batch_size, cal.max_steps, config.eval_every,
                                      config.eval_batch_size)
        cal_dir = run.path("calibration")
        cal_result = run_cell(
            _cell_config(config.experiment, f"{config.experiment.name}_calibration", cal_dir, config.model, benchmark,
                         cal_training, cal_cell),
            config_path, device, cal_dir, evaluate_test=False,
            stop_condition=lambda val: val.exact_match >= cal.converge_exact_match, log=log,
        )
        converged = cal_result["stopped_reason"] == "stop_condition"
        budget = cal.budget_multiplier * cal_result["steps_run"] if converged else None
        log(f"Calibration converged={converged} after {cal_result['steps_run']} steps → fixed budget {budget}")

        floor_results = []
        if converged:
            log(f"=== Pilot P1 stage B: fixed budget of {budget} steps, no early stopping ===")
            training = replace(cal_training, max_steps=budget)
            for pilot_cell in config.floor_cells:
                cell = CellSpec(pilot_cell.family, pilot_cell.depth, benchmark.group, pilot_cell.task, pilot_cell.t, seed)
                cell_dir = run.path("cells") / cell.cell_id
                floor_results.append(run_cell(
                    _cell_config(config.experiment, f"{config.experiment.name}_{cell.cell_id}", cell_dir,
                                 config.model, benchmark, training, cell),
                    config_path, device, cell_dir, evaluate_test=False, log=log))

        stage_b = [_cell_report(r, config.floor_threshold) for r in floor_results]
        step_time = _fit_step_time([cal_result, *floor_results])
        grid_estimate = None
        if step_time and budget:
            per_run = {k: (step_time["intercept_s"] + step_time["per_block_execution_s"] * k) * budget
                       for k in benchmark.k_values}
            runs_per_k = len(benchmark.tasks) * len(benchmark.t_values) * len(benchmark.model_seeds)
            grid_estimate = {
                "scope": "Tesseract main+controls grid only (3 tasks × 4 T × 4 K × 3 seeds); training steps only, "
                         "excluding evaluation, baselines and the Z60 ladder; linear fit of step time on block calls",
                "training_hours_per_run_by_k": {k: v / 3600 for k, v in per_run.items()},
                "total_training_hours": sum(v * runs_per_k for v in per_run.values()) / 3600,
                "device": str(device),
            }

        floor_hit = [r["cell_id"] for r in stage_b if r["at_floor"]]
        report = {
            "gates_run_id": gates_meta.get("run_id"),
            "gates_verdict": gates_meta.get("verdict"),
            "protocol": {
                "stage_a": f"{cal.family} K={cal.depth} {cal.task} T={cal.t}; stop when validation exact match >= "
                           f"{cal.converge_exact_match}% (evaluated every {config.eval_every} steps); safety limit "
                           f"{cal.max_steps} steps",
                "budget_rule": f"exactly {cal.budget_multiplier} × steps-to-criterion",
                "stage_b": "fixed budget, no early stopping; best checkpoint by validation only",
                "floor": f"best-checkpoint validation chance-normalised token accuracy < {config.floor_threshold}",
                "test_split": "not evaluated in P1 (reserved for the grid); only its inputs are excluded from training",
                "optimizer": asdict(config.optimizer), "batch_size": config.batch_size, "tau": benchmark.tau,
            },
            "compute_environment": {
                "device": str(device),
                "cpu_only": not torch.cuda.is_available(),
                "torch_version": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "torch_num_threads": torch.get_num_threads(),
                "git": run_meta.get("git"),
                "command": run_meta.get("command"),
                "note": "CPU-only environment: all P1 timings and results are CPU results; no GPU was used.",
            },
            "stage_a": {**_cell_report(cal_result), "converged": converged, "steps_to_criterion": cal_result["steps_run"] if converged else None},
            "fixed_step_budget": budget,
            "stage_b": stage_b,
            "floor_hit_cells": floor_hit,
            "action": ("STOP: stage A did not reach the criterion" if not converged else
                       "STOP AND DIAGNOSE: floor reached" if floor_hit else "STOP: pilot complete; awaiting review"),
            "step_time_fit": step_time,
            "grid_compute_estimate": grid_estimate,
        }
        write_json(run.path("pilot_report.json"), report)
        summary = _pilot_summary(report)
        run.path("summary.txt").write_text(summary + "\n", encoding="utf-8")
        log(summary)
        verdict = "FAIL" if not converged else ("COMPLETED_FLOOR_HIT" if floor_hit else "COMPLETED")
        run.finish(verdict, {"fixed_step_budget": budget, "converged": converged, "floor_hit_cells": floor_hit})
    return report


def _pilot_summary(report: Dict) -> str:
    def cell_line(r: Dict) -> str:
        v = r["best_val"] or {}
        pos = v.get("per_position_accuracy") or []
        bptt = r["bptt_on_best_checkpoint"]
        return (f"  {r['cell_id']}: steps {r['steps_run']} ({r['stopped_reason']}), best step {r['best_step']} | "
                f"train loss {r['final_train_loss_interval_mean']:.4f} | val loss {r['best_val_loss']:.4f} | "
                f"val tok {v.get('token_accuracy', float('nan')):.2f}% | norm {v.get('chance_normalised_token_accuracy', float('nan')):.4f} | "
                f"val EM {v.get('exact_match', float('nan')):.2f}% | per-position {min(pos, default=float('nan')):.1f}–"
                f"{max(pos, default=float('nan')):.1f}% | grad mean/max {r['gradient_norm_stats']['mean']:.3f}/"
                f"{r['gradient_norm_stats']['max']:.3f} | non-finite {r['non_finite']} | block calls "
                f"{r['block_calls_per_forward']} | {r['model']['parameter_count']:,} params | "
                f"{r['seconds_per_train_step'] * 1000:.1f} ms/step, total {r['total_wall_seconds']:.0f} s | "
                f"BPTT {'pass' if bptt and bptt['passed'] else 'FAIL/none'}"
                + (f" | at_floor={r['at_floor']}" if "at_floor" in r else ""))

    lines = ["PHASE 2 — PILOT P1", "=" * 60, f"Gates: {report['gates_run_id']} ({report['gates_verdict']})",
             f"Environment: {report['compute_environment']['device']}, CPU-only={report['compute_environment']['cpu_only']}, "
             f"torch {report['compute_environment']['torch_version']}",
             "Stage A:", cell_line(report["stage_a"]),
             f"  converged={report['stage_a']['converged']} steps_to_criterion={report['stage_a']['steps_to_criterion']}",
             f"Fixed step budget: {report['fixed_step_budget']}"]
    if report["stage_b"]:
        lines.append("Stage B:")
        lines += [cell_line(r) for r in report["stage_b"]]
    lines.append(f"Floor hit: {report['floor_hit_cells'] or 'none'}")
    if report["grid_compute_estimate"]:
        lines.append(f"Estimated Tesseract-grid training time: {report['grid_compute_estimate']['total_training_hours']:.1f} h "
                     f"({report['grid_compute_estimate']['device']})")
    lines.append(f"ACTION: {report['action']}")
    return "\n".join(lines)


# ============================================================================
# 3b. P1b capacity / learnability diagnostic (§9)
# ============================================================================

DIAGNOSTIC_ARTIFACTS = ["diagnostic_report.json", "summary.txt", "cells"]
CURVE_COLUMNS = ("step", "train_loss_interval_mean", "val_loss", "val_token_accuracy",
                 "val_chance_normalised_token_accuracy", "val_exact_match", "gradient_norm_interval_max")


def _read_curve(cell_dir: Path) -> List[Dict]:
    with open(Path(cell_dir) / "val_curve.csv", newline="", encoding="utf-8") as f:
        return [{key: float(row[key]) for key in CURVE_COLUMNS if row.get(key) not in (None, "")}
                for row in csv.DictReader(f)]


def _headline(result: Dict) -> Dict:
    best, final = result["best_val"] or {}, result["final_val"] or {}
    return {
        "parameter_count": result["model"]["parameter_count"],
        "forward_flops_per_sequence": result["model"]["forward_flops_per_sequence"],
        "block_calls_per_forward": result["block_calls_per_forward"],
        "steps_run": result["steps_run"],
        "best_step": result["best_step"],
        "best_val_chance_normalised": best.get("chance_normalised_token_accuracy"),
        "final_val_chance_normalised": final.get("chance_normalised_token_accuracy"),
        "best_val_loss": result["best_val_loss"],
        "final_val_loss": result["final_val_loss"],
        "final_val_token_accuracy": final.get("token_accuracy"),
        "final_val_exact_match": final.get("exact_match"),
    }


def _difference(a: Dict, b: Dict) -> Dict:
    """a − b for the numeric headline metrics."""
    keys = ("best_val_chance_normalised", "final_val_chance_normalised", "best_val_loss", "final_val_loss",
            "final_val_token_accuracy", "final_val_exact_match")
    return {key: (a[key] - b[key]) if a.get(key) is not None and b.get(key) is not None else None for key in keys}


def run_capacity_diagnostic(config: CapacityDiagnosticConfig, config_path: Path, device: torch.device, run_dir: Path,
                            benchmark: Optional[BenchmarkConfig] = None, gates_run_dir: Optional[Path] = None,
                            reference_dir: Optional[Path] = None, log: Callable[[str], None] = print) -> Dict:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"{run_dir} already contains results; it is preserved. Use a new output directory.")
    benchmark = benchmark or load_benchmark(config.benchmark)
    gates_meta = require_gates_passed(gates_run_dir or resolve_run_dir(DEFAULT_GATES_DIR))
    seed = config.experiment.seed
    reference_dir = Path(reference_dir or resolve_run_dir(config.reference_pilot_output_dir))

    # P1 references: read-only, fingerprinted.
    references = {}
    for k in config.k_values:
        cell = CellSpec("tesseract", k, benchmark.group, "main", config.t, seed)
        path = reference_dir / "cells" / cell.cell_id / "result.json"
        if path.is_file():
            result = json.loads(path.read_bytes())
            ref_model = yaml.safe_load((path.parent / "config.yaml").read_text(encoding="utf-8"))["model"]
            references[k] = {"result_sha256": _sha256_file(path), "headline": _headline(result), "model_config": ref_model,
                             "curve": _read_curve(path.parent)}

    training = TrainingConfig(config.optimizer, config.batch_size, config.max_steps, config.eval_every,
                              config.eval_batch_size)
    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, DIAGNOSTIC_ARTIFACTS) as run:
        run_meta = json.loads((run_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
        runs: Dict[str, Dict[int, Dict]] = {}
        model_configs = {}
        for variant in config.models:
            model_cfg = resolve_model_variant(variant, config.vocab_size)
            model_configs[variant.name] = asdict(model_cfg)
            for k in config.k_values:
                cell = CellSpec("tesseract", k, benchmark.group, "main", config.t, seed)
                cell_dir = run.path("cells") / variant.name / cell.cell_id
                log(f"=== P1b: model={variant.name} ({variant.base}) T={config.t} K={k} steps={config.max_steps} ===")
                result = run_cell(
                    _cell_config(config.experiment, f"{config.experiment.name}_{variant.name}_{cell.cell_id}", cell_dir,
                                 model_cfg, benchmark, training, cell),
                    config_path, device, cell_dir, evaluate_test=False, compute_coverage=False, log=log)
                report = _cell_report(result, config.floor_threshold)
                final_cn = (result["final_val"] or {}).get("chance_normalised_token_accuracy")
                report.update(
                    model_variant=variant.name, model_base=variant.base, curve=_read_curve(cell_dir),
                    at_floor_best=report.pop("at_floor"),
                    at_floor_final=final_cn is None or final_cn < config.floor_threshold,
                    headline=_headline(result),
                )
                runs.setdefault(variant.name, {})[k] = report

        names = [m.name for m in config.models]
        k_low, k_high = min(config.k_values), max(config.k_values)
        comparisons = {
            "within_width_K_high_minus_K_low": {
                name: {"k_high": k_high, "k_low": k_low,
                       "parameter_count_equal": runs[name][k_high]["headline"]["parameter_count"]
                       == runs[name][k_low]["headline"]["parameter_count"],
                       **_difference(runs[name][k_high]["headline"], runs[name][k_low]["headline"])}
                for name in names
            },
            "across_width_last_minus_first": {
                f"K={k}": {"larger": names[-1], "smaller": names[0],
                           **_difference(runs[names[-1]][k]["headline"], runs[names[0]][k]["headline"])}
                for k in config.k_values
            },
        }
        small_name = next((m.name for m in config.models if m.base == "prototype_small"), None)
        if small_name is not None:
            comparisons["small_20k_minus_P1_reference"] = {
                f"K={k}": {"reference_model_config_identical": references[k]["model_config"] == model_configs[small_name],
                           "reference_steps": references[k]["headline"]["steps_run"],
                           **_difference(runs[small_name][k]["headline"], references[k]["headline"])}
                for k in config.k_values if k in references
            }

        diagnostic = {
            "section": "PHASE2_BENCHMARK_DESIGN.md §9",
            "scope": "capacity/learnability diagnostic; NOT evidence for the Tesseract depth hypothesis",
            "gates_run_id": gates_meta.get("run_id"),
            "gates_verdict": gates_meta.get("verdict"),
            "settings": {"t": config.t, "n": benchmark.n, "k_values": list(config.k_values), "seed": seed,
                         "optimizer": asdict(config.optimizer), "batch_size": config.batch_size,
                         "eval_every": config.eval_every, "max_steps": config.max_steps, "early_stopping": False,
                         "test_split_evaluated": False, "floor_threshold": config.floor_threshold,
                         "chance_level_val_loss_ln_vocab": float(np.log(config.vocab_size))},
            "model_configs": model_configs,
            "compute_environment": {"device": str(device), "cpu_only": not torch.cuda.is_available(),
                                    "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
                                    "torch_num_threads": torch.get_num_threads(), "git": run_meta.get("git"),
                                    "command": run_meta.get("command")},
            "runs": {name: {str(k): r for k, r in per_k.items()} for name, per_k in runs.items()},
            "p1_references_1500_steps": {str(k): v for k, v in references.items()},
            "comparisons": comparisons,
            "single_seed_note": "one seed per cell: differences are descriptive, not statistical claims",
        }
        write_json(run.path("diagnostic_report.json"), diagnostic)
        summary = _diagnostic_summary(diagnostic)
        run.path("summary.txt").write_text(summary + "\n", encoding="utf-8")
        log(summary)
        floor_runs = [f"{n}/K={k}" for n, per_k in runs.items() for k, r in per_k.items() if r["at_floor_final"]]
        run.finish("COMPLETED", {"final_checkpoint_at_floor": floor_runs})
    return diagnostic


def _diagnostic_summary(d: Dict) -> str:
    lines = ["PHASE 2 — P1b CAPACITY / LEARNABILITY DIAGNOSTIC (not depth-hypothesis evidence)", "=" * 72,
             f"T={d['settings']['t']} n={d['settings']['n']} seed={d['settings']['seed']} steps={d['settings']['max_steps']} "
             f"(no early stopping, validation only) | chance val loss ln(V)={d['settings']['chance_level_val_loss_ln_vocab']:.4f} | "
             f"device {d['compute_environment']['device']} CPU-only={d['compute_environment']['cpu_only']}"]
    for name, per_k in d["runs"].items():
        for k, r in per_k.items():
            h = r["headline"]
            curve = r["curve"]
            marks = [row for row in curve if int(row["step"]) in (250, 1500, 5000, 10000, 15000, 20000)]
            lines.append(
                f"{name} K={k}: {h['parameter_count']:,} params, {h['block_calls_per_forward']} block calls, "
                f"final val loss {h['final_val_loss']:.4f}, final norm {h['final_val_chance_normalised']:.4f}, "
                f"final tok {h['final_val_token_accuracy']:.2f}%, final EM {h['final_val_exact_match']:.2f}% | "
                f"best norm {h['best_val_chance_normalised']:.4f} @ {h['best_step']} | floor best/final "
                f"{r['at_floor_best']}/{r['at_floor_final']} | grad mean/max {r['gradient_norm_stats']['mean']:.3f}/"
                f"{r['gradient_norm_stats']['max']:.3f} | BPTT {'pass' if (r['bptt_on_best_checkpoint'] or {}).get('passed') else 'FAIL'} | "
                f"{r['seconds_per_train_step'] * 1000:.1f} ms/step, {r['total_wall_seconds'] / 60:.1f} min")
            lines.append("    curve (step: val loss / norm / EM): " + ", ".join(
                f"{int(row['step'])}: {row['val_loss']:.3f}/{row['val_chance_normalised_token_accuracy']:.3f}/"
                f"{row['val_exact_match']:.1f}" for row in marks))
    for section, entries in d["comparisons"].items():
        lines.append(f"{section}:")
        for key, entry in entries.items():
            lines.append(f"  {key}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in entry.items()))
    return "\n".join(lines)
