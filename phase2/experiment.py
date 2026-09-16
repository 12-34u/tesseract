"""Phase 2 experiment: matrix enumeration and guarded stage / single-cell execution.

PHASE2_BENCHMARK_DESIGN.md §10 (Amendment 02, draft until frozen). Training starts
only when all of the following hold:

1. ``model_base`` and ``max_steps`` are no longer PENDING_P1B;
2. Amendment 02 is frozen;
3. the read-only benchmark integrity check passes;
4. the P1b run has completed;
5. the stage (or cell) is explicitly confirmed.

Completed cells are never re-run, and incomplete cell directories are never
overwritten. Stage summaries report validation metrics only. Test metrics are
analysed by ``analyze_experiment``, and only once every planned cell is
complete.
"""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from phase2.config import (
    Amendment02Config,
    AmendmentConfig,
    BenchmarkConfig,
    CellRunConfig,
    CellSpec,
    ModelVariant,
    Phase2ExperimentConfig,
    TrainingConfig,
    load_amendment,
    load_amendment_02,
    load_benchmark,
    resolve_model_variant,
)
from phase2.inference import analyze, stage_review
from phase2.integrity import check_benchmark_integrity
from phase2.plots import load_curves, write_report
from phase2.runner import DEFAULT_GATES_DIR, run_cell, splits_dir
from utils.config import resolve_config_path
from utils.paths import resolve_run_dir
from utils.run_artifacts import METADATA_FILENAME, RunRecorder, write_json

P1B_RUN_DIR = "phase2/p1b_capacity_diagnostic"
STAGE_ARTIFACTS = ["stage_summary.json", "summary.txt"]
ANALYSIS_ARTIFACTS = ["analysis.json", "report.md", "fig1_delta_vs_t.png", "fig2_score_vs_k.png",
                      "fig3_score_vs_flops.png", "fig4_learning_curves.png"]


# ============================================================================
# Matrix
# ============================================================================


def width_label(model_base: str) -> str:
    return Path(model_base).stem


def enumerate_stage(config: Phase2ExperimentConfig, stage_name: str, group: str) -> List[CellSpec]:
    stage = config.stage(stage_name)
    return [CellSpec(stage.family, depth, group, task, t, seed)
            for seed in stage.seeds for task in stage.tasks for t in stage.t_values for depth in stage.depths]


def enumerate_matrix(config: Phase2ExperimentConfig, group: str) -> Dict[str, List[CellSpec]]:
    matrix = {stage.name: enumerate_stage(config, stage.name, group) for stage in config.stages}
    ids = [cell.cell_id for cells in matrix.values() for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("the same cell appears in more than one stage")
    return matrix


def cell_directory(run_dir: Path, model_base: str, cell: CellSpec) -> Path:
    return Path(run_dir) / "cells" / width_label(model_base) / cell.cell_id


def _amendment_name(config: Phase2ExperimentConfig, marker: str) -> str:
    names = [a for a in config.amendments if marker in a]
    if len(names) != 1:
        raise ValueError(f"expected exactly one {marker} entry in amendments, got {names}")
    return names[0]


def amendment_01_of(config: Phase2ExperimentConfig) -> AmendmentConfig:
    return load_amendment(_amendment_name(config, "amendment_01"))


def amendment_02_of(config: Phase2ExperimentConfig) -> Amendment02Config:
    return load_amendment_02(_amendment_name(config, "amendment_02"))


def _sha256(path: Path) -> Optional[str]:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# ============================================================================
# Guards
# ============================================================================


def check_preconditions(config: Phase2ExperimentConfig, amendment02: Amendment02Config, integrity: Dict,
                        p1b_run_dir: Path, confirm: bool) -> Dict:
    config.require_frozen()
    if not amendment02.frozen:
        raise RuntimeError("Amendment 02 is still a DRAFT. It must be frozen, after approval of the model width and "
                           "budget, before any Phase 2 training.")
    if not integrity.get("passed"):
        failed = [name for name, ok in integrity.get("checks", {}).items() if not ok]
        raise RuntimeError(f"Benchmark integrity check failed: {failed}")
    meta_path = Path(p1b_run_dir) / METADATA_FILENAME
    p1b_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if p1b_meta.get("status") != "completed":
        raise RuntimeError(f"P1b has not completed (status={p1b_meta.get('status')}); Phase 2 training is blocked.")
    if not confirm:
        raise RuntimeError("Phase 2 training requires explicit confirmation (--confirm-stage).")
    return p1b_meta


def _prepare(config: Phase2ExperimentConfig, config_path: Path, confirm: bool, benchmark: Optional[BenchmarkConfig],
             amendment02: Optional[Amendment02Config], integrity: Optional[Dict], p1b_run_dir: Optional[Path]
             ) -> Tuple[BenchmarkConfig, Amendment02Config, Dict]:
    config.require_frozen()  # cheap check first
    benchmark = benchmark or load_benchmark(config.benchmark)
    amendment02 = amendment02 or amendment_02_of(config)
    if integrity is None:
        integrity = check_benchmark_integrity(benchmark, amendment_01_of(config), resolve_run_dir(DEFAULT_GATES_DIR),
                                              splits_dir())
    p1b_run_dir = Path(p1b_run_dir or resolve_run_dir(P1B_RUN_DIR))
    p1b_meta = check_preconditions(config, amendment02, integrity, p1b_run_dir, confirm)
    provenance = {
        "experiment_config_sha256": _sha256(config_path),
        "amendments": list(config.amendments),
        "amendment_02_status": amendment02.status,
        "amendment_02_sha256": _sha256(resolve_config_path(_amendment_name(config, "amendment_02"))),
        "integrity_passed": True,
        "amended_gates_run_id": integrity.get("details", {}).get("amended_gates_run_id"),
        "p1b_run_id": p1b_meta.get("run_id"),
        "p1b_report_sha256": _sha256(p1b_run_dir / "diagnostic_report.json"),
        "model_base": config.model_base,
        "max_steps": config.max_steps,
    }
    return benchmark, amendment02, provenance


# ============================================================================
# Execution
# ============================================================================


def _execute_cell(config: Phase2ExperimentConfig, config_path: Path, device: torch.device, run_dir: Path,
                  benchmark: BenchmarkConfig, cell: CellSpec, provenance: Dict,
                  log: Callable[[str], None]) -> Tuple[Dict, str]:
    cell_dir = cell_directory(run_dir, config.model_base, cell)
    meta_path = cell_dir / METADATA_FILENAME
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "completed" and (cell_dir / "result.json").is_file():
            if meta.get("verdict") == "FAIL":
                raise RuntimeError(
                    f"{cell_dir} holds a FAILED run ({meta.get('run_id')}): training stopped on a non-finite loss, so "
                    "the cell has no usable result. It is preserved; inspect it and decide explicitly before "
                    "re-running. It is not silently treated as complete.")
            log(f"[skip] {cell.cell_id}: already completed ({meta.get('run_id')})")
            return json.loads((cell_dir / "result.json").read_text(encoding="utf-8")), "skipped_completed"
        raise RuntimeError(f"{cell_dir} holds an incomplete run (status={meta.get('status')}). It is preserved; "
                           "inspect it before re-running.")
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise RuntimeError(f"{cell_dir} is not empty and has no run metadata; refusing to overwrite.")

    model_config = resolve_model_variant(ModelVariant(width_label(config.model_base), config.model_base), config.vocab_size)
    training = TrainingConfig(config.optimizer, config.batch_size, config.max_steps, config.eval_every,
                              config.eval_batch_size)
    cell_config = CellRunConfig(
        experiment=replace(config.experiment, name=f"{config.experiment.name}_{cell.cell_id}", output_dir=str(cell_dir),
                           seed=cell.seed),
        model=model_config, benchmark=benchmark, training=training, cell=cell)
    result = run_cell(cell_config, config_path, device, cell_dir, evaluate_test=True, compute_coverage=False,
                      provenance=provenance, log=log)
    return result, "ran"


def _validation_row(result: Dict) -> Dict:
    best, final = result.get("best_val") or {}, result.get("final_val") or {}
    return {
        "cell_id": result["cell_id"], **{k: result["cell"][k] for k in ("family", "task", "t", "depth", "seed")},
        "steps_run": result["steps_run"], "stopped_reason": result["stopped_reason"], "best_step": result["best_step"],
        "best_val_chance_normalised": best.get("chance_normalised_token_accuracy"),
        "best_val_exact_match": best.get("exact_match"), "best_val_loss": result.get("best_val_loss"),
        "final_val_chance_normalised": final.get("chance_normalised_token_accuracy"),
        "final_val_loss": result.get("final_val_loss"), "non_finite": result["non_finite"],
        "block_calls_per_forward": result["block_calls_per_forward"],
        "parameter_count": result["model"]["parameter_count"], "head_dim": result["model"].get("head_dim"),
        "forward_flops_per_sequence": result["model"]["forward_flops_per_sequence"],
        "seconds_per_train_step": result["seconds_per_train_step"], "total_wall_seconds": result.get("total_wall_seconds"),
    }


def _number(value, spec: str = ".4f", missing: str = "n/a") -> str:
    """Format a metric that may be absent.

    A cell whose loss goes non-finite before the first evaluation has no
    best_val/final_val at all. Formatting None with a float spec raises, which
    would destroy the stage summary after the training itself had finished.
    """
    return missing if value is None else format(value, spec)


def _stage_text(summary: Dict) -> str:
    lines = [f"PHASE 2 STAGE {summary['stage']['name']} — validation metrics only", "=" * 70,
             f"model_base={summary['model_base']} max_steps={summary['max_steps']} cells={len(summary['cells'])}"]
    for row in summary["validation_only"]:
        diverged = " [NON-FINITE LOSS]" if (row.get("non_finite") or {}).get("loss") else ""
        lines.append(f"  {row['cell_id']}: best val norm {_number(row['best_val_chance_normalised'])} "
                     f"@ {row['best_step'] if row['best_step'] is not None else 'n/a'}, "
                     f"final val norm {_number(row['final_val_chance_normalised'])}, "
                     f"final val loss {_number(row['final_val_loss'])}, "
                     f"calls {row['block_calls_per_forward']}, {row['parameter_count']:,} params, "
                     f"{_number(row['seconds_per_train_step'] * 1000, '.1f')} ms/step{diverged}")
    if "stage1_review" in summary:
        review = summary["stage1_review"]
        lines.append(f"Stage review (T={review['t']}): {review['status']} → {review['action']}")
    return "\n".join(lines)


def run_stage(config: Phase2ExperimentConfig, config_path: Path, device: torch.device, run_dir: Path, stage_name: str,
              confirm: bool, benchmark: Optional[BenchmarkConfig] = None, amendment02: Optional[Amendment02Config] = None,
              integrity: Optional[Dict] = None, p1b_run_dir: Optional[Path] = None,
              log: Callable[[str], None] = print) -> Dict:
    stage = config.stage(stage_name)
    benchmark, amendment02, provenance = _prepare(config, config_path, confirm, benchmark, amendment02, integrity,
                                                  p1b_run_dir)
    results, statuses = [], {}
    for cell in enumerate_stage(config, stage_name, benchmark.group):
        result, status = _execute_cell(config, config_path, device, run_dir, benchmark, cell, provenance, log)
        results.append(result)
        statuses[cell.cell_id] = status

    summary = {
        "stage": asdict(stage),
        "model_base": config.model_base,
        "max_steps": config.max_steps,
        "cells": statuses,
        "validation_only": [_validation_row(r) for r in results],
        "note": "test metrics are not summarised per stage; they are analysed once all planned cells are complete",
    }
    review = amendment02.stage1_review
    if stage.family == "tesseract" and "main" in stage.tasks and review.t in stage.t_values:
        summary["stage1_review"] = stage_review(results, review)

    stage_dir = Path(run_dir) / "stages" / stage_name
    with RunRecorder(stage_dir, f"{config.experiment.name}_stage_{stage_name}", config, config_path, device,
                     STAGE_ARTIFACTS) as run:
        write_json(run.path("stage_summary.json"), summary)
        text = _stage_text(summary)
        run.path("summary.txt").write_text(text + "\n", encoding="utf-8")
        log(text)
        run.finish("COMPLETED", {"cells": len(results), "stage1_review": summary.get("stage1_review")})
    return summary


def run_one_cell(config: Phase2ExperimentConfig, config_path: Path, device: torch.device, run_dir: Path, cell_id: str,
                 confirm: bool, benchmark: Optional[BenchmarkConfig] = None,
                 amendment02: Optional[Amendment02Config] = None, integrity: Optional[Dict] = None,
                 p1b_run_dir: Optional[Path] = None, log: Callable[[str], None] = print) -> Dict:
    benchmark, _, provenance = _prepare(config, config_path, confirm, benchmark, amendment02, integrity, p1b_run_dir)
    matches = [c for cells in enumerate_matrix(config, benchmark.group).values() for c in cells if c.cell_id == cell_id]
    if not matches:
        raise ValueError(f"{cell_id!r} is not a planned cell")
    result, _ = _execute_cell(config, config_path, device, run_dir, benchmark, matches[0], provenance, log)
    return result


# ============================================================================
# Analysis
# ============================================================================


def analyze_experiment(config: Phase2ExperimentConfig, config_path: Path, device: torch.device, run_dir: Path,
                       benchmark: Optional[BenchmarkConfig] = None, amendment01: Optional[AmendmentConfig] = None,
                       amendment02: Optional[Amendment02Config] = None, log: Callable[[str], None] = print) -> Dict:
    config.require_frozen()
    benchmark = benchmark or load_benchmark(config.benchmark)
    amendment01 = amendment01 or amendment_01_of(config)
    amendment02 = amendment02 or amendment_02_of(config)
    results, missing = [], []
    for cells in enumerate_matrix(config, benchmark.group).values():
        for cell in cells:
            cell_dir = cell_directory(run_dir, config.model_base, cell)
            meta_path = cell_dir / METADATA_FILENAME
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            # verdict FAIL means the run finished but diverged; it is not a usable result.
            complete = meta.get("status") == "completed" and meta.get("verdict") != "FAIL" \
                and (cell_dir / "result.json").is_file()
            if complete:
                results.append(json.loads((cell_dir / "result.json").read_text(encoding="utf-8")))
            else:
                missing.append(cell.cell_id)
    if missing:
        raise RuntimeError(f"{len(missing)} planned cells are not complete; the pre-registered analysis runs only on the "
                           f"complete matrix (first missing: {missing[:3]}).")

    analysis = analyze(results, amendment01, amendment02.decision_rule)
    curves = load_curves(Path(run_dir) / "cells" / width_label(config.model_base))
    with RunRecorder(Path(run_dir) / "analysis", f"{config.experiment.name}_analysis", config, config_path, device,
                     ANALYSIS_ARTIFACTS) as run:
        write_json(run.path("analysis.json"), analysis)
        write_report(analysis, run.run_dir, curves)
        outcome = analysis["checkpoints"]["best"]["decision_rule"]["outcome"]
        log(f"Decision rule outcome (best-validation checkpoint): {outcome}")
        run.finish("COMPLETED", {"results": len(results), "decision_rule_outcome_best": outcome,
                                 "decision_rule_outcome_final": analysis["checkpoints"]["final"]["decision_rule"]["outcome"]})
    return analysis


def describe_matrix(config: Phase2ExperimentConfig, group: str) -> str:
    matrix = enumerate_matrix(config, group)
    lines = [f"Phase 2 matrix — model_base={config.model_base or 'PENDING_P1B'}, "
             f"max_steps={config.max_steps or 'PENDING_P1B'}"]
    for stage in config.stages:
        lines.append(f"  {stage.name:<14} {stage.family:<14} tasks={list(stage.tasks)} T={list(stage.t_values)} "
                     f"depths={list(stage.depths)} seeds={list(stage.seeds)} → {len(matrix[stage.name])} runs")
    lines.append(f"  total: {sum(len(v) for v in matrix.values())} runs")
    return "\n".join(lines)
