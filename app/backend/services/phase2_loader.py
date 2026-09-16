"""Read-only Phase 2 status for the dashboard.

Phase 2 has a pre-registered design but, until P1b completes and a width and
budget are approved, no experimental results. This loader therefore reports two
different kinds of thing and never blurs them:

* **planned configuration** — read from ``configs/phase2/*.yaml``. These files
  exist today, so the benchmark constants, T roles, protocol, amendments and
  the 138-run matrix are all real, current, checked-in values;
* **observed results** — read from ``runs/phase2/``. These mostly do not exist
  yet, and every accessor returns an explicit ``missing``/``pending`` state
  rather than a placeholder.

Nothing here invents a number. Every value is either read from a file or
derived from values read from files, and each block carries the path it came
from. A value that was not recorded is ``null``.

Like ``results_loader``, this module does not import torch or numpy. It does
import ``phase2.p1b_decision``, which is pure standard library, so the D5
decision shown in the dashboard is produced by the same pre-registered
evaluator the research code uses rather than a second implementation.
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.results_loader import REPO_ROOT, ArtifactError, read_json, read_text, read_yaml

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.p1b_decision import D5DataError, evaluate_d5, render_d5_decision  # noqa: E402

PHASE2_CONFIG_DIR = REPO_ROOT / "configs" / "phase2"

BENCHMARK_CONFIG = "benchmark.yaml"
AMENDMENT_01_CONFIG = "amendment_01.yaml"
AMENDMENT_02_CONFIG = "amendment_02_draft.yaml"
EXPERIMENT_CONFIG = "phase2_experiment.yaml"
P1B_CONFIG = "capacity_diagnostic_p1b.yaml"

# Run directories, relative to the runs root. These mirror the output_dir values
# in the configs above; they are read from the configs where one exists.
GATES_RAW_DIR = "phase2/gates"
GATES_AMENDED_DIR = "phase2/gates_amendment01"
P1B_DIR = "phase2/p1b_capacity_diagnostic"
EXPERIMENT_DIR = "phase2/experiment"

PENDING_SENTINEL = "PENDING_P1B"

# Editorial numbering for the plan table. The authoritative identity of a stage
# is its config name; this only supplies the label the protocol document uses.
# A stage that is not listed falls back to sequential numbering, so adding a
# stage to the config cannot silently drop it from the view.
STAGE_LABELS = {
    ("t4", "main"): "Stage #1",
    ("t8", "main"): "Stage #2",
    ("anchors", "main"): "Stage #2b",
    ("controls", "c1_span"): "Stage #3",
    ("controls", "c2_word"): "Stage #4",
    ("unrolled", "main"): "Stage #5",
    ("param_matched", "main"): "Stage #6",
    ("width_scaled", "main"): "Stage #7",
}

TASK_LABELS = {"main": "A5 main", "c1_span": "C1 span control", "c2_word": "C2 word control"}
FAMILY_LABELS = {
    "tesseract": "Tesseract (shared block)",
    "unrolled": "Unrolled (non-shared)",
    "param_matched": "Parameter-matched",
    "width_scaled": "Width-scaled K=1",
}

MISSING_GATES_MESSAGE = "Awaiting gate artifacts — run scripts/phase2_gates.py, then scripts/phase2_gates_amended.py."
MISSING_P1B_MESSAGE = "P1b results not available yet."
MISSING_RESULTS_MESSAGE = (
    "No Phase 2 experiment results yet. Training is blocked until P1b completes and the model width, "
    "the step budget and Amendment 02 are approved."
)


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_head() -> Dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except (OSError, ValueError):
        return {"commit": None, "dirty": None}
    if head.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}


class Phase2Loader:
    """Planned configuration and observed artifacts for Phase 2."""

    def __init__(self, runs_root: Path, config_dir: Path = PHASE2_CONFIG_DIR) -> None:
        self.runs_root = Path(runs_root)
        self.config_dir = Path(config_dir)

    # -- configs ---------------------------------------------------------------

    def _config(self, name: str) -> Optional[Any]:
        return read_yaml(self.config_dir / name)

    def _config_source(self, name: str) -> str:
        return _rel(self.config_dir / name)

    def run_dir(self, relative: str) -> Path:
        return self.runs_root / relative

    # -- benchmark, roles, protocol -------------------------------------------

    def get_benchmark(self) -> Dict[str, Any]:
        config = self._config(BENCHMARK_CONFIG)
        if not isinstance(config, dict):
            return {"status": "missing", "message": f"{self._config_source(BENCHMARK_CONFIG)} not found."}
        return {
            "status": "available",
            "source": self._config_source(BENCHMARK_CONFIG),
            "group": config.get("group"),
            "group_description": "A5 non-solvable group cellular automaton",
            "transition": "F(s)[i] = s[i] · s[i+1]",
            "n": config.get("n"),
            "t_values": config.get("t_values"),
            "k_values": config.get("k_values"),
            "tasks": config.get("tasks"),
            "ablation_groups": config.get("ablation_groups"),
            "data_seed": config.get("data_seed"),
            "val_size": config.get("val_size"),
            "test_size": config.get("test_size"),
            "length_n": config.get("length_n"),
            "length_test_size": config.get("length_test_size"),
            "interpolation_t_values": config.get("interpolation_t_values"),
            "model_seeds": config.get("model_seeds"),
            "tau": config.get("tau"),
        }

    def get_t_roles(self) -> Dict[str, Any]:
        config = self._config(AMENDMENT_01_CONFIG)
        if not isinstance(config, dict):
            return {"status": "missing", "message": f"{self._config_source(AMENDMENT_01_CONFIG)} not found."}
        return {
            "status": "available",
            "source": self._config_source(AMENDMENT_01_CONFIG),
            "anchor": config.get("anchor_t_values"),
            "diagnostic": config.get("diagnostic_t_values"),
            "primary_depth": config.get("primary_depth_t_values"),
        }

    def get_protocol(self) -> Dict[str, Any]:
        experiment = self._config(EXPERIMENT_CONFIG)
        benchmark = self._config(BENCHMARK_CONFIG)
        if not isinstance(experiment, dict):
            return {"status": "missing", "message": f"{self._config_source(EXPERIMENT_CONFIG)} not found."}
        optimizer = experiment.get("optimizer") or {}
        return {
            "status": "available",
            "source": self._config_source(EXPERIMENT_CONFIG),
            "model_seeds": (benchmark or {}).get("model_seeds"),
            "optimizer": optimizer.get("name"),
            "learning_rate": optimizer.get("learning_rate"),
            "weight_decay": optimizer.get("weight_decay"),
            "batch_size": experiment.get("batch_size"),
            "eval_every": experiment.get("eval_every"),
            "eval_batch_size": experiment.get("eval_batch_size"),
            "vocab_size": experiment.get("vocab_size"),
            "early_stopping": False,
            "primary_checkpoint": "best-validation checkpoint",
            "sensitivity_checkpoint": "final checkpoint",
            "test_policy": "test evaluated only after training completes",
        }

    # -- decision state ---------------------------------------------------------

    def get_decisions(self) -> Dict[str, Any]:
        a1 = self._config(AMENDMENT_01_CONFIG)
        a2 = self._config(AMENDMENT_02_CONFIG)
        experiment = self._config(EXPERIMENT_CONFIG)

        amendment_01: Dict[str, Any] = {"status": "missing", "message": "amendment_01.yaml not found."}
        if isinstance(a1, dict):
            recorded = a1.get("recorded_before_training")
            amendment_01 = {
                "status": "available",
                "source": self._config_source(AMENDMENT_01_CONFIG),
                "id": a1.get("id"),
                "date": a1.get("date"),
                "recorded_before_training": recorded,
                "state": "ACTIVE — RECORDED" if recorded else "RECORDED AFTER TRAINING",
                "tone": "ok" if recorded else "fail",
                "summary": (a1.get("summary") or "").strip(),
            }

        amendment_02: Dict[str, Any] = {"status": "missing", "message": "amendment_02_draft.yaml not found."}
        if isinstance(a2, dict):
            frozen = bool(a2.get("frozen"))
            amendment_02 = {
                "status": "available",
                "source": self._config_source(AMENDMENT_02_CONFIG),
                "id": a2.get("id"),
                "date": a2.get("date"),
                "draft_status": a2.get("status"),
                "frozen": frozen,
                "state": "FROZEN" if frozen else "DRAFT — NOT FROZEN",
                "tone": "ok" if frozen else "warn",
                "decisions": [
                    {"id": d.get("id"), "title": d.get("title")}
                    for d in (a2.get("decisions") or []) if isinstance(d, dict)
                ],
                "decision_rule": a2.get("decision_rule"),
                "pending_p1b": a2.get("pending_p1b"),
            }

        def pending_field(key: str) -> Dict[str, Any]:
            if not isinstance(experiment, dict) or key not in experiment:
                return {"status": "missing", "message": f"{key} not found in {EXPERIMENT_CONFIG}."}
            raw = experiment[key]
            pending = raw == PENDING_SENTINEL
            return {
                "status": "available",
                "source": self._config_source(EXPERIMENT_CONFIG),
                "pending": pending,
                "value": None if pending else raw,
                "state": "PENDING P1B" if pending else str(raw),
                "tone": "warn" if pending else "ok",
            }

        return {
            "amendment_01": amendment_01,
            "amendment_02": amendment_02,
            "model_width": pending_field("model_base"),
            "training_budget": pending_field("max_steps"),
            "p1b": self.get_p1b_status(),
        }

    # -- P1b --------------------------------------------------------------------

    def get_p1b_status(self) -> Dict[str, Any]:
        run_dir = self.run_dir(P1B_DIR)
        metadata = read_json(run_dir / "run_metadata.json")
        report_path = run_dir / "diagnostic_report.json"
        if metadata is None and not report_path.is_file():
            return {"status": "pending", "state": "PENDING", "tone": "warn", "run_dir": P1B_DIR,
                    "message": MISSING_P1B_MESSAGE, "run": None}
        run_status = (metadata or {}).get("status")
        state = {"running": "RUNNING", "completed": "COMPLETE", "failed": "FAILED"}.get(run_status, "UNKNOWN")
        return {
            "status": "available",
            "state": state,
            "tone": {"COMPLETE": "ok", "RUNNING": "info", "FAILED": "fail"}.get(state, "warn"),
            "run_dir": P1B_DIR,
            "run": metadata,
            "report_present": report_path.is_file(),
        }

    def get_p1b(self) -> Dict[str, Any]:
        """The P1b diagnostic and, when it is present, the pre-registered D5 decision."""
        status = self.get_p1b_status()
        run_dir = self.run_dir(P1B_DIR)
        report_path = run_dir / "diagnostic_report.json"
        if not report_path.is_file():
            return {**status, "status": "missing", "message": MISSING_P1B_MESSAGE, "report": None, "d5": None}

        report = read_json(report_path)  # raises ArtifactError on malformed JSON
        if not isinstance(report, dict):
            raise ArtifactError("diagnostic_report.json: expected a JSON object")

        settings = report.get("settings") or {}
        cells: List[Dict[str, Any]] = []
        for width, per_k in sorted((report.get("runs") or {}).items()):
            if not isinstance(per_k, dict):
                raise ArtifactError(f"diagnostic_report.json: runs.{width} is not an object")
            for k, run in sorted(per_k.items(), key=lambda kv: int(kv[0])):
                headline = (run or {}).get("headline") or {}
                cells.append({
                    "width": width,
                    "model_base": (run or {}).get("model_base"),
                    "k": int(k),
                    "parameter_count": headline.get("parameter_count"),
                    "block_calls_per_forward": headline.get("block_calls_per_forward"),
                    "steps_run": headline.get("steps_run"),
                    "best_val_chance_normalised": headline.get("best_val_chance_normalised"),
                    "final_val_chance_normalised": headline.get("final_val_chance_normalised"),
                    "final_val_loss": headline.get("final_val_loss"),
                    "final_val_exact_match": headline.get("final_val_exact_match"),
                    "at_floor_best": (run or {}).get("at_floor_best"),
                    "at_floor_final": (run or {}).get("at_floor_final"),
                })

        d5: Dict[str, Any]
        try:
            decision = evaluate_d5(report)
            d5 = {"status": "available", "decision": decision, "text": render_d5_decision(decision)}
        except D5DataError as exc:
            d5 = {"status": "error", "message": str(exc), "decision": None, "text": None}

        return {
            **status,
            "status": "available",
            "report_path": _rel(report_path),
            "report_sha256": _sha256(report_path),
            "settings": settings,
            "single_seed_note": report.get("single_seed_note"),
            "scope": report.get("scope"),
            "cells": cells,
            "d5": d5,
        }

    # -- gates ------------------------------------------------------------------

    def get_gates(self) -> Dict[str, Any]:
        raw_dir = self.run_dir(GATES_RAW_DIR)
        report = read_json(raw_dir / "gates_report.json")
        if report is None:
            return {"status": "missing", "message": MISSING_GATES_MESSAGE,
                    "raw_run_dir": GATES_RAW_DIR, "amended_run_dir": GATES_AMENDED_DIR}
        if not isinstance(report, dict) or "primary" not in report:
            raise ArtifactError("gates_report.json: missing 'primary'")

        primary = report["primary"]
        per_t = primary.get("per_T") or {}
        roles = self.get_t_roles()
        diagnostic_t = set(roles.get("diagnostic") or [])

        rows = []
        for key in sorted(per_t, key=lambda k: int(k)):
            t = int(key)
            entry = per_t[key] or {}
            g2 = entry.get("G2") or {}
            applicable = bool(g2.get("applicable"))
            g2_passed = entry.get("G2_passed")
            rows.append({
                "t": t,
                "role": "diagnostic" if t in diagnostic_t else None,
                "g1_passed": entry.get("G1_passed"),
                "g1_tv_distance": (entry.get("G1_uniformity") or {}).get("tv_distance"),
                "g1_min_sensitivity": (entry.get("G1_sensitivity") or {}).get("min_inside_fraction"),
                "g2_applicable": applicable,
                # The raw verdict is reported exactly as recorded. A failure at a
                # diagnostic T is annotated by Amendment 01, never rewritten.
                "g2_passed": g2_passed,
                "g2_state": "N/A" if not applicable else ("PASS" if g2_passed else "FAIL"),
                "g2_max_token_agreement": g2.get("max_token_agreement"),
                "g2_candidate": g2.get("max_token_agreement_candidate"),
                "amendment_01_diagnostic": bool(t in diagnostic_t and g2_passed is False),
            })

        amended = read_json(self.run_dir(GATES_AMENDED_DIR) / "amendment_evaluation.json")
        amended_meta = read_json(self.run_dir(GATES_AMENDED_DIR) / "run_metadata.json")
        return {
            "status": "available",
            "raw_run_dir": GATES_RAW_DIR,
            "amended_run_dir": GATES_AMENDED_DIR,
            "group": primary.get("group"),
            "n": primary.get("n"),
            "raw_verdict": "PASS" if report.get("passed") else "FAIL",
            "raw_primary_passed": primary.get("passed"),
            "cycles_passed": primary.get("G1_cycles_passed"),
            "self_test_passed": report.get("self_test_passed"),
            "per_t": rows,
            "run": read_json(raw_dir / "run_metadata.json"),
            "amended": None if amended is None else {
                "amended_verdict": amended.get("amended_verdict"),
                "raw_gate_verdict": amended.get("raw_gate_verdict"),
                "checks": amended.get("checks"),
                "run": amended_meta,
            },
        }

    # -- plan -------------------------------------------------------------------

    def get_plan(self) -> Dict[str, Any]:
        experiment = self._config(EXPERIMENT_CONFIG)
        if not isinstance(experiment, dict):
            return {"status": "missing", "message": f"{self._config_source(EXPERIMENT_CONFIG)} not found."}

        stages = experiment.get("stages") or []
        if not isinstance(stages, list) or not stages:
            raise ArtifactError(f"{EXPERIMENT_CONFIG}: 'stages' must be a non-empty list")

        blocked_reasons = self._blocked_reasons(experiment)
        completed = self._completed_cells()
        rows, fallback = [], 0
        for stage in stages:
            if not isinstance(stage, dict):
                raise ArtifactError(f"{EXPERIMENT_CONFIG}: each stage must be a mapping")
            name = stage.get("name")
            seeds = stage.get("seeds") or []
            t_values = stage.get("t_values") or []
            depths = stage.get("depths") or []
            for task in stage.get("tasks") or []:
                fallback += 1
                planned = len(seeds) * len(t_values) * len(depths)
                done = completed.get((name, task), 0)
                rows.append({
                    "stage": name,
                    "label": STAGE_LABELS.get((name, task), f"Stage #{fallback}"),
                    "purpose": stage.get("purpose"),
                    "family": stage.get("family"),
                    "family_label": FAMILY_LABELS.get(stage.get("family"), stage.get("family")),
                    "task": task,
                    "task_label": TASK_LABELS.get(task, task),
                    "t_values": t_values,
                    "depths": depths,
                    "depth_symbol": "L" if stage.get("family") in ("unrolled", "param_matched") else "K",
                    "seeds": seeds,
                    "planned_runs": planned,
                    "completed_runs": done,
                    "state": self._stage_state(planned, done, blocked_reasons),
                })

        return {
            "status": "available",
            "source": self._config_source(EXPERIMENT_CONFIG),
            "stages": rows,
            "total_planned_runs": sum(r["planned_runs"] for r in rows),
            "total_completed_runs": sum(r["completed_runs"] for r in rows),
            "blocked_reasons": blocked_reasons,
            "experiment_run_dir": EXPERIMENT_DIR,
        }

    def _blocked_reasons(self, experiment: Dict[str, Any]) -> List[str]:
        reasons = []
        if experiment.get("model_base") == PENDING_SENTINEL:
            reasons.append("model width is PENDING_P1B")
        if experiment.get("max_steps") == PENDING_SENTINEL:
            reasons.append("step budget is PENDING_P1B")
        a2 = self._config(AMENDMENT_02_CONFIG)
        if isinstance(a2, dict) and not a2.get("frozen"):
            reasons.append("Amendment 02 is a draft and is not frozen")
        p1b = self.get_p1b_status()
        if p1b.get("state") != "COMPLETE":
            reasons.append(f"P1b is {p1b.get('state', 'PENDING')}")
        return reasons

    @staticmethod
    def _stage_state(planned: int, completed: int, blocked_reasons: List[str]) -> str:
        if completed and completed >= planned:
            return "COMPLETE"
        if completed:
            return "RUNNING"
        if blocked_reasons:
            return "BLOCKED"
        return "READY"

    def _completed_cells(self) -> Dict[tuple, int]:
        """Completed experiment cells on disk, counted per (stage, task).

        Cell directories are named ``<family>_d<depth>_<group>_<task>_T<t>_s<seed>``
        by ``phase2.config.CellSpec``. Nothing is counted unless the cell wrote a
        result and its metadata says it completed without a FAIL verdict.
        """
        cells_root = self.run_dir(EXPERIMENT_DIR) / "cells"
        if not cells_root.is_dir():
            return {}
        experiment = self._config(EXPERIMENT_CONFIG)
        stages = (experiment or {}).get("stages") or []
        counts: Dict[tuple, int] = {}
        for width_dir in cells_root.iterdir():
            if not width_dir.is_dir():
                continue
            for cell_dir in width_dir.iterdir():
                if not cell_dir.is_dir() or not (cell_dir / "result.json").is_file():
                    continue
                metadata = read_json(cell_dir / "run_metadata.json") or {}
                if metadata.get("status") != "completed" or metadata.get("verdict") == "FAIL":
                    continue
                key = self._match_stage(cell_dir.name, stages)
                if key is not None:
                    counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _match_stage(cell_id: str, stages: List[Any]) -> Optional[tuple]:
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            family = stage.get("family")
            if not cell_id.startswith(f"{family}_d"):
                continue
            for task in stage.get("tasks") or []:
                for t in stage.get("t_values") or []:
                    for depth in stage.get("depths") or []:
                        for seed in stage.get("seeds") or []:
                            if cell_id.endswith(f"_{task}_T{t}_s{seed}") and f"_d{depth}_" in cell_id:
                                return (stage.get("name"), task)
        return None

    # -- results (not produced yet) --------------------------------------------

    def get_results(self) -> Dict[str, Any]:
        """Phase 2 experiment results.

        The analysis writes ``analysis.json`` under the experiment run directory
        once every planned cell is complete. Until then this reports the pending
        state and the fields the view will show, so the shape of the future
        payload is visible without any value being invented.
        """
        analysis_path = self.run_dir(EXPERIMENT_DIR) / "analysis" / "analysis.json"
        plan = self.get_plan()
        planned_metrics = [
            "chance_normalised_token_accuracy", "exact_match", "per_position_accuracy",
            "delta_s_per_seed", "did_s_per_seed", "ci95_t_interval", "bootstrap_ci95",
            "best_checkpoint", "final_checkpoint", "parameter_count", "forward_flops_per_sequence",
            "seconds_per_train_step", "block_calls_per_forward", "floor_ceiling", "per_seed_results",
        ]
        if not analysis_path.is_file():
            return {
                "status": "missing",
                "message": MISSING_RESULTS_MESSAGE,
                "analysis_path": _rel(analysis_path),
                "total_planned_runs": plan.get("total_planned_runs"),
                "total_completed_runs": plan.get("total_completed_runs"),
                "blocked_reasons": plan.get("blocked_reasons"),
                "planned_metrics": planned_metrics,
                "analysis": None,
            }
        analysis = read_json(analysis_path)
        if not isinstance(analysis, dict):
            raise ArtifactError("analysis.json: expected a JSON object")
        return {
            "status": "available",
            "analysis_path": _rel(analysis_path),
            "analysis_sha256": _sha256(analysis_path),
            "total_planned_runs": plan.get("total_planned_runs"),
            "total_completed_runs": plan.get("total_completed_runs"),
            "planned_metrics": planned_metrics,
            "analysis": analysis,
            "run": read_json(self.run_dir(EXPERIMENT_DIR) / "analysis" / "run_metadata.json"),
        }

    # -- aggregate ---------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        def guarded(fn):
            try:
                return fn()
            except ArtifactError as exc:
                return {"status": "error", "message": str(exc)}

        configs = {
            name: {"path": self._config_source(name), "sha256": _sha256(self.config_dir / name),
                   "present": (self.config_dir / name).is_file()}
            for name in (BENCHMARK_CONFIG, AMENDMENT_01_CONFIG, AMENDMENT_02_CONFIG, EXPERIMENT_CONFIG, P1B_CONFIG)
        }
        return {
            "project": "Tesseract — Phase 2",
            "phase": "PHASE 2 — CURRENT RESEARCH / IN PROGRESS",
            "configured": any(c["present"] for c in configs.values()),
            "benchmark": guarded(self.get_benchmark),
            "t_roles": guarded(self.get_t_roles),
            "protocol": guarded(self.get_protocol),
            "decisions": guarded(self.get_decisions),
            "gates": guarded(self.get_gates),
            "plan": guarded(self.get_plan),
            "p1b": guarded(self.get_p1b),
            "results": guarded(self.get_results),
            "provenance": {"git": _git_head(), "runs_root": str(self.runs_root), "configs": configs},
        }
