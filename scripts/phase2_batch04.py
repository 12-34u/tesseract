#!/usr/bin/env python3
"""Batch 04 — T = 4 with the residual fix and the full curriculum
(PHASE2_BATCH_04_PREREG.md).

Batch 03's configuration with the depth moved to 4, the ladder extended to
T = 1 -> 2 -> 3 -> 4, K = 2 dropped and the budget at 40,000 steps. The residual
fix stays on at every K.

T = 4 passes G2, so no shortcut attribution is computed at the target depth; the
ladder still passes through T = 2 and T = 3, which fail G2, and agreement is
recorded there in the stage-boundary diagnostics.

Exact match is a reported co-primary metric. The classification rule is
unchanged, so exact match constrains the interpretation rather than the verdict:
at T = 3 a K = 1 cell cleared the token-accuracy bar at 64 % while producing
0.15 % exact match, and that must not read as learning the function.

``--smoke`` exercises the whole path at trivial step counts to catch a crash
before the real run spends five GPU-hours reaching one. It writes to a separate
directory, records the flag in its report, and cannot be mistaken for a result:
PHASE2_BATCH_04_PREREG.md §2 fixes the configuration, and a smoke report carries
``"smoke": true`` and a step count that is not 40,000.

Usage:
    python scripts/phase2_batch04.py [--smoke] [--resume]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from phase2.batch01 import CurriculumStream, apply_per_step_norm, hidden_state_norms  # noqa: E402
from phase2.batch02 import (  # noqa: E402
    SHORTCUT_CANDIDATE,
    shortcut_agreement_percent,
    stage_boundary_steps,
)
from phase2.batch04 import (  # noqa: E402
    BATCH04_CONFIG as PROBE_CONFIG,
    DIAGNOSTIC_T_VALUES,
    EXACT_MATCH_FLOOR_PERCENT,
    GROWTH_LIMIT,
    bound_held,
    exact_match_reading,
    inherited_matches_batch03,
    outcome as batch04_outcome,
)
from phase2.compute import compute_provenance, configure_determinism  # noqa: E402
from phase2.config import CellSpec, ModelVariant, TrainingConfig, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.probes import classify  # noqa: E402
from phase2.reclassify import amend_report, distinct_predicted_tokens, majority_class_percent  # noqa: E402
from phase2.runner import splits_dir  # noqa: E402
from phase2.salvage import BOUNDARY_MISSING_FIELDS, salvage  # noqa: E402
from phase2.groups import get_group  # noqa: E402
from phase2.metrics import compute_metrics  # noqa: E402
from phase2.training import predict_with_loss, train_fixed_steps  # noqa: E402
from utils.config import OptimizerConfig  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402

PREREG = Path("configs/phase2/probe_preregistration.yaml")
ROOT = "phase2/batch04"
NAME = "t4_k_sweep"
TARGET_T = PROBE_CONFIG["target_t"]
WIDTH = ModelVariant(name="medium", base="phase2/prototype_medium")
K_VALUES = tuple(PROBE_CONFIG["k_values"])
SEEDS = (0, 1, 2)
BATCH, EVAL_EVERY = 128, 250


#: Trivial substitute for PROBE_CONFIG, for --smoke only. Same code path, same
#: number of stage boundaries, ~50 steps instead of 40,000.
SMOKE_CONFIG = {**PROBE_CONFIG, "curriculum_stages": [1, 2, 3, 4], "steps_per_stage": 10,
                "target_steps": 10, "steps": 50, "k_values": [1, 8], "seeds": 1}


def resume_dir(name: str) -> Optional[Path]:
    """The existing run directory a resume should continue into, or None.

    Resume is the one case where writing into a populated directory is correct,
    because the point is to keep the cells already in it.
    """
    base = resolve_run_dir(f"{ROOT}/{name}")
    candidates = sorted((p for p in base.parent.glob(base.name + "*")
                         if p.is_dir() and "_smoke" not in p.name and (p / "cells").is_dir()),
                        key=lambda p: p.name)
    return candidates[-1] if candidates else None


def out_dir(name: str) -> Path:
    """Never overwrite: a rerun goes to a sibling __attemptN directory."""
    base = resolve_run_dir(f"{ROOT}/{name}")
    if not base.exists() or not any(base.iterdir()):
        return base
    attempt = 2
    while (base.parent / f"{base.name}__attempt{attempt}").exists():
        attempt += 1
    return base.parent / f"{base.name}__attempt{attempt}"


def setup():
    if not torch.cuda.is_available():
        print("CUDA unavailable; Batch 04 runs on GPU (Amendment 02a P1).", file=sys.stderr)
        raise SystemExit(2)
    device = torch.device("cuda")
    determinism = configure_determinism(device)
    provenance = compute_provenance(device, determinism)
    provenance["batch"] = "batch04"
    provenance["preregistration"] = "PHASE2_BATCH_04_PREREG.md"
    benchmark = load_benchmark()
    model_cfg = resolve_model_variant(WIDTH, 60)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    splits = {t: store.get(benchmark.group, "main", "val", t, benchmark.n, benchmark.val_size)
              for t in DIAGNOSTIC_T_VALUES}
    return device, provenance, benchmark, model_cfg, store, splits


def training_config(steps: int, eval_batch: int, eval_every: int = EVAL_EVERY) -> TrainingConfig:
    return TrainingConfig(optimizer=OptimizerConfig("adamw", 1e-3, 0.01), batch_size=BATCH,
                          max_steps=steps, eval_every=eval_every, eval_batch_size=eval_batch)


def per_example_se(predictions, targets) -> float:
    per = (predictions == targets).mean(axis=1) * 100.0
    return float(per.std(ddof=1) / np.sqrt(len(per)))


def depth_diagnostics(model, splits, group, device, eval_batch) -> list:
    """Validation numbers at every diagnostic depth, for the model as it stands.

    Called at each stage boundary and again on the best checkpoint. Records the
    shortcut agreement wherever G2 found a candidate, and the distinct-token
    count so a collapsed stage is visible as such rather than as an accuracy.
    """
    rows = []
    for t in DIAGNOSTIC_T_VALUES:
        split = splits[t]
        predictions, loss = predict_with_loss(model, split.inputs, split.targets, device, eval_batch)
        metrics = compute_metrics(predictions, split.targets, 60)
        row = {
            "t": t,
            "token_accuracy": metrics.token_accuracy,
            "exact_match": metrics.exact_match,
            "loss": float(loss),
            "distinct_predicted_tokens": distinct_predicted_tokens(predictions),
            "majority_class_percent": majority_class_percent(split.targets),
        }
        if t in SHORTCUT_CANDIDATE:
            row["shortcut_candidate"] = SHORTCUT_CANDIDATE[t]
            row["shortcut_agreement_percent"] = shortcut_agreement_percent(predictions, group, split.inputs, t)
        rows.append(row)
    return rows


def bounded_growth_gate(device, benchmark, model_cfg, val, run_dir, k_values) -> bool:
    """Arm 2c-i, reused unchanged: per-step RMS must be bounded at every K.

    Blocks the whole batch on failure. A null result at an unbounded K would be
    uninterpretable in exactly the way Batch 02's K=8 was, so running the cells
    anyway would reproduce the defect this batch exists to remove.
    """
    rows = []
    for k in k_values:
        set_seed(SEEDS[0])
        model = build_model("tesseract", k, model_cfg, benchmark.n).model.to(device)
        patched = apply_per_step_norm(model)
        norms = hidden_state_norms(model, val.inputs[:256], device)
        growth = norms[-1]["z_L_rms"] / norms[0]["z_L_rms"]
        rows.append({"k": k, "patched_module": patched, "per_step": norms,
                     "z_L_rms_first": norms[0]["z_L_rms"], "z_L_rms_last": norms[-1]["z_L_rms"],
                     "growth_ratio": growth, "bounded": bool(growth < GROWTH_LIMIT)})
        print(f"  K={k}: rms {norms[0]['z_L_rms']:.4f} -> {norms[-1]['z_L_rms']:.4f}  growth {growth:.4f}x"
              f"  {'OK' if growth < GROWTH_LIMIT else 'FAIL'}")
    passed = all(r["bounded"] for r in rows)
    write_json(run_dir / "bounded_growth_gate.json",
               {"gate": "batch04-i", "growth_limit": GROWTH_LIMIT, "rows": rows, "gate_passed": passed})
    print(f"  bounded-growth gate: {'PASS' if passed else 'FAIL'}")
    return passed


def recover_cell(cell, cell_dir, checkpoint, k, seed, model_cfg, benchmark, splits, group, device,
                 salvaged_cell):
    """Rebuild a finished cell's record from its checkpoint, without retraining.

    Everything measured on the final model is recomputed here, so those fields
    are artifact-backed exactly as a freshly trained cell's are: accuracy, loss,
    distinct tokens, shortcut agreement, the per-depth diagnostics and the
    post-training bound.

    The curve and the stage boundaries cannot be recomputed — they were measured
    during training — so they come from the console log and are tagged
    accordingly. `stage_boundary_missing_fields` names what the log did not
    print, so a salvaged cell never reads as though it were whole.
    """
    val = splits[TARGET_T]
    set_seed(seed)
    built = build_model("tesseract", k, model_cfg, benchmark.n)
    model = built.model.to(device)
    patched = apply_per_step_norm(model, elementwise_affine=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    predictions, loss = predict_with_loss(model, val.inputs, val.targets, device, benchmark.val_size)
    accuracy = float((predictions == val.targets).mean() * 100.0)
    agreement = (shortcut_agreement_percent(predictions, group, val.inputs, TARGET_T)
                 if TARGET_T in SHORTCUT_CANDIDATE else None)
    distinct = distinct_predicted_tokens(predictions)
    target_metrics = compute_metrics(predictions, val.targets, 60)
    post = hidden_state_norms(model, val.inputs[:256], device)
    growth = post[-1]["z_L_rms"] / post[0]["z_L_rms"]
    held = bound_held(growth)

    curve = (salvaged_cell or {}).get("curve") or []
    boundaries = (salvaged_cell or {}).get("stage_boundary_validation") or []
    record = {
        "seed": seed, "k": k,
        "best_val": {"token_accuracy": accuracy, "loss": float(loss)},
        # The final-checkpoint validation loss IS recoverable: the last curve
        # point is the evaluation at the final step, which is what the training
        # loop records as final_val_loss. Without it the rule's above-chance-by-
        # loss arm would be silently disabled for recovered cells.
        "final_val": {"token_accuracy": accuracy, "loss": (curve[-1]["val_loss"] if curve else None)},
        "num_evaluations": len(curve),
        "per_example_token_accuracy_se": per_example_se(predictions, val.targets),
        "cell_dir": str(cell_dir),
    }
    entry = {
        "k": k, "seed": seed, "cell_id": cell.cell_id,
        "steps_run": curve[-1]["step"] if curve else None,
        "stopped_reason": "recovered_from_checkpoint",
        "parameter_count": built.parameter_count,
        "patched_module": patched,
        "per_step_norm_elementwise_affine": False,
        "post_training_per_step_rms": post,
        "post_training_growth_ratio": growth,
        "post_training_bound_held": held,
        "stage_boundary_validation": boundaries,
        "best_checkpoint_validation_by_t": depth_diagnostics(model, splits, group, device, benchmark.val_size),
        "best_val_distinct_predicted_tokens": distinct,
        "best_val_shortcut_agreement_percent": agreement,
        "best_val_exact_match": target_metrics.exact_match,
        "curve": curve,
        # Provenance, per field group, so a reader can see which cells are
        # artifact-backed throughout and which are not.
        "source": "recovered_from_checkpoint",
        "recomputed_from_checkpoint": [
            "best_val", "best_val_distinct_predicted_tokens", "best_val_shortcut_agreement_percent",
            "best_checkpoint_validation_by_t", "post_training_per_step_rms",
            "post_training_growth_ratio", "post_training_bound_held",
        ],
        "recovered_from_console_log": ["curve", "stage_boundary_validation", "steps_run",
                                       "final_val.loss"],
        "stage_boundary_missing_fields": list(BOUNDARY_MISSING_FIELDS),
        "console_summary": (salvaged_cell or {}).get("console_summary"),
    }
    console = entry["console_summary"] or {}
    drift = (None if not console else abs(console.get("best_val_token_accuracy", 0.0) - accuracy))
    entry["console_vs_recomputed_accuracy_delta"] = drift
    print(f"=== {NAME}: K={k} seed={seed} RECOVERED from checkpoint "
          f"(no retrain; curve and boundaries from console log)")
    print(f"    best val token {accuracy:.4f}%  EXACT MATCH {target_metrics.exact_match:.4f}%  "
          f"loss {float(loss):.4f}  distinct {distinct}")
    print(f"    post-training z_L RMS {post[0]['z_L_rms']:.4f} -> {post[-1]['z_L_rms']:.4f} "
          f"(growth {growth:.4f}x) {'bound held' if held else 'BOUND BROKE'}")
    if drift is not None:
        # A mismatch would mean the checkpoint and the log are not from the same
        # run, which is the one failure mode a resume must not paper over.
        print(f"    console log said {console['best_val_token_accuracy']:.4f}%  "
              f"delta {drift:.4f} points {'OK' if drift < 0.01 else '*** MISMATCH ***'}")
    return entry, record


def run(args) -> int:
    smoke = bool(getattr(args, "smoke", False))
    config = SMOKE_CONFIG if smoke else PROBE_CONFIG
    k_values = tuple(config["k_values"])
    seeds = SEEDS[:config["seeds"]]
    eval_every = 5 if smoke else EVAL_EVERY
    device, provenance, benchmark, model_cfg, store, splits = setup()
    val = splits[TARGET_T]
    resume = bool(getattr(args, "resume", False)) and not smoke
    run_dir = (resume_dir(NAME) if resume else None) or out_dir(f"{NAME}_smoke" if smoke else NAME)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Mid-training measurements cannot be recomputed from a checkpoint, so a
    # resumed cell takes its curve and stage boundaries from the console log.
    # Tagged as log-sourced: a console log has no checksum and its boundary rows
    # are a subset of the recorded field.
    salvaged = {}
    if resume:
        log = Path(f"runs/phase2/batch04_{NAME}.console.log")
        if log.is_file():
            salvaged = salvage(log)
            print(f"=== resume: {run_dir}")
            print(f"    salvaged {sum(1 for v in salvaged.values() if v['complete'])} complete cells from {log}")
        else:
            print(f"=== resume: {run_dir} (no console log at {log}; curves will be absent)")
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    group = get_group(benchmark.group)
    boundaries = stage_boundary_steps(config["curriculum_stages"], config["steps_per_stage"],
                                      config["target_steps"])
    assert boundaries[-1] == config["steps"], boundaries
    # A drift in a field Batch 03 is meant to inherit would make the comparison
    # against Batch 02 uninterpretable, so it stops the run rather than being
    # discovered afterwards.
    if not inherited_matches_batch03():
        print("Batch 04 config drifted from Batch 03 on an inherited field; refusing to run.", file=sys.stderr)
        return 2

    print("=== Batch 04 gate: per-step hidden-state RMS must be bounded at every K")
    if not bounded_growth_gate(device, benchmark, model_cfg, val, run_dir, k_values):
        print("Bounded-growth gate failed; no training cell runs (PHASE2_BATCH_04_PREREG.md §3).", file=sys.stderr)
        return 1

    records, detail = [], []

    def write_partial():
        """Write what exists after every cell.

        The batch lost nine finished cells to a reboot because the report is
        only written once every cell completes. A partial written per cell costs
        milliseconds and makes that unrepeatable.
        """
        write_json(run_dir / f"{NAME}_partial.json",
                   {"batch": "03", "partial": True, "cells_done": len(detail),
                    "cells_expected": len(k_values) * len(seeds),
                    "config": dict(config), "provenance": provenance,
                    "records": records, "detail": detail})

    for k in k_values:
        for seed in seeds:
            cell = CellSpec("tesseract", k, benchmark.group, "main", TARGET_T, seed)
            cell_dir = run_dir / "cells" / cell.cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = cell_dir / "best_checkpoint.pt"
            if resume and checkpoint.is_file():
                entry, record = recover_cell(cell, cell_dir, checkpoint, k, seed, model_cfg, benchmark,
                                             splits, group, device, salvaged.get((k, seed)))
                records.append(record)
                detail.append(entry)
                write_partial()
                continue

            set_seed(seed)
            built = build_model("tesseract", k, model_cfg, benchmark.n)
            model = built.model.to(device)
            # The one change against Batch 02, applied at every K.
            patched = apply_per_step_norm(model, elementwise_affine=False)

            stream = CurriculumStream(benchmark.group, "main", benchmark.n, BATCH, benchmark.data_seed, seed,
                                      list(config["curriculum_stages"]), config["steps_per_stage"],
                                      TARGET_T, config["target_steps"], excluded_inputs=[val.inputs])
            schedule = stream.stage_boundaries()

            def boundary_probe(step, _model=model, _schedule=schedule):
                """The fix for Batch 01's recorded deviation, as a report field."""
                stage = next((s for s in _schedule if s["ends_at_step"] == step), None)
                rows = depth_diagnostics(_model, splits, group, device, benchmark.val_size)
                print(f"    [stage boundary] step {step} ends stage T={stage['t'] if stage else '?'}: "
                      + "  ".join(f"T{r['t']} acc {r['token_accuracy']:.4f}% loss {r['loss']:.4f}"
                                  + (f" sc {r['shortcut_agreement_percent']:.2f}%"
                                     if "shortcut_agreement_percent" in r else "")
                                  for r in rows))
                return {"stage_t": stage["t"] if stage else None,
                        "stage_index": _schedule.index(stage) if stage else None,
                        "validation_by_t": rows}

            print(f"=== {NAME}{' SMOKE' if smoke else ''}: target T={TARGET_T} K={k} seed={seed} "
                  f"steps={config['steps']} fix=per_step(non-affine) "
                  f"curriculum={config['curriculum_stages']}->{TARGET_T}")
            outcome = train_fixed_steps(model, stream, val, None,
                                        training_config(config["steps"], benchmark.val_size, eval_every),
                                        device, 60,
                                        step_hooks={b: boundary_probe for b in boundaries},
                                        log=print)

            torch.save(outcome.final_state, cell_dir / "final_checkpoint.pt")
            final_boundary_records = list(outcome.hook_records)
            if outcome.best_state is not None:
                torch.save(outcome.best_state, cell_dir / "best_checkpoint.pt")
                model.load_state_dict(outcome.best_state)

            predictions, loss = predict_with_loss(model, val.inputs, val.targets, device, benchmark.val_size)
            accuracy = float((predictions == val.targets).mean() * 100.0)
            # T=4 passes G2, so there is no shallow candidate to attribute to and
            # no agreement to record at the target depth. The ladder's T=2 and
            # T=3 rungs still get it, in the stage-boundary diagnostics.
            agreement = (shortcut_agreement_percent(predictions, group, val.inputs, TARGET_T)
                         if TARGET_T in SHORTCUT_CANDIDATE else None)
            distinct = distinct_predicted_tokens(predictions)
            target_metrics = compute_metrics(predictions, val.targets, 60)
            records.append({
                "seed": seed, "k": k,
                "best_val": {"token_accuracy": accuracy, "loss": float(loss)},
                "final_val": {"token_accuracy": accuracy, "loss": outcome.final_val_loss},
                "num_evaluations": sum(1 for row in outcome.history if "val_loss" in row),
                "per_example_token_accuracy_se": per_example_se(predictions, val.targets),
                "cell_dir": str(cell_dir),
            })
            # Addendum 01 §A1.3: the bound must hold AFTER training, per cell.
            post = hidden_state_norms(model, val.inputs[:256], device)
            growth = post[-1]["z_L_rms"] / post[0]["z_L_rms"]
            held = bound_held(growth)
            print(f"    post-training z_L RMS {post[0]['z_L_rms']:.4f} -> {post[-1]['z_L_rms']:.4f} "
                  f"(growth {growth:.4f}x) {'bound held' if held else 'BOUND BROKE'}")

            detail.append({
                "k": k, "seed": seed, "cell_id": cell.cell_id, "steps_run": outcome.steps_run,
                "stopped_reason": outcome.stopped_reason, "parameter_count": built.parameter_count,
                "patched_module": patched,
                "per_step_norm_elementwise_affine": False,
                "post_training_per_step_rms": post,
                "post_training_growth_ratio": growth,
                "post_training_bound_held": held,
                "stage_boundaries": schedule,
                # The deviation fix: one entry per boundary, recorded, not printed.
                "stage_boundary_validation": final_boundary_records,
                "best_checkpoint_validation_by_t": depth_diagnostics(model, splits, group, device,
                                                                    benchmark.val_size),
                "best_val_distinct_predicted_tokens": distinct,
                "best_val_shortcut_agreement_percent": agreement,
                # Co-primary metric (prereg §4). Conjunctive over all 17 ring
                # positions, so it does not reward many independent local errors
                # the way token accuracy does.
                "best_val_exact_match": target_metrics.exact_match,
                "curve": [{"step": r["step"], "val_loss": r.get("val_loss"),
                           "val_token_accuracy": r.get("val_token_accuracy")}
                          for r in outcome.history if "val_loss" in r],
            })
            print(f"    best val token {accuracy:.4f}%  EXACT MATCH {target_metrics.exact_match:.4f}%  "
                  f"loss {float(loss):.4f}  distinct {distinct}")
            write_partial()

    by_k = {str(k): classify([r for r in records if r["k"] == k], prereg) for k in k_values}
    majority = majority_class_percent(val.targets)
    distinct_by_k_seed = {
        str(k): {d["seed"]: d["best_val_distinct_predicted_tokens"] for d in detail if d["k"] == k}
        for k in k_values
    }
    amended = amend_report(by_k, majority, distinct_by_k_seed)

    # Exact match per K: the best value over that K's seeds, since one cell
    # producing correct rings is what shows the function is learnable at all.
    exact_by_k = {}
    for k in k_values:
        values = [d["best_val_exact_match"] for d in detail
                  if d["k"] == k and d.get("best_val_exact_match") is not None]
        exact_by_k[k] = max(values) if values else None

    readings = {str(k): exact_match_reading(amended[str(k)]["classification_amended"], exact_by_k[k])
                for k in k_values}
    bounds = {str(k): all(d["post_training_bound_held"] for d in detail if d["k"] == k) for k in k_values}
    learnable_k = [k for k in k_values if amended[str(k)]["classification_amended"] == "LEARNABLE"]

    applied = batch04_outcome(
        {k: amended[str(k)]["classification_amended"] for k in k_values},
        exact_by_k,
        {k: bounds[str(k)] for k in k_values},
        k_values)

    report = {
        "batch": "04", "probe": f"{NAME}_smoke" if smoke else NAME, "smoke": smoke,
        "resumed": resume,
        "recovered_cells": [[d["k"], d["seed"]] for d in detail
                            if d.get("source") == "recovered_from_checkpoint"],
        "learnable_k": learnable_k,
        "best_exact_match_by_k": {str(k): exact_by_k[k] for k in k_values},
        "exact_match_floor_percent": EXACT_MATCH_FLOOR_PERCENT,
        "exact_match_reading_by_k": readings,
        "post_training_bound_held_by_k": bounds,
        "preregistered_outcome": applied,
        "question": "with the stream bounded and a curriculum climbing to it, is T=4 learnable, and at which K?",
        "preregistration": "PHASE2_BATCH_04_PREREG.md",
        "config": dict(config),
        "target_t": TARGET_T,
        "majority_class_percent_t4": majority,
        "g2_passes_at_target_depth": True,
        "validation_split_sha256": {str(t): s.sha256 for t, s in splits.items()},
        "classification_by_k": by_k,
        "amended_classification_by_k": amended,
        "provenance": provenance,
        "detail": detail,
    }
    report_name = f"{NAME}_smoke_report.json" if smoke else f"{NAME}_report.json"
    write_json(run_dir / report_name, report)

    print(f"\n--- {NAME}{' SMOKE' if smoke else ''} ---")
    print(f"  T=4 majority-class floor: {majority:.4f} %   exact-match floor: {EXACT_MATCH_FLOOR_PERCENT} %")
    for k in k_values:
        a, o = amended[str(k)], by_k[str(k)]
        em = exact_by_k[k]
        print(f"  K={k}: {o['classification']} -> {a['classification_amended']}"
              f"{'  CHANGED' if a['changed'] else ''}"
              f"   best exact match {('%.4f %%' % em) if em is not None else '—'}"
              f"   -> {readings[str(k)]['reading']}")
    print(f"  learnable K: {learnable_k}   post-training bound held by K: {bounds}")
    print(f"  pre-registered outcome: {applied['outcome']}")
    print(f"     {applied['reason']}")
    print(f"report: {run_dir / report_name}")
    if smoke:
        print("SMOKE ONLY — not a result. See PHASE2_BATCH_04_PREREG.md §2 for the real configuration.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny step counts, separate directory; exercises the path, not a result")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run: skip cells with a best checkpoint, recompute "
                             "their metrics from it, and take their curves from the console log")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
