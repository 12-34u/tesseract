#!/usr/bin/env python3
"""Batch 05 — depth or compute? (PHASE2_BATCH_05_PREREG.md)

Three arms at T = 3 with forward compute matched to within 0.008 %, differing
only in how that compute is spent:

    A  tesseract K=4      837,436 params   4 block calls
    B  width_scaled       3,214,678        1 block call
    C  unrolled L=4       3,206,716        4 block calls

Arm A uses 3.84x fewer parameters than either alternative, which is the point: if
it matches or beats them, depth is being bought without parameters.

The curriculum, budget and seeds are Batch 03's exactly, so arm A replicates its
K=4 cells; failing to do so is a batch failure rather than a finding. The fix is
on for all three arms, which required implementing per-step normalisation for the
unrolled family (prereg §3).

The comparison is keyed on exact match, with the threshold and escalation band
fixed before any cell ran. ``--smoke`` exercises the path at trivial step counts
into a separate directory. ``--resume`` continues an interrupted run.

Usage:
    python scripts/phase2_batch05.py [--smoke] [--resume]
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
    DIAGNOSTIC_T_VALUES,
    SHORTCUT_CANDIDATE,
    attribute_classification,
    shortcut_agreement_percent,
    stage_boundary_steps,
)
from phase2.batch05 import (  # noqa: E402
    ARMS,
    BATCH05_CONFIG as PROBE_CONFIG,
    GROWTH_LIMIT,
    arm_is_excluded,
    bound_held,
    compare,
    inherited_matches_batch03,
)
from phase2.compute import compute_provenance, configure_determinism  # noqa: E402
from phase2.config import ModelVariant, TrainingConfig, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.groups import get_group  # noqa: E402
from phase2.metrics import compute_metrics  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.probes import classify  # noqa: E402
from phase2.reclassify import amend_report, distinct_predicted_tokens, majority_class_percent  # noqa: E402
from phase2.runner import splits_dir  # noqa: E402
from phase2.salvage import salvage  # noqa: E402
from phase2.training import predict_with_loss, train_fixed_steps  # noqa: E402
from utils.config import OptimizerConfig  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402

PREREG = Path("configs/phase2/probe_preregistration.yaml")
ROOT = "phase2/batch05"
NAME = "t3_family"
TARGET_T = PROBE_CONFIG["target_t"]
WIDTH = ModelVariant(name="medium", base="phase2/prototype_medium")
SEEDS = (0, 1, 2)
BATCH, EVAL_EVERY = 128, 250

#: Trivial substitute for the real configuration, for --smoke only. Same code
#: path and the same number of stage boundaries, ~40 steps instead of 35,000.
SMOKE_CONFIG = {**PROBE_CONFIG, "steps_per_stage": 10, "target_steps": 10, "steps": 40, "seeds": 1}


def resume_dir(name: str) -> Optional[Path]:
    """The existing run directory a resume should continue into, or None."""
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
        print("CUDA unavailable; Batch 05 runs on GPU (Amendment 02a P1).", file=sys.stderr)
        raise SystemExit(2)
    device = torch.device("cuda")
    determinism = configure_determinism(device)
    provenance = compute_provenance(device, determinism)
    provenance["batch"] = "batch05"
    provenance["preregistration"] = "PHASE2_BATCH_05_PREREG.md"
    benchmark = load_benchmark()
    model_cfg = resolve_model_variant(WIDTH, 60)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    splits = {t: store.get(benchmark.group, "main", "val", t, benchmark.n, benchmark.val_size)
              for t in DIAGNOSTIC_T_VALUES}
    return device, provenance, benchmark, model_cfg, splits


def training_config(steps: int, eval_batch: int, eval_every: int = EVAL_EVERY) -> TrainingConfig:
    return TrainingConfig(optimizer=OptimizerConfig("adamw", 1e-3, 0.01), batch_size=BATCH,
                          max_steps=steps, eval_every=eval_every, eval_batch_size=eval_batch)


def per_example_se(predictions, targets) -> float:
    per = (predictions == targets).mean(axis=1) * 100.0
    return float(per.std(ddof=1) / np.sqrt(len(per)))


def build_arm(family: str, depth: int, model_cfg, n: int, device, seed: int):
    """Build one arm's model with the fix applied. Returns (built, model, patched)."""
    set_seed(seed)
    built = build_model(family, depth, model_cfg, n)
    model = built.model.to(device)
    patched = apply_per_step_norm(model, elementwise_affine=False)
    return built, model, patched


def depth_diagnostics(model, splits, group, device, eval_batch) -> list:
    """Validation numbers at every diagnostic depth, including exact match.

    Exact match is the batch's primary metric, so it is recorded at every
    boundary and not only on the best checkpoint.
    """
    rows = []
    for t in DIAGNOSTIC_T_VALUES:
        split = splits[t]
        predictions, loss = predict_with_loss(model, split.inputs, split.targets, device, eval_batch)
        metrics = compute_metrics(predictions, split.targets, 60)
        row = {"t": t, "token_accuracy": metrics.token_accuracy, "exact_match": metrics.exact_match,
               "loss": float(loss), "distinct_predicted_tokens": distinct_predicted_tokens(predictions),
               "majority_class_percent": majority_class_percent(split.targets)}
        if t in SHORTCUT_CANDIDATE:
            row["shortcut_candidate"] = SHORTCUT_CANDIDATE[t]
            row["shortcut_agreement_percent"] = shortcut_agreement_percent(predictions, group, split.inputs, t)
        rows.append(row)
    return rows


def bounded_growth_gate(device, benchmark, model_cfg, val, run_dir, arms) -> bool:
    """Per-step RMS must be bounded for every arm before any cell trains.

    Blocks the whole batch on failure. Arm C previously could not be bounded at
    all, and running it unbounded against a bounded arm A would reproduce the
    defect that invalidated Batch 02's K comparison.
    """
    rows = []
    for arm_id, family, depth, label in arms:
        built, model, patched = build_arm(family, depth, model_cfg, benchmark.n, device, SEEDS[0])
        norms = hidden_state_norms(model, val.inputs[:256], device)
        growth = norms[-1]["z_L_rms"] / norms[0]["z_L_rms"]
        rows.append({"arm": arm_id, "family": family, "depth": depth, "label": label,
                     "patched_module": patched, "block_executions": len(norms),
                     "parameter_count": built.parameter_count,
                     "forward_flops_per_sequence": built.forward_flops_per_sequence,
                     "z_L_rms_first": norms[0]["z_L_rms"], "z_L_rms_last": norms[-1]["z_L_rms"],
                     "growth_ratio": growth, "bounded": bool(growth < GROWTH_LIMIT),
                     # Arm B executes one block, so it has no between-step growth
                     # to bound; its pass is trivially true rather than earned.
                     "bound_is_vacuous": len(norms) == 1})
        print(f"  {arm_id} {family:14s} params={built.parameter_count:>9,} "
              f"flops={built.forward_flops_per_sequence:>11,}  "
              f"rms {norms[0]['z_L_rms']:.4f} -> {norms[-1]['z_L_rms']:.4f}  growth {growth:.4f}x"
              f"  {'OK' if growth < GROWTH_LIMIT else 'FAIL'}"
              f"{'  (vacuous: 1 block call)' if len(norms) == 1 else ''}")
    passed = all(r["bounded"] for r in rows)
    write_json(run_dir / "bounded_growth_gate.json",
               {"gate": "batch05-i", "growth_limit": GROWTH_LIMIT, "rows": rows, "gate_passed": passed})
    print(f"  bounded-growth gate: {'PASS' if passed else 'FAIL'}")
    return passed


def cell_id(arm_id: str, family: str, depth: int, seed: int) -> str:
    return f"{arm_id}_{family}_d{depth}_A5_main_T{TARGET_T}_s{seed}"


def recover_cell(arm, cell_dir, checkpoint, seed, model_cfg, benchmark, splits, group, device, salvaged_cell):
    """Rebuild a finished cell from its checkpoint, without retraining."""
    arm_id, family, depth, label = arm
    val = splits[TARGET_T]
    built, model, patched = build_arm(family, depth, model_cfg, benchmark.n, device, seed)
    # weights_only=True: these are plain state_dicts, so there is no reason to
    # let a checkpoint unpickle arbitrary objects.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    predictions, loss = predict_with_loss(model, val.inputs, val.targets, device, benchmark.val_size)
    metrics = compute_metrics(predictions, val.targets, 60)
    agreement = shortcut_agreement_percent(predictions, group, val.inputs, TARGET_T)
    distinct = distinct_predicted_tokens(predictions)
    post = hidden_state_norms(model, val.inputs[:256], device)
    growth = post[-1]["z_L_rms"] / post[0]["z_L_rms"]
    curve = (salvaged_cell or {}).get("curve") or []

    record = {
        "seed": seed, "arm": arm_id,
        "best_val": {"token_accuracy": metrics.token_accuracy, "loss": float(loss)},
        "final_val": {"token_accuracy": metrics.token_accuracy,
                      "loss": (curve[-1]["val_loss"] if curve else None)},
        "num_evaluations": len(curve),
        "per_example_token_accuracy_se": per_example_se(predictions, val.targets),
        "cell_dir": str(cell_dir),
    }
    entry = {
        "arm": arm_id, "family": family, "depth": depth, "label": label, "seed": seed,
        "cell_id": cell_dir.name, "steps_run": curve[-1]["step"] if curve else None,
        "stopped_reason": "recovered_from_checkpoint", "patched_module": patched,
        "parameter_count": built.parameter_count,
        "forward_flops_per_sequence": built.forward_flops_per_sequence,
        "seconds_per_train_step": None,
        "post_training_per_step_rms": post, "post_training_growth_ratio": growth,
        "post_training_bound_held": bound_held(growth),
        "stage_boundary_validation": (salvaged_cell or {}).get("stage_boundary_validation") or [],
        "best_checkpoint_validation_by_t": depth_diagnostics(model, splits, group, device, benchmark.val_size),
        "best_val_exact_match": metrics.exact_match,
        "best_val_distinct_predicted_tokens": distinct,
        "best_val_shortcut_agreement_percent": agreement,
        "curve": curve,
        "source": "recovered_from_checkpoint",
        "recovered_from_console_log": ["curve", "stage_boundary_validation", "steps_run", "final_val.loss"],
    }
    console = (salvaged_cell or {}).get("console_summary") or {}
    drift = (None if not console
             else abs(console.get("best_val_token_accuracy", 0.0) - metrics.token_accuracy))
    entry["console_vs_recomputed_accuracy_delta"] = drift
    print(f"=== {NAME}: arm {arm_id} seed={seed} RECOVERED from checkpoint (no retrain)")
    print(f"    exact match {metrics.exact_match:.4f}%  token {metrics.token_accuracy:.4f}%  "
          f"growth {growth:.4f}x {'held' if bound_held(growth) else 'BROKE'}")
    if drift is not None:
        print(f"    console log said {console['best_val_token_accuracy']:.4f}%  delta {drift:.4f} "
              f"{'OK' if drift < 0.01 else '*** MISMATCH ***'}")
    return entry, record


def run(args) -> int:
    import time

    smoke = bool(getattr(args, "smoke", False))
    config = SMOKE_CONFIG if smoke else PROBE_CONFIG
    seeds = SEEDS[:config["seeds"]]
    eval_every = 5 if smoke else EVAL_EVERY
    resume = bool(getattr(args, "resume", False)) and not smoke

    device, provenance, benchmark, model_cfg, splits = setup()
    val = splits[TARGET_T]
    group = get_group(benchmark.group)
    run_dir = (resume_dir(NAME) if resume else None) or out_dir(f"{NAME}_smoke" if smoke else NAME)
    run_dir.mkdir(parents=True, exist_ok=True)
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))

    salvaged = {}
    if resume:
        log = Path(f"runs/phase2/batch05_{NAME}.console.log")
        if log.is_file():
            salvaged = salvage(log)
            print(f"=== resume: {run_dir} ({sum(1 for v in salvaged.values() if v['complete'])} cells in log)")

    boundaries = stage_boundary_steps(config["curriculum_stages"], config["steps_per_stage"],
                                      config["target_steps"])
    assert boundaries[-1] == config["steps"], boundaries
    if not inherited_matches_batch03():
        print("Batch 05 config drifted from Batch 03 on an inherited field; refusing to run.", file=sys.stderr)
        return 2

    print("=== Batch 05 gate: per-step hidden-state RMS must be bounded for every arm")
    if not bounded_growth_gate(device, benchmark, model_cfg, val, run_dir, ARMS):
        print("Bounded-growth gate failed; no training cell runs (PHASE2_BATCH_05_PREREG.md §3.1).",
              file=sys.stderr)
        return 1

    records, detail = [], []

    def write_partial():
        write_json(run_dir / f"{NAME}_partial.json",
                   {"batch": "05", "partial": True, "cells_done": len(detail),
                    "cells_expected": len(ARMS) * len(seeds), "config": dict(config),
                    "provenance": provenance, "records": records, "detail": detail})

    for arm in ARMS:
        arm_id, family, depth, label = arm
        for seed in seeds:
            cell_dir = run_dir / "cells" / cell_id(arm_id, family, depth, seed)
            cell_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = cell_dir / "best_checkpoint.pt"
            if resume and checkpoint.is_file():
                entry, record = recover_cell(arm, cell_dir, checkpoint, seed, model_cfg, benchmark,
                                             splits, group, device, salvaged.get((depth, seed)))
                records.append(record)
                detail.append(entry)
                write_partial()
                continue

            built, model, patched = build_arm(family, depth, model_cfg, benchmark.n, device, seed)
            stream = CurriculumStream(benchmark.group, "main", benchmark.n, BATCH, benchmark.data_seed, seed,
                                      list(config["curriculum_stages"]), config["steps_per_stage"],
                                      TARGET_T, config["target_steps"], excluded_inputs=[val.inputs])
            schedule = stream.stage_boundaries()

            def boundary_probe(step, _model=model, _schedule=schedule):
                stage = next((s for s in _schedule if s["ends_at_step"] == step), None)
                rows = depth_diagnostics(_model, splits, group, device, benchmark.val_size)
                print(f"    [stage boundary] step {step} ends stage T={stage['t'] if stage else '?'}: "
                      + "  ".join(f"T{r['t']} acc {r['token_accuracy']:.4f}% EM {r['exact_match']:.4f}%"
                                  for r in rows))
                return {"stage_t": stage["t"] if stage else None, "validation_by_t": rows}

            print(f"=== {NAME}{' SMOKE' if smoke else ''}: arm {arm_id} ({label}) seed={seed} "
                  f"steps={config['steps']} params={built.parameter_count:,} "
                  f"flops={built.forward_flops_per_sequence:,} fix=per_step(non-affine)")
            started = time.perf_counter()
            outcome = train_fixed_steps(model, stream, val, None,
                                        training_config(config["steps"], benchmark.val_size, eval_every),
                                        device, 60, step_hooks={b: boundary_probe for b in boundaries},
                                        log=print)
            elapsed = time.perf_counter() - started

            torch.save(outcome.final_state, cell_dir / "final_checkpoint.pt")
            boundary_records = list(outcome.hook_records)
            if outcome.best_state is not None:
                torch.save(outcome.best_state, cell_dir / "best_checkpoint.pt")
                model.load_state_dict(outcome.best_state)

            predictions, loss = predict_with_loss(model, val.inputs, val.targets, device, benchmark.val_size)
            metrics = compute_metrics(predictions, val.targets, 60)
            agreement = shortcut_agreement_percent(predictions, group, val.inputs, TARGET_T)
            distinct = distinct_predicted_tokens(predictions)
            post = hidden_state_norms(model, val.inputs[:256], device)
            growth = post[-1]["z_L_rms"] / post[0]["z_L_rms"]
            held = bound_held(growth)

            records.append({
                "seed": seed, "arm": arm_id,
                "best_val": {"token_accuracy": metrics.token_accuracy, "loss": float(loss)},
                "final_val": {"token_accuracy": metrics.token_accuracy, "loss": outcome.final_val_loss},
                "num_evaluations": sum(1 for r in outcome.history if "val_loss" in r),
                "per_example_token_accuracy_se": per_example_se(predictions, val.targets),
                "cell_dir": str(cell_dir),
            })
            detail.append({
                "arm": arm_id, "family": family, "depth": depth, "label": label, "seed": seed,
                "cell_id": cell_dir.name, "steps_run": outcome.steps_run,
                "stopped_reason": outcome.stopped_reason, "patched_module": patched,
                "parameter_count": built.parameter_count,
                "forward_flops_per_sequence": built.forward_flops_per_sequence,
                # Equal FLOPs is not equal time: one wide matmul and four narrow
                # sequential ones differ in GPU utilisation, so wall clock is
                # recorded rather than inferred from the FLOP count.
                "seconds_per_train_step": elapsed / max(outcome.steps_run, 1),
                "per_step_norm_elementwise_affine": False,
                "post_training_per_step_rms": post, "post_training_growth_ratio": growth,
                "post_training_bound_held": held,
                "stage_boundaries": schedule,
                "stage_boundary_validation": boundary_records,
                "best_checkpoint_validation_by_t": depth_diagnostics(model, splits, group, device,
                                                                    benchmark.val_size),
                "best_val_exact_match": metrics.exact_match,
                "best_val_distinct_predicted_tokens": distinct,
                "best_val_shortcut_agreement_percent": agreement,
                "curve": [{"step": r["step"], "val_loss": r.get("val_loss"),
                           "val_token_accuracy": r.get("val_token_accuracy")}
                          for r in outcome.history if "val_loss" in r],
                "source": "trained",
            })
            print(f"    EXACT MATCH {metrics.exact_match:.4f}%  token {metrics.token_accuracy:.4f}%  "
                  f"loss {float(loss):.4f}  distinct {distinct}  sc {agreement:.2f}%  "
                  f"growth {growth:.4f}x {'held' if held else 'BROKE'}  "
                  f"{elapsed / max(outcome.steps_run, 1):.5f} s/step")
            write_partial()

    arm_ids = [a[0] for a in ARMS]
    by_arm = {a: classify([r for r in records if r["arm"] == a], prereg) for a in arm_ids}
    majority = majority_class_percent(val.targets)
    distinct_by_arm_seed = {a: {d["seed"]: d["best_val_distinct_predicted_tokens"]
                                for d in detail if d["arm"] == a} for a in arm_ids}
    amended = amend_report(by_arm, majority, distinct_by_arm_seed)
    attribution = {a: attribute_classification(
        amended[a], {d["seed"]: d["best_val_shortcut_agreement_percent"] for d in detail if d["arm"] == a})
        for a in arm_ids}

    exact_by_arm, bounds, excluded = {}, {}, []
    for a in arm_ids:
        cells = [d for d in detail if d["arm"] == a]
        values = [c["best_val_exact_match"] for c in cells if c.get("best_val_exact_match") is not None]
        exact_by_arm[a] = max(values) if values else None
        bounds[a] = bool(cells) and all(c["post_training_bound_held"] for c in cells)
        why = arm_is_excluded(amended[a]["classification_amended"], attribution[a]["attribution"],
                              exact_by_arm[a], bounds[a])
        if why:
            excluded.append({"arm": a, "reason": why})

    comparison = compare(exact_by_arm, [e["arm"] for e in excluded])

    report = {
        "batch": "05", "probe": f"{NAME}_smoke" if smoke else NAME, "smoke": smoke, "resumed": resume,
        "question": "at matched forward compute, does recursion beat width and does weight sharing cost anything?",
        "preregistration": "PHASE2_BATCH_05_PREREG.md",
        "config": dict(config), "target_t": TARGET_T,
        "majority_class_percent_t3": majority,
        "validation_split_sha256": {str(t): s.sha256 for t, s in splits.items()},
        "classification_by_arm": by_arm,
        "amended_classification_by_arm": amended,
        "shortcut_attribution_by_arm": attribution,
        "best_exact_match_by_arm": exact_by_arm,
        "post_training_bound_held_by_arm": bounds,
        "excluded_arms": excluded,
        "comparison": comparison,
        "provenance": provenance, "detail": detail,
    }
    report_name = f"{NAME}_smoke_report.json" if smoke else f"{NAME}_report.json"
    write_json(run_dir / report_name, report)

    print(f"\n--- {NAME}{' SMOKE' if smoke else ''} ---")
    print(f"  T=3 majority-class floor: {majority:.4f} %   shortcut ceiling: 26.6257 %")
    for a in arm_ids:
        em = exact_by_arm[a]
        print(f"  arm {a}: {by_arm[a]['classification']} -> {amended[a]['classification_amended']}"
              f"   best exact match {('%.4f %%' % em) if em is not None else '—'}"
              f"   attribution {attribution[a]['attribution']}   bound {bounds[a]}")
    if excluded:
        print(f"  excluded: {excluded}")
    print(f"  COMPARISON: {comparison['reading']}")
    print(f"     {comparison['reason']}")
    print(f"report: {run_dir / report_name}")
    if smoke:
        print("SMOKE ONLY — not a result. See PHASE2_BATCH_05_PREREG.md §4.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny step counts, separate directory; exercises the path, not a result")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from cells that already have a best checkpoint")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
