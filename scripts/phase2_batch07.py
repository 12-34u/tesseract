#!/usr/bin/env python3
"""Batch 07 — is K = 8's T = 4 result depth or compute? (PHASE2_BATCH_07_PREREG.md)

Three arms at T = 4, a primary benchmark depth, with forward compute matched to
within 0.004 %:

    A  tesseract K=8      837,436 params   8 block calls
    B  width_scaled       6,383,992        1 block call
    C  unrolled L=8       6,365,756        8 block calls

Arm A is 7.6x smaller than either alternative. Its three cells are REUSED from
Batch 06, whose configuration is bit-identical, plus one fresh cell at seed 0 as a
reproduction check against 98.7793 % exact match.

Phase 1 runs arms B and C at the inherited lr 1e-3. Phase 2 runs them at 3e-4,
automatically and only if phase 1 places arm A ahead of both — the reading whose
interpretation the inherited rate would undermine.

The plateau test is the corrected conjunction (magnitude AND t > 2), pre-registered
for Batch 07 onward and not retroactive.

Usage:
    python scripts/phase2_batch07.py [--smoke] [--resume]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from phase2.batch01 import CurriculumStream, apply_per_step_norm, hidden_state_norms  # noqa: E402
from phase2.batch02 import SHORTCUT_CANDIDATE, shortcut_agreement_percent, stage_boundary_steps  # noqa: E402
from phase2.batch04 import DIAGNOSTIC_T_VALUES, EXACT_MATCH_FLOOR_PERCENT  # noqa: E402
from phase2.batch05 import arm_is_excluded  # noqa: E402
from phase2.batch07 import (  # noqa: E402
    ARMS,
    ARM_A_REPRODUCTION_EXACT_MATCH,
    ARM_A_REPRODUCTION_SEED,
    ARM_A_REPRODUCTION_TOLERANCE_POINTS,
    BATCH06_CONSOLE_LOG,
    BATCH06_RUN_DIR,
    BATCH07_CONFIG as PROBE_CONFIG,
    GROWTH_LIMIT,
    PHASE1_LR,
    PHASE2_LR,
    bound_held,
    compare,
    final_slope,
    has_plateaued,
    phase2_reading,
    phase2_triggered,
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
ROOT = "phase2/batch07"
NAME = "t4_family"
TARGET_T = PROBE_CONFIG["target_t"]
WIDTH = ModelVariant(name="medium", base="phase2/prototype_medium")
SEEDS = (0, 1, 2)
BATCH, EVAL_EVERY = PROBE_CONFIG["batch_size"], 250

#: Evaluation batch, reduced from the 2,048-example split to fit the 3.68 GiB card.
#: Arm B's d_ff of 11,836 needs 3.33 GiB to evaluate the split in one pass and
#: OOMs; at 512 the peak is 0.88 GiB. Evaluation is chunked, and for a trained
#: model the batch is exactly equivalent — arm A's Batch 06 checkpoint gives
#: identical exact match, token accuracy and loss at 256, 512, 1024 and 2048. For a
#: cell sitting at chance the argmax can flip on near-tied logits, which can move a
#: distinct-token count marginally; that only ever affects the collapse check,
#: whose role is to refuse a positive claim, so it cannot manufacture one.
EVAL_BATCH = 512

#: Trivial substitute for --smoke only: same code path, same boundary count,
#: ~50 steps instead of 90,000, and no arm A reuse (nothing to reuse from).
SMOKE_CONFIG = {**PROBE_CONFIG, "rung_steps": [10, 10, 10, 10], "target_steps": 10,
                "steps": 50, "seeds": 1}


def resume_dir(name: str) -> Optional[Path]:
    base = resolve_run_dir(f"{ROOT}/{name}")
    candidates = sorted((p for p in base.parent.glob(base.name + "*")
                         if p.is_dir() and "_smoke" not in p.name and (p / "cells").is_dir()),
                        key=lambda p: p.name)
    return candidates[-1] if candidates else None


def out_dir(name: str) -> Path:
    base = resolve_run_dir(f"{ROOT}/{name}")
    if not base.exists() or not any(base.iterdir()):
        return base
    attempt = 2
    while (base.parent / f"{base.name}__attempt{attempt}").exists():
        attempt += 1
    return base.parent / f"{base.name}__attempt{attempt}"


def setup():
    if not torch.cuda.is_available():
        print("CUDA unavailable; Batch 07 runs on GPU (Amendment 02a P1).", file=sys.stderr)
        raise SystemExit(2)
    device = torch.device("cuda")
    determinism = configure_determinism(device)
    provenance = compute_provenance(device, determinism)
    provenance["batch"] = "batch07"
    provenance["preregistration"] = "PHASE2_BATCH_07_PREREG.md"
    benchmark = load_benchmark()
    model_cfg = resolve_model_variant(WIDTH, 60)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    # benchmark.val_size, not EVAL_BATCH: EVAL_BATCH is the forward-pass chunk,
    # the split is the frozen 2,048-example validation set.
    splits = {t: store.get(benchmark.group, "main", "val", t, benchmark.n, benchmark.val_size)
              for t in DIAGNOSTIC_T_VALUES}
    return device, provenance, benchmark, model_cfg, splits


def training_config(steps: int, eval_batch: int, lr: float, eval_every: int = EVAL_EVERY) -> TrainingConfig:
    """The learning rate is a parameter because phase 2 varies it."""
    return TrainingConfig(optimizer=OptimizerConfig(PROBE_CONFIG["optimizer"], lr, PROBE_CONFIG["weight_decay"]),
                          batch_size=BATCH, max_steps=steps, eval_every=eval_every, eval_batch_size=eval_batch)


def optimiser_record(lr: float) -> Dict:
    """The resolved optimiser configuration, recorded per cell (prereg §3)."""
    return {"optimizer": PROBE_CONFIG["optimizer"], "learning_rate": lr,
            "weight_decay": PROBE_CONFIG["weight_decay"], "batch_size": BATCH}


def per_example_se(predictions, targets) -> float:
    per = (predictions == targets).mean(axis=1) * 100.0
    return float(per.std(ddof=1) / np.sqrt(len(per)))


def build_arm(family: str, depth: int, model_cfg, n: int, device, seed: int):
    set_seed(seed)
    built = build_model(family, depth, model_cfg, n)
    model = built.model.to(device)
    patched = apply_per_step_norm(model, elementwise_affine=False)
    return built, model, patched


def depth_diagnostics(model, splits, group, device, eval_batch) -> List[Dict]:
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


def bounded_growth_gate(device, benchmark, model_cfg, val, run_dir) -> bool:
    """Per-step RMS must be bounded for every arm before any cell trains."""
    rows = []
    for arm_id, family, depth, label in ARMS:
        built, model, patched = build_arm(family, depth, model_cfg, benchmark.n, device, SEEDS[0])
        norms = hidden_state_norms(model, val.inputs[:256], device)
        growth = norms[-1]["z_L_rms"] / norms[0]["z_L_rms"]
        del model
        torch.cuda.empty_cache()
        rows.append({"arm": arm_id, "family": family, "depth": depth, "label": label,
                     "patched_module": patched, "block_executions": len(norms),
                     "parameter_count": built.parameter_count,
                     "forward_flops_per_sequence": built.forward_flops_per_sequence,
                     "growth_ratio": growth, "bounded": bool(growth < GROWTH_LIMIT),
                     "bound_is_vacuous": len(norms) == 1})
        print(f"  {arm_id} {family:14s} params={built.parameter_count:>9,} "
              f"flops={built.forward_flops_per_sequence:>11,}  growth {growth:.4f}x"
              f"  {'OK' if growth < GROWTH_LIMIT else 'FAIL'}"
              f"{'  (vacuous: 1 block call)' if len(norms) == 1 else ''}")
    passed = all(r["bounded"] for r in rows)
    write_json(run_dir / "bounded_growth_gate.json",
               {"gate": "batch07-i", "growth_limit": GROWTH_LIMIT, "rows": rows, "gate_passed": passed})
    print(f"  bounded-growth gate: {'PASS' if passed else 'FAIL'}")
    return passed


def cell_name(arm_id: str, family: str, depth: int, seed: int, lr: float, tag: str = "") -> str:
    rate = f"lr{lr:g}".replace("-", "m").replace(".", "p")
    return f"{arm_id}_{family}_d{depth}_{rate}_T{TARGET_T}_s{seed}{tag}"


def finish_cell(entry: Dict) -> Dict:
    """Attach the slope statistics and plateau verdict to a completed cell."""
    stats = final_slope(entry["curve"]) if entry.get("curve") else None
    entry["final_slope"] = stats
    entry["has_plateaued"] = has_plateaued(stats)
    return entry


def evaluate_model(model, built, patched, val, splits, group, device, benchmark) -> Dict:
    """Everything measured on the final model, shared by trained and reused cells."""
    predictions, loss = predict_with_loss(model, val.inputs, val.targets, device, EVAL_BATCH)
    metrics = compute_metrics(predictions, val.targets, 60)
    post = hidden_state_norms(model, val.inputs[:256], device)
    growth = post[-1]["z_L_rms"] / post[0]["z_L_rms"]
    return {
        "metrics": metrics, "loss": float(loss),
        "distinct": distinct_predicted_tokens(predictions),
        "shortcut_agreement_percent": (shortcut_agreement_percent(predictions, group, val.inputs, TARGET_T)
                                       if TARGET_T in SHORTCUT_CANDIDATE else None),
        "post_training_per_step_rms": post, "post_training_growth_ratio": growth,
        "post_training_bound_held": bound_held(growth),
        "best_checkpoint_validation_by_t": depth_diagnostics(model, splits, group, device, EVAL_BATCH),
        "per_example_token_accuracy_se": per_example_se(predictions, val.targets),
        "parameter_count": built.parameter_count,
        "forward_flops_per_sequence": built.forward_flops_per_sequence,
        "patched_module": patched,
    }


def reuse_arm_a(model_cfg, benchmark, splits, group, device, salvaged) -> Tuple[List[Dict], List[Dict]]:
    """Arm A's three cells, recomputed from Batch 06's checkpoints (prereg §2.1)."""
    arm_id, family, depth, label = ARMS[0]
    val = splits[TARGET_T]
    base = resolve_run_dir(BATCH06_RUN_DIR) / "cells"
    entries, records = [], []
    for seed in SEEDS:
        checkpoint = base / f"tesseract_d{depth}_A5_main_T{TARGET_T}_s{seed}" / "best_checkpoint.pt"
        if not checkpoint.is_file():
            print(f"!!! arm A seed {seed}: no Batch 06 checkpoint at {checkpoint}", file=sys.stderr)
            continue
        built, model, patched = build_arm(family, depth, model_cfg, benchmark.n, device, seed)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        ev = evaluate_model(model, built, patched, val, splits, group, device, benchmark)
        cell = salvaged.get((depth, seed)) or {}
        curve = cell.get("curve") or []
        records.append({
            "seed": seed, "arm": arm_id, "phase": 1,
            "best_val": {"token_accuracy": ev["metrics"].token_accuracy, "loss": ev["loss"]},
            "final_val": {"token_accuracy": ev["metrics"].token_accuracy,
                          "loss": curve[-1]["val_loss"] if curve else None},
            "num_evaluations": len(curve),
            "per_example_token_accuracy_se": ev["per_example_token_accuracy_se"],
            "cell_dir": str(checkpoint.parent),
        })
        entries.append(finish_cell({
            "arm": arm_id, "family": family, "depth": depth, "label": label, "seed": seed, "phase": 1,
            "cell_id": checkpoint.parent.name, "source": "reused_from_batch06",
            "optimiser": optimiser_record(PHASE1_LR),
            "steps_run": curve[-1]["step"] if curve else None,
            "stopped_reason": "reused_from_batch06", "seconds_per_train_step": None,
            "stage_boundary_validation": cell.get("stage_boundary_validation") or [],
            "best_val_exact_match": ev["metrics"].exact_match,
            "best_val_distinct_predicted_tokens": ev["distinct"],
            "best_val_shortcut_agreement_percent": ev["shortcut_agreement_percent"],
            "curve": curve,
            "recovered_from_console_log": ["curve", "stage_boundary_validation", "steps_run", "final_val.loss"],
            **{k: ev[k] for k in ("post_training_per_step_rms", "post_training_growth_ratio",
                                 "post_training_bound_held", "best_checkpoint_validation_by_t",
                                 "parameter_count", "forward_flops_per_sequence", "patched_module")},
        }))
        print(f"=== {NAME}: arm A seed {seed} REUSED from Batch 06 (no retrain)")
        print(f"    EXACT MATCH {ev['metrics'].exact_match:.4f}%  token {ev['metrics'].token_accuracy:.4f}%  "
              f"growth {ev['post_training_growth_ratio']:.4f}x")
    return entries, records


def train_cell(arm, seed, lr, phase, config, run_dir, model_cfg, benchmark, splits, group, device,
               eval_every, tag="") -> Tuple[Dict, Dict]:
    """Train one cell and return (detail entry, classification record)."""
    arm_id, family, depth, label = arm
    val = splits[TARGET_T]
    cell_dir = run_dir / "cells" / cell_name(arm_id, family, depth, seed, lr, tag)
    cell_dir.mkdir(parents=True, exist_ok=True)

    built, model, patched = build_arm(family, depth, model_cfg, benchmark.n, device, seed)
    stream = CurriculumStream(benchmark.group, "main", benchmark.n, BATCH, benchmark.data_seed, seed,
                              list(config["curriculum_stages"]), list(config["rung_steps"]),
                              TARGET_T, config["target_steps"], excluded_inputs=[val.inputs])
    schedule = stream.stage_boundaries()
    boundaries = [b["ends_at_step"] for b in schedule]

    def boundary_probe(step, _model=model, _schedule=schedule):
        stage = next((b for b in _schedule if b["ends_at_step"] == step), None)
        rows = depth_diagnostics(_model, splits, group, device, EVAL_BATCH)
        print(f"    [stage boundary] step {step} ends stage T={stage['t'] if stage else '?'}: "
              + "  ".join(f"T{r['t']} acc {r['token_accuracy']:.4f}% EM {r['exact_match']:.4f}%" for r in rows))
        return {"stage_t": stage["t"] if stage else None, "validation_by_t": rows}

    print(f"=== {NAME}: phase {phase} arm {arm_id} ({label}) seed={seed} lr={lr:g} "
          f"steps={config['steps']} params={built.parameter_count:,} "
          f"flops={built.forward_flops_per_sequence:,}{' ' + tag if tag else ''}")
    started = time.perf_counter()
    outcome = train_fixed_steps(model, stream, val, None,
                               training_config(config["steps"], EVAL_BATCH, lr, eval_every),
                               device, 60, step_hooks={b: boundary_probe for b in boundaries}, log=print)
    elapsed = time.perf_counter() - started

    torch.save(outcome.final_state, cell_dir / "final_checkpoint.pt")
    boundary_records = list(outcome.hook_records)
    if outcome.best_state is not None:
        torch.save(outcome.best_state, cell_dir / "best_checkpoint.pt")
        model.load_state_dict(outcome.best_state)

    ev = evaluate_model(model, built, patched, val, splits, group, device, benchmark)
    record = {
        "seed": seed, "arm": arm_id, "phase": phase,
        "best_val": {"token_accuracy": ev["metrics"].token_accuracy, "loss": ev["loss"]},
        "final_val": {"token_accuracy": ev["metrics"].token_accuracy, "loss": outcome.final_val_loss},
        "num_evaluations": sum(1 for r in outcome.history if "val_loss" in r),
        "per_example_token_accuracy_se": ev["per_example_token_accuracy_se"],
        "cell_dir": str(cell_dir),
    }
    entry = finish_cell({
        "arm": arm_id, "family": family, "depth": depth, "label": label, "seed": seed, "phase": phase,
        "cell_id": cell_dir.name, "source": "trained", "tag": tag or None,
        "optimiser": optimiser_record(lr),
        "steps_run": outcome.steps_run, "stopped_reason": outcome.stopped_reason,
        "seconds_per_train_step": elapsed / max(outcome.steps_run, 1),
        "stage_boundaries": schedule, "stage_boundary_validation": boundary_records,
        "best_val_exact_match": ev["metrics"].exact_match,
        "best_val_distinct_predicted_tokens": ev["distinct"],
        "best_val_shortcut_agreement_percent": ev["shortcut_agreement_percent"],
        "curve": [{"step": r["step"], "val_loss": r.get("val_loss"),
                   "val_token_accuracy": r.get("val_token_accuracy")}
                  for r in outcome.history if "val_loss" in r],
        **{k: ev[k] for k in ("post_training_per_step_rms", "post_training_growth_ratio",
                              "post_training_bound_held", "best_checkpoint_validation_by_t",
                              "parameter_count", "forward_flops_per_sequence", "patched_module")},
    })
    stats = entry["final_slope"] or {}
    print(f"    EXACT MATCH {ev['metrics'].exact_match:.4f}%  token {ev['metrics'].token_accuracy:.4f}%  "
          f"loss {ev['loss']:.4f}  distinct {ev['distinct']}  "
          f"growth {ev['post_training_growth_ratio']:.4f}x "
          f"{'held' if ev['post_training_bound_held'] else 'BROKE'}")
    if stats:
        print(f"    slope {stats['slope_points_per_1000_steps']:+.3f}/1k  t={stats['t_statistic']:.2f}  "
              f"-> {'plateaued' if entry['has_plateaued'] else 'STILL RISING'}  "
              f"{entry['seconds_per_train_step']:.5f} s/step")
    return entry, record


def summarise(detail, records, prereg, val, arm_ids) -> Dict:
    """Classify, amend, and compare, over whichever cells are present."""
    by_arm = {a: classify([r for r in records if r["arm"] == a], prereg) for a in arm_ids}
    majority = majority_class_percent(val.targets)
    distinct_by_arm = {a: {d["seed"]: d["best_val_distinct_predicted_tokens"]
                           for d in detail if d["arm"] == a} for a in arm_ids}
    amended = amend_report(by_arm, majority, distinct_by_arm)
    exact, bounds, plateaued, excluded = {}, {}, {}, []
    for a in arm_ids:
        cells = [d for d in detail if d["arm"] == a]
        values = [c["best_val_exact_match"] for c in cells if c.get("best_val_exact_match") is not None]
        exact[a] = max(values) if values else None
        bounds[a] = bool(cells) and all(c["post_training_bound_held"] for c in cells)
        flags = [c["has_plateaued"] for c in cells
                 if c["has_plateaued"] is not None
                 and (c.get("best_val_exact_match") or 0.0) >= EXACT_MATCH_FLOOR_PERCENT]
        plateaued[a] = all(flags) if flags else None
        why = arm_is_excluded(amended[a]["classification_amended"], None, exact[a], bounds[a])
        if why:
            excluded.append({"arm": a, "reason": why})
    comparison = compare(exact, [e["arm"] for e in excluded])
    learned = [a for a in arm_ids if plateaued[a] is not None]
    all_plateaued = (all(plateaued[a] for a in learned) if learned else None)
    return {"classification_by_arm": by_arm, "amended_classification_by_arm": amended,
            "best_exact_match_by_arm": exact, "post_training_bound_held_by_arm": bounds,
            "plateaued_by_arm": plateaued, "all_plateaued": all_plateaued,
            "excluded_arms": excluded, "comparison": comparison,
            "majority_class_percent": majority}


def run(args) -> int:
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
    arm_ids = [a[0] for a in ARMS]

    boundaries = stage_boundary_steps(config["curriculum_stages"], config["rung_steps"], config["target_steps"])
    assert boundaries[-1] == config["steps"], boundaries

    print("=== Batch 07 gate: per-step hidden-state RMS must be bounded for every arm")
    if not bounded_growth_gate(device, benchmark, model_cfg, val, run_dir):
        print("Bounded-growth gate failed; no training cell runs (PHASE2_BATCH_07_PREREG.md §4).",
              file=sys.stderr)
        return 1

    detail, records = [], []

    def write_partial(phase):
        write_json(run_dir / f"{NAME}_partial.json",
                   {"batch": "07", "partial": True, "phase": phase, "cells_done": len(detail),
                    "config": dict(config), "provenance": provenance,
                    "records": records, "detail": detail})

    # ---- phase 1 -----------------------------------------------------------
    if not smoke:
        log = Path(BATCH06_CONSOLE_LOG)
        salvaged = salvage(log) if log.is_file() else {}
        print(f"=== arm A: reusing Batch 06's cells ({len(salvaged)} cells in its log)")
        a_entries, a_records = reuse_arm_a(model_cfg, benchmark, splits, group, device, salvaged)
        detail.extend(finish_cell(e) for e in a_entries)
        records.extend(a_records)
        write_partial(1)

        # The reproduction check: one fresh arm A cell at seed 0.
        entry, record = train_cell(ARMS[0], ARM_A_REPRODUCTION_SEED, PHASE1_LR, 1, config, run_dir,
                                   model_cfg, benchmark, splits, group, device, eval_every,
                                   tag="_repro")
        delta = abs(entry["best_val_exact_match"] - ARM_A_REPRODUCTION_EXACT_MATCH)
        entry["reproduction_check"] = {
            "target_exact_match": ARM_A_REPRODUCTION_EXACT_MATCH, "delta_points": delta,
            "tolerance_points": ARM_A_REPRODUCTION_TOLERANCE_POINTS,
            "passed": delta <= ARM_A_REPRODUCTION_TOLERANCE_POINTS}
        print(f"    REPRODUCTION CHECK: {entry['best_val_exact_match']:.4f}% vs "
              f"{ARM_A_REPRODUCTION_EXACT_MATCH}% target, delta {delta:.4f} -> "
              f"{'PASS' if entry['reproduction_check']['passed'] else '*** FAIL — environment drifted ***'}")
        detail.append(entry)
        # The repro cell is a check, not a fourth arm A sample, so it is kept out
        # of the classification records to avoid double-counting seed 0.
        write_partial(1)

    for arm in (ARMS if smoke else ARMS[1:]):
        for seed in seeds:
            entry, record = train_cell(arm, seed, PHASE1_LR, 1, config, run_dir, model_cfg, benchmark,
                                       splits, group, device, eval_every)
            detail.append(entry)
            records.append(record)
            write_partial(1)

    phase1 = summarise(detail, records, prereg, val, arm_ids)
    trigger = phase2_triggered(phase1["comparison"]["reading"], phase1["all_plateaued"])
    print(f"\n--- {NAME} phase 1 ---")
    for a in arm_ids:
        em = phase1["best_exact_match_by_arm"][a]
        print(f"  arm {a}: {phase1['amended_classification_by_arm'][a]['classification_amended']}"
              f"   best exact match {('%.4f %%' % em) if em is not None else '—'}"
              f"   plateaued {phase1['plateaued_by_arm'][a]}")
    print(f"  COMPARISON: {phase1['comparison']['reading']}")
    print(f"     {phase1['comparison']['reason']}")
    print(f"  PHASE 2 {'TRIGGERED' if trigger['triggered'] else 'not triggered'}: {trigger['reason']}")

    # ---- phase 2, conditional ---------------------------------------------
    phase2 = None
    if trigger["triggered"] and not smoke:
        print(f"\n=== phase 2: arms B and C at lr {PHASE2_LR:g} (prereg §7)")
        p2_detail, p2_records = [], []
        for arm in ARMS[1:]:
            for seed in seeds:
                entry, record = train_cell(arm, seed, PHASE2_LR, 2, config, run_dir, model_cfg, benchmark,
                                           splits, group, device, eval_every)
                detail.append(entry)
                p2_detail.append(entry)
                p2_records.append(record)
                write_partial(2)
        p2_exact = {}
        for a in arm_ids[1:]:
            values = [c["best_val_exact_match"] for c in p2_detail if c["arm"] == a
                      and c.get("best_val_exact_match") is not None]
            p2_exact[a] = max(values) if values else None
        phase2 = {"learning_rate": PHASE2_LR, "best_exact_match_by_arm": p2_exact,
                  "reading": phase2_reading(phase1["best_exact_match_by_arm"], p2_exact)}
        print(f"\n--- {NAME} phase 2 ---")
        for a in arm_ids[1:]:
            print(f"  arm {a} at {PHASE2_LR:g}: best exact match "
                  f"{('%.4f %%' % p2_exact[a]) if p2_exact[a] is not None else '—'}")
        print(f"  READING: {phase2['reading']['reading']}")
        print(f"     {phase2['reading']['reason']}")

    report = {
        "batch": "07", "probe": f"{NAME}_smoke" if smoke else NAME, "smoke": smoke, "resumed": resume,
        "question": "at matched forward compute, is K=8's T=4 result depth or compute?",
        "preregistration": "PHASE2_BATCH_07_PREREG.md",
        "config": dict(config), "target_t": TARGET_T,
        "g2_passes_at_target_depth": True,
        "exact_match_floor_percent": EXACT_MATCH_FLOOR_PERCENT,
        "validation_split_sha256": {str(t): s.sha256 for t, s in splits.items()},
        "phase1": phase1, "phase2_trigger": trigger, "phase2": phase2,
        "provenance": provenance, "detail": detail,
    }
    name = f"{NAME}_smoke_report.json" if smoke else f"{NAME}_report.json"
    write_json(run_dir / name, report)
    print(f"\nreport: {run_dir / name}")
    if smoke:
        print("SMOKE ONLY — not a result. See PHASE2_BATCH_07_PREREG.md §3.")
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
