#!/usr/bin/env python3
"""Batch 01 diagnostic arms (PHASE2_BATCH_01_PREREG.md, commit 7dbd7b6).

Each arm changes exactly one thing against the reference configuration, which is
Probe A and has already been run. Nothing here is part of the Phase 2 grid, and
no number produced by these arms is evidence for the depth hypothesis.

The test split is never read. Existing result directories are never overwritten:
a rerun goes to a sibling __attemptN directory.

Usage:
    python scripts/phase2_batch01.py --arm 1     # fixed-set fit (gate)
    python scripts/phase2_batch01.py --arm 2a    # step-0 loss, fix off vs on
    python scripts/phase2_batch01.py --arm 2b    # REF + residual fix
    python scripts/phase2_batch01.py --arm 3     # curriculum
    python scripts/phase2_batch01.py --arm 4     # warmup + 2x steps
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from phase2.batch01 import (  # noqa: E402
    CurriculumStream,
    FixedSetStream,
    WarmupCosine,
    apply_final_layer_norm,
    step_zero_loss,
)
from phase2.compute import compute_provenance, configure_determinism  # noqa: E402
from phase2.config import CellSpec, ModelVariant, TrainingConfig, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.groups import get_group  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.probes import classify  # noqa: E402
from phase2.runner import splits_dir  # noqa: E402
from phase2.training import evaluate_split_with_loss, predict_with_loss, train_fixed_steps  # noqa: E402
from utils.config import OptimizerConfig  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402
import yaml  # noqa: E402

PREREG = Path("configs/phase2/probe_preregistration.yaml")
ROOT = "phase2/batch01"
TARGET_T = 4
WIDTH = ModelVariant(name="medium", base="phase2/prototype_medium")
K_VALUES = (1, 8)
SEEDS = (0, 1, 2)
BATCH, EVAL_EVERY = 128, 250
FIXED_SET_SIZE = 512
STEP0_THRESHOLD_NATS = 0.5


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
        print("CUDA unavailable; Batch 01 runs on GPU (Amendment 02a P1).", file=sys.stderr)
        raise SystemExit(2)
    device = torch.device("cuda")
    determinism = configure_determinism(device)
    provenance = compute_provenance(device, determinism)
    provenance["batch"] = "batch01"
    provenance["preregistration"] = "PHASE2_BATCH_01_PREREG.md@7dbd7b6"
    benchmark = load_benchmark()
    model_cfg = resolve_model_variant(WIDTH, 60)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    val = store.get(benchmark.group, "main", "val", TARGET_T, benchmark.n, benchmark.val_size)
    return device, provenance, benchmark, model_cfg, val


def training_config(steps: int, eval_batch: int) -> TrainingConfig:
    return TrainingConfig(optimizer=OptimizerConfig("adamw", 1e-3, 0.01), batch_size=BATCH,
                          max_steps=steps, eval_every=EVAL_EVERY, eval_batch_size=eval_batch)


def per_example_se(predictions, targets) -> float:
    per = (predictions == targets).mean(axis=1) * 100.0
    return float(per.std(ddof=1) / np.sqrt(len(per)))


def record_for(model, val, device, benchmark, seed, k, outcome, cell_dir) -> dict:
    """Validation record in the shape phase2.probes.classify expects."""
    predictions, loss = predict_with_loss(model, val.inputs, val.targets, device, benchmark.val_size)
    accuracy = float((predictions == val.targets).mean() * 100.0)
    evaluations = sum(1 for row in outcome.history if "val_loss" in row)
    return {
        "seed": seed, "k": k,
        "best_val": {"token_accuracy": accuracy, "loss": float(loss)},
        "final_val": {"token_accuracy": accuracy, "loss": outcome.final_val_loss},
        "num_evaluations": evaluations,
        "per_example_token_accuracy_se": per_example_se(predictions, val.targets),
        "cell_dir": str(cell_dir),
    }


def build(cell_family, k, model_cfg, n, device, seed, fix: bool):
    set_seed(seed)
    built = build_model(cell_family, k, model_cfg, n)
    model = built.model.to(device)
    patched = apply_final_layer_norm(model, model_cfg.vocab_size) if fix else None
    return model, built, patched


# ---------------------------------------------------------------------------


def arm1(args):
    """Fixed-set fit: can the model optimise at this depth at all?"""
    device, provenance, benchmark, model_cfg, val = setup()
    run_dir = out_dir("arm1_fixed_set")
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for k in K_VALUES:
        seed = SEEDS[0]
        stream = FixedSetStream(benchmark.group, "main", TARGET_T, benchmark.n, BATCH, FIXED_SET_SIZE,
                                benchmark.data_seed, seed, excluded_inputs=[val.inputs])
        fit_split = stream.as_split()
        model, built, _ = build("tesseract", k, model_cfg, benchmark.n, device, seed, fix=False)
        print(f"=== Arm 1: T={TARGET_T} K={k} seed={seed} fixed set {FIXED_SET_SIZE} steps={args.steps}")
        outcome = train_fixed_steps(model, stream, fit_split, None, training_config(args.steps, FIXED_SET_SIZE),
                                    device, 60, log=print)
        metrics, loss = evaluate_split_with_loss(model, fit_split, device, 60, FIXED_SET_SIZE)
        row = {
            "k": k, "seed": seed, "fixed_set_size": FIXED_SET_SIZE,
            "fixed_set_sha256": fit_split.sha256,
            "steps_run": outcome.steps_run,
            "fit_exact_match": metrics.exact_match,
            "fit_token_accuracy": metrics.token_accuracy,
            "fit_loss": float(loss),
            "fits": bool(metrics.exact_match >= 99.0),
            "curve": [{"step": r["step"], "loss": r.get("val_loss"), "exact_match": r.get("val_exact_match"),
                       "token_accuracy": r.get("val_token_accuracy")}
                      for r in outcome.history if "val_loss" in r],
            "parameter_count": built.parameter_count,
        }
        results.append(row)
        print(f"    fit EM {metrics.exact_match:.2f}%  token {metrics.token_accuracy:.2f}%  loss {loss:.4f}"
              f"  -> {'FITS' if row['fits'] else 'DOES NOT FIT'}")

    report = {"arm": "1", "question": "can the model optimise at T=4 at all?",
              "note": "trained and evaluated on the same fixed set; NOT task performance",
              "bar": "train exact match >= 99%", "provenance": provenance, "results": results,
              "gate_passed": any(r["fits"] for r in results)}
    write_json(run_dir / "arm1_report.json", report)
    print(f"\nArm 1 gate: {'PASS' if report['gate_passed'] else 'FAIL'} -> {run_dir/'arm1_report.json'}")
    return 0


def arm2a(args):
    """Step-0 loss with and without the residual fix. No training."""
    device, provenance, benchmark, model_cfg, val = setup()
    run_dir = out_dir("arm2a_step0")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in (1, 2, 4, 8):
        off_model, _, _ = build("tesseract", k, model_cfg, benchmark.n, device, SEEDS[0], fix=False)
        off = step_zero_loss(off_model, val, device, benchmark.val_size)
        on_model, _, patched = build("tesseract", k, model_cfg, benchmark.n, device, SEEDS[0], fix=True)
        on = step_zero_loss(on_model, val, device, benchmark.val_size)
        rows.append({"k": k, "step0_loss_fix_off": off, "step0_loss_fix_on": on, "patched_module": patched})
        print(f"  K={k}: off={off:9.4f}  on={on:8.4f}")

    k1 = next(r for r in rows if r["k"] == 1)
    k8 = next(r for r in rows if r["k"] == 8)
    gap_off = abs(k8["step0_loss_fix_off"] - k1["step0_loss_fix_off"])
    gap_on = abs(k8["step0_loss_fix_on"] - k1["step0_loss_fix_on"])
    report = {
        "arm": "2a", "question": "how large is the residual-stream confound, and does the fix remove it?",
        "chance_loss_nats": float(np.log(60)), "rows": rows,
        "k8_minus_k1_gap_fix_off": gap_off, "k8_minus_k1_gap_fix_on": gap_on,
        "threshold_nats": STEP0_THRESHOLD_NATS,
        "arm2b_gated_on": bool(gap_on <= STEP0_THRESHOLD_NATS),
        "provenance": provenance,
    }
    write_json(run_dir / "arm2a_report.json", report)
    print(f"\n  gap K8-K1: off={gap_off:.4f}  on={gap_on:.4f}  (threshold {STEP0_THRESHOLD_NATS})")
    print(f"Arm 2b gated {'ON' if report['arm2b_gated_on'] else 'OFF'} -> {run_dir/'arm2a_report.json'}")
    return 0


def _training_arm(name, args, *, fix: bool, curriculum: bool, steps: int, schedule=None, question=""):
    device, provenance, benchmark, model_cfg, val = setup()
    run_dir = out_dir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    records, detail = [], []

    for k in K_VALUES:
        for seed in SEEDS:
            cell = CellSpec("tesseract", k, benchmark.group, "main", TARGET_T, seed)
            cell_dir = run_dir / "cells" / cell.cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            model, built, patched = build("tesseract", k, model_cfg, benchmark.n, device, seed, fix=fix)

            if curriculum:
                stream = CurriculumStream(benchmark.group, "main", benchmark.n, BATCH, benchmark.data_seed, seed,
                                          [1, 2, 4], 5000, TARGET_T, 20000, excluded_inputs=[val.inputs])
                boundaries = stream.stage_boundaries()
            else:
                from phase2.data import TrainingStream
                stream = TrainingStream(benchmark.group, "main", TARGET_T, benchmark.n, BATCH,
                                        benchmark.data_seed, seed, excluded_inputs=[val.inputs])
                boundaries = None

            lr_fn = (lambda step: schedule.lr_at(step)) if schedule is not None else None
            print(f"=== {name}: T={TARGET_T} K={k} seed={seed} steps={steps} fix={fix} curriculum={curriculum}")
            outcome = train_fixed_steps(model, stream, val, None, training_config(steps, benchmark.val_size),
                                        device, 60, lr_schedule=lr_fn, log=print)

            torch.save(outcome.final_state, cell_dir / "final_checkpoint.pt")
            if outcome.best_state is not None:
                torch.save(outcome.best_state, cell_dir / "best_checkpoint.pt")
                model.load_state_dict(outcome.best_state)
            record = record_for(model, val, device, benchmark, seed, k, outcome, cell_dir)
            records.append(record)

            entry = {"k": k, "seed": seed, "cell_id": cell.cell_id, "steps_run": outcome.steps_run,
                     "stopped_reason": outcome.stopped_reason, "patched_module": patched,
                     "curve": [{"step": r["step"], "val_loss": r.get("val_loss"),
                                "val_token_accuracy": r.get("val_token_accuracy")}
                               for r in outcome.history if "val_loss" in r]}
            if boundaries:
                entry["stage_boundaries"] = boundaries
                entry["per_stage_diagnostics"] = _stage_diagnostics(model, benchmark, device, outcome, boundaries)
            detail.append(entry)
            print(f"    best val token {record['best_val']['token_accuracy']:.4f}%  loss {record['best_val']['loss']:.4f}")

    by_k = {str(k): classify([r for r in records if r["k"] == k], prereg) for k in K_VALUES}
    report = {"arm": name, "question": question, "fix": fix, "curriculum": curriculum, "steps": steps,
              "schedule": schedule.__dict__ if schedule else None,
              "classification_by_k": by_k, "provenance": provenance, "detail": detail}
    write_json(run_dir / f"{name}_report.json", report)
    print(f"\n--- {name} ---")
    for k, o in by_k.items():
        print(f"  K={k}: {o['classification']}  above-chance seeds {o['seeds_above_chance']}")
    print(f"report: {run_dir / f'{name}_report.json'}")
    return 0


def _stage_diagnostics(model, benchmark, device, outcome, boundaries):
    """Validation accuracy at T=1,2,4 after training. Diagnostic only."""
    store = SplitStore(splits_dir(), benchmark.data_seed)
    rows = []
    for t in (1, 2, 4):
        split = store.get(benchmark.group, "main", "val", t, benchmark.n, benchmark.val_size)
        metrics, loss = evaluate_split_with_loss(model, split, device, 60, benchmark.val_size)
        rows.append({"t": t, "token_accuracy": metrics.token_accuracy, "loss": float(loss),
                     "exact_match": metrics.exact_match})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["1", "2a", "2b", "3", "4"])
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()

    if args.arm == "1":
        return arm1(args)
    if args.arm == "2a":
        return arm2a(args)
    if args.arm == "2b":
        return _training_arm("arm2b_residual_fix", args, fix=True, curriculum=False, steps=20000,
                             question="does removing the residual confound change learnability?")
    if args.arm == "3":
        return _training_arm("arm3_curriculum", args, fix=False, curriculum=True, steps=35000,
                             question="is a cold-start gradient the blocker?")
    if args.arm == "4":
        return _training_arm("arm4_warmup", args, fix=False, curriculum=False, steps=40000,
                             schedule=WarmupCosine(peak=1e-3, final=1e-4, warmup_steps=2000, total_steps=40000),
                             question="is the plateau an optimisation artefact?")
    return 1


if __name__ == "__main__":
    sys.exit(main())
