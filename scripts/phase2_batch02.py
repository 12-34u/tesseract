#!/usr/bin/env python3
"""Batch 02 — the T = 3 learnability probe (PHASE2_BATCH_02_PREREG.md).

Arm 3's curriculum with the ladder's top moved from 4 to 3, and nothing else
changed: medium width, no residual fix, K in {1, 8}, seeds 0/1/2, 5,000 steps
per ladder stage and 20,000 at the target, constant 1e-3 AdamW.

Two things Batch 01 did not do:

* validation accuracy at T = 1, 2, 3 is recorded **at each stage boundary**, as
  a report field, which is the deviation Batch 01 §6 recorded and this fixes;
* every cell's agreement with the G2-failing shallow candidate at T = 3 is
  recorded, because that candidate scores 26.6 % and the LEARNABLE bar is
  6.67 %, so accuracy alone cannot tell composition from the shortcut.

The classification is the pre-registered rule and the Addendum 03 amended rule,
both reported, with T = 3's own majority-class frequency. The test split is
never read.

``--smoke`` exercises the whole path at trivial step counts to catch a crash
before the real run spends three GPU-hours reaching one. It writes to a
separate directory, records the flag in its report, and cannot be mistaken for
a result: PHASE2_BATCH_02_PREREG.md §3 fixes the configuration, and a smoke
report carries ``"smoke": true`` and a step count that is not 35,000.

Usage:
    python scripts/phase2_batch02.py [--smoke]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from phase2.batch01 import CurriculumStream  # noqa: E402
from phase2.batch02 import (  # noqa: E402
    DIAGNOSTIC_T_VALUES,
    PROBE_CONFIG,
    SHORTCUT_CANDIDATE,
    attribute_classification,
    shortcut_agreement_percent,
    stage_boundary_steps,
)
from phase2.compute import compute_provenance, configure_determinism  # noqa: E402
from phase2.config import CellSpec, ModelVariant, TrainingConfig, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.probes import classify  # noqa: E402
from phase2.reclassify import amend_report, distinct_predicted_tokens, majority_class_percent  # noqa: E402
from phase2.runner import splits_dir  # noqa: E402
from phase2.groups import get_group  # noqa: E402
from phase2.metrics import compute_metrics  # noqa: E402
from phase2.training import predict_with_loss, train_fixed_steps  # noqa: E402
from utils.config import OptimizerConfig  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402

PREREG = Path("configs/phase2/probe_preregistration.yaml")
ROOT = "phase2/batch02"
NAME = "t3_curriculum"
TARGET_T = PROBE_CONFIG["target_t"]
WIDTH = ModelVariant(name="medium", base="phase2/prototype_medium")
K_VALUES = tuple(PROBE_CONFIG["k_values"])
SEEDS = (0, 1, 2)
BATCH, EVAL_EVERY = 128, 250


#: Trivial substitute for PROBE_CONFIG, for --smoke only. Same code path, same
#: number of stage boundaries, ~40 steps instead of 35,000.
SMOKE_CONFIG = {**PROBE_CONFIG, "curriculum_stages": [1, 2, 3], "steps_per_stage": 10,
                "target_steps": 10, "steps": 40, "k_values": [1], "seeds": 1}


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
        print("CUDA unavailable; Batch 02 runs on GPU (Amendment 02a P1).", file=sys.stderr)
        raise SystemExit(2)
    device = torch.device("cuda")
    determinism = configure_determinism(device)
    provenance = compute_provenance(device, determinism)
    provenance["batch"] = "batch02"
    provenance["preregistration"] = "PHASE2_BATCH_02_PREREG.md"
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


def run(args) -> int:
    smoke = bool(getattr(args, "smoke", False))
    config = SMOKE_CONFIG if smoke else PROBE_CONFIG
    k_values = tuple(config["k_values"])
    seeds = SEEDS[:config["seeds"]]
    eval_every = 5 if smoke else EVAL_EVERY
    device, provenance, benchmark, model_cfg, store, splits = setup()
    val = splits[TARGET_T]
    run_dir = out_dir(f"{NAME}_smoke" if smoke else NAME)
    run_dir.mkdir(parents=True, exist_ok=True)
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    group = get_group(benchmark.group)
    boundaries = stage_boundary_steps(config["curriculum_stages"], config["steps_per_stage"],
                                      config["target_steps"])
    assert boundaries[-1] == config["steps"], boundaries
    records, detail = [], []

    for k in k_values:
        for seed in seeds:
            cell = CellSpec("tesseract", k, benchmark.group, "main", TARGET_T, seed)
            cell_dir = run_dir / "cells" / cell.cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            set_seed(seed)
            built = build_model("tesseract", k, model_cfg, benchmark.n)
            model = built.model.to(device)

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
                  f"steps={config['steps']} fix=none curriculum={config['curriculum_stages']}->{TARGET_T}")
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
            agreement = shortcut_agreement_percent(predictions, group, val.inputs, TARGET_T)
            distinct = distinct_predicted_tokens(predictions)
            records.append({
                "seed": seed, "k": k,
                "best_val": {"token_accuracy": accuracy, "loss": float(loss)},
                "final_val": {"token_accuracy": accuracy, "loss": outcome.final_val_loss},
                "num_evaluations": sum(1 for row in outcome.history if "val_loss" in row),
                "per_example_token_accuracy_se": per_example_se(predictions, val.targets),
                "cell_dir": str(cell_dir),
            })
            detail.append({
                "k": k, "seed": seed, "cell_id": cell.cell_id, "steps_run": outcome.steps_run,
                "stopped_reason": outcome.stopped_reason, "parameter_count": built.parameter_count,
                "stage_boundaries": schedule,
                # The deviation fix: one entry per boundary, recorded, not printed.
                "stage_boundary_validation": final_boundary_records,
                "best_checkpoint_validation_by_t": depth_diagnostics(model, splits, group, device,
                                                                    benchmark.val_size),
                "best_val_distinct_predicted_tokens": distinct,
                "best_val_shortcut_agreement_percent": agreement,
                "curve": [{"step": r["step"], "val_loss": r.get("val_loss"),
                           "val_token_accuracy": r.get("val_token_accuracy")}
                          for r in outcome.history if "val_loss" in r],
            })
            print(f"    best val token {accuracy:.4f}%  loss {float(loss):.4f}  "
                  f"distinct {distinct}  shortcut agreement {agreement:.2f}%")

    by_k = {str(k): classify([r for r in records if r["k"] == k], prereg) for k in k_values}
    majority = majority_class_percent(val.targets)
    distinct_by_k_seed = {
        str(k): {d["seed"]: d["best_val_distinct_predicted_tokens"] for d in detail if d["k"] == k}
        for k in k_values
    }
    amended = amend_report(by_k, majority, distinct_by_k_seed)
    attribution = {
        str(k): attribute_classification(
            amended[str(k)],
            {d["seed"]: d["best_val_shortcut_agreement_percent"] for d in detail if d["k"] == k})
        for k in k_values
    }

    report = {
        "batch": "02", "probe": f"{NAME}_smoke" if smoke else NAME, "smoke": smoke,
        "question": "is T=3 learnable under the curriculum that solves T=1 and T=2?",
        "preregistration": "PHASE2_BATCH_02_PREREG.md",
        "config": dict(config),
        "target_t": TARGET_T,
        "majority_class_percent_t3": majority,
        "validation_split_sha256": {str(t): s.sha256 for t, s in splits.items()},
        "classification_by_k": by_k,
        "amended_classification_by_k": amended,
        "shortcut_attribution_by_k": attribution,
        "provenance": provenance,
        "detail": detail,
    }
    report_name = f"{NAME}_smoke_report.json" if smoke else f"{NAME}_report.json"
    write_json(run_dir / report_name, report)

    print(f"\n--- {NAME}{' SMOKE' if smoke else ''} ---")
    print(f"  T=3 majority-class floor: {majority:.4f} %")
    for k in k_values:
        a, o = amended[str(k)], by_k[str(k)]
        print(f"  K={k}: {o['classification']} -> {a['classification_amended']}"
              f"{'  CHANGED' if a['changed'] else ''}"
              f"   attribution {attribution[str(k)]['attribution']}")
    print(f"report: {run_dir / report_name}")
    if smoke:
        print("SMOKE ONLY — not a result. See PHASE2_BATCH_02_PREREG.md §3 for the real configuration.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny step counts, separate directory; exercises the path, not a result")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
