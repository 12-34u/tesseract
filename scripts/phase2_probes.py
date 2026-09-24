#!/usr/bin/env python3
"""P1c probes A and B, classified by the pre-registered rule.

Probe A: T=4, medium width, K in {1,8}, 3 seeds, cold start.
Probe B: T=8, medium width, K in {1,8}, 3 seeds, warm-started from a T=1
         solution trained at the same K. Diagnostic only.

Validation data only: the test split is never read. The classification comes
from configs/phase2/probe_preregistration.yaml, committed before execution, and
is applied mechanically by phase2.probes.classify.

Usage:
    python scripts/phase2_probes.py --probe a
    python scripts/phase2_probes.py --probe b
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from phase2.compute import compute_provenance, configure_determinism  # noqa: E402
from phase2.config import ModelVariant, TrainingConfig, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.groups import get_group  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.probes import classify  # noqa: E402
from phase2.runner import _cell_config, run_cell, splits_dir  # noqa: E402
from phase2.training import predict_with_loss  # noqa: E402
from utils.config import OptimizerConfig, RunConfig  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from phase2.config import CellSpec  # noqa: E402

PREREG_PATH = Path("configs/phase2/probe_preregistration.yaml")


def load_prereg() -> dict:
    return yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_state(state, model_cfg, benchmark, cell, split, device, vocab_size):
    """Validation metrics for a given parameter state, without training anything."""
    built = build_model(cell.family, cell.depth, model_cfg, benchmark.n)
    model = built.model.to(device)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    predictions, loss = predict_with_loss(model, split.inputs, split.targets, device, benchmark.val_size)
    correct = (predictions == split.targets)
    per_example = correct.mean(axis=1) * 100.0
    return {
        "token_accuracy": float(per_example.mean()),
        "exact_match": float((correct.all(axis=1) * 100.0).mean()),
        "loss": float(loss),
        "per_example_token_accuracy_se": float(per_example.std(ddof=1) / np.sqrt(len(per_example))),
    }


def initial_state(model_cfg, benchmark, cell):
    """The exact seeded initialisation run_cell would use, for the step-0 measurement."""
    set_seed(cell.seed)
    return {k: v.clone() for k, v in build_model(cell.family, cell.depth, model_cfg, benchmark.n)
            .model.state_dict().items()}


def num_evaluations(cell_dir: Path) -> int:
    path = Path(cell_dir) / "val_curve.csv"
    if not path.is_file():
        return 1
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.DictReader(f) if row.get("val_loss"))


def training_config(steps: int, benchmark, eval_every: int, batch_size: int) -> TrainingConfig:
    return TrainingConfig(optimizer=OptimizerConfig("adamw", 1e-3, 0.01), batch_size=batch_size,
                          max_steps=steps, eval_every=eval_every, eval_batch_size=benchmark.val_size)


def run_one(cell, model_cfg, benchmark, training, device, cell_dir, provenance, warm_start=None, stop_condition=None):
    experiment = RunConfig(name=f"phase2_probe_{cell.cell_id}", output_dir=str(cell_dir), seed=cell.seed, device=str(device))
    return run_cell(_cell_config(experiment, experiment.name, cell_dir, model_cfg, benchmark, training, cell),
                    PREREG_PATH, device, cell_dir, evaluate_test=False, compute_coverage=False,
                    provenance=provenance, warm_start=warm_start, stop_condition=stop_condition)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["a", "b"], required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable; probes must run in the CUDA environment (Amendment 02a P1).", file=sys.stderr)
        return 2

    prereg = load_prereg()
    spec = prereg["probes"][f"probe_{args.probe}"]
    steps = prereg["operational"]["fixed_step_budget"]
    seeds = prereg["operational"]["seeds"]

    device = torch.device("cuda")
    determinism = configure_determinism(device)
    provenance = compute_provenance(device, determinism)
    provenance["preregistration"] = {"id": prereg["id"], "date": prereg["date"],
                                     "sha256": sha256_file(PREREG_PATH)}

    benchmark = load_benchmark()
    model_cfg = resolve_model_variant(ModelVariant(name="medium", base="phase2/prototype_medium"), 60)
    vocab = get_group(benchmark.group).order
    store = SplitStore(splits_dir(), benchmark.data_seed)

    run_dir = resolve_run_dir(args.output_dir or f"phase2/probe_{args.probe}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"{run_dir} already contains results; it is preserved. Use a new output directory.")
    run_dir.mkdir(parents=True, exist_ok=True)

    target_t = spec["t"]
    target_val = store.get(benchmark.group, "main", "val", target_t, benchmark.n, benchmark.val_size)
    training = training_config(steps, benchmark, args.eval_every, args.batch_size)

    records, detail = [], []
    for k in spec["k_values"]:
        for seed in seeds:
            cell = CellSpec("tesseract", k, benchmark.group, "main", target_t, seed)
            cell_dir = run_dir / "cells" / cell.cell_id
            entry = {"k": k, "seed": seed, "cell_id": cell.cell_id, "t": target_t}

            # Auxiliary: step-0 validation loss, to separate residual-stream growth from depth.
            entry["step0"] = evaluate_state(initial_state(model_cfg, benchmark, cell), model_cfg, benchmark,
                                            cell, target_val, device, vocab)

            warm_start = None
            if spec["warm_start"]:
                warm_cell = CellSpec("tesseract", k, benchmark.group, "main", 1, seed)
                warm_dir = run_dir / "warm_start" / warm_cell.cell_id
                print(f"=== Probe B stage 1: T=1 K={k} seed={seed} (warm-start source) ===")
                warm_result = run_one(warm_cell, model_cfg, benchmark,
                                      training_config(steps, benchmark, args.eval_every, args.batch_size),
                                      device, warm_dir, provenance,
                                      stop_condition=lambda val: val.exact_match >= 99.0)
                checkpoint = warm_dir / ("best_checkpoint.pt" if (warm_dir / "best_checkpoint.pt").is_file()
                                         else "final_checkpoint.pt")
                state = torch.load(checkpoint, map_location=device)
                warm_start = {"state_dict": state, "source": str(checkpoint), "sha256": sha256_file(checkpoint),
                              "stage1_cell": warm_cell.cell_id, "stage1_steps_run": warm_result["steps_run"],
                              "stage1_stopped_reason": warm_result["stopped_reason"],
                              "stage1_best_val": warm_result.get("best_val")}
                # Auxiliary: what the T=1 solution scores on T=8 before any T=8 training.
                warm_start_reference = evaluate_state(state, model_cfg, benchmark, cell, target_val, device, vocab)
                entry["warm_start_reference_on_target_T"] = warm_start_reference
                entry["stage1"] = {k2: v for k2, v in warm_start.items() if k2 != "state_dict"}
                print(f"    stage 1 finished: {warm_result['stopped_reason']} at step {warm_result['steps_run']}; "
                      f"warm-start scores {warm_start_reference['token_accuracy']:.3f}% on T={target_t}")

            print(f"=== Probe {args.probe.upper()}: T={target_t} K={k} seed={seed} steps={steps} ===")
            result = run_one(cell, model_cfg, benchmark, training, device, cell_dir, provenance, warm_start=warm_start)

            best_state = cell_dir / "best_checkpoint.pt"
            best_eval = evaluate_state(torch.load(best_state, map_location=device), model_cfg, benchmark, cell,
                                       target_val, device, vocab) if best_state.is_file() else None
            record = {
                "seed": seed,
                "k": k,
                "best_val": {"token_accuracy": (best_eval or {}).get("token_accuracy"),
                             "loss": (best_eval or {}).get("loss")},
                "final_val": {"token_accuracy": (result.get("final_val") or {}).get("token_accuracy"),
                              "loss": result.get("final_val_loss")},
                "num_evaluations": num_evaluations(cell_dir),
                "per_example_token_accuracy_se": (best_eval or {}).get("per_example_token_accuracy_se"),
            }
            entry.update(record=record, steps_run=result["steps_run"], stopped_reason=result["stopped_reason"],
                         best_val_recorded=result.get("best_val"), final_val_recorded=result.get("final_val"))
            records.append(record)
            detail.append(entry)
            print(f"    best val token acc {record['best_val']['token_accuracy']}%  "
                  f"loss {record['best_val']['loss']}  (final loss {record['final_val']['loss']})")

    by_k = {}
    for k in spec["k_values"]:
        subset = [r for r in records if r["k"] == k]
        by_k[str(k)] = classify(subset, prereg)

    report = {
        "probe": args.probe,
        "spec": spec,
        "fixed_step_budget": steps,
        "seeds": seeds,
        "test_split_read": False,
        "provenance": provenance,
        "classification_by_k": by_k,
        "classification_pooled_over_k": classify(records, prereg),
        "detail": detail,
    }
    write_json(run_dir / "probe_report.json", report)

    print(f"\n--- Probe {args.probe.upper()} classification (pre-registered rule) ---")
    for k, outcome in by_k.items():
        print(f"  K={k}: {outcome['classification']}  "
              f"(learnable seeds {outcome['seeds_meeting_learnable_bar']}, "
              f"above-chance seeds {outcome['seeds_above_chance']})")
    print(f"report: {run_dir / 'probe_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
