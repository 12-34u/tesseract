#!/usr/bin/env python3
"""Audit diagnostic: is the cellular automaton's 0 % validation exact match real?

This is NOT a Phase 1 experiment, and the dashboard does not show it. It re-trains
selected (T, K) cells with the unmodified experiment code and configuration
(same seeds, same RNG order, so each cell should reproduce its results.csv row
exactly) and then:

1. Recomputes train/val token accuracy and exact match with an independent
   pure-Python implementation on the final model.
2. Evaluates a freshly regenerated copy of the training set, which must match
   the train metrics exactly.
3. Reports how many tokens are wrong in each validation sequence, and the
   per-position validation accuracy.
4. Compares the observed exact match with the estimate token_acc ** seq_len
   (what you would expect if token errors were independent).
5. Control: re-trains the same cell on a larger training set, to check whether
   the evaluation pipeline can register validation exact matches at all.

Usage:
    python scripts/diagnose_ca_generalization.py [--cells 1:4 8:4] [--control-train-size 4096]

Writes <runs root>/ca_generalization_diagnostic/diagnostic.json (not versioned).
"""

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.toy_cellular_automaton import CellularAutomatonDataset
from experiments.cellular_automaton import make_datasets, train_cell
from models.tesseract import TesseractModel
from utils.config import CellularAutomatonConfig, load_config
from utils.device import resolve_device
from utils.paths import resolve_run_dir, runs_root
from utils.seed import set_seed


def independent_metrics(predictions: List[List[int]], targets: List[List[int]]) -> Dict[str, Any]:
    """Token accuracy / exact match / error counts with plain Python (no torch)."""
    correct_tokens = total_tokens = exact = 0
    errors_per_sequence = []
    for pred, tgt in zip(predictions, targets):
        matches = [p == t for p, t in zip(pred, tgt)]
        correct_tokens += sum(matches)
        total_tokens += len(matches)
        exact += all(matches)
        errors_per_sequence.append(len(matches) - sum(matches))
    return {
        "token_accuracy": 100.0 * correct_tokens / total_tokens,
        "exact_match": 100.0 * exact / len(targets),
        "errors_per_sequence": errors_per_sequence,
    }


@torch.no_grad()
def predict(model: TesseractModel, inputs: torch.Tensor) -> List[List[int]]:
    model.eval()
    return model(inputs)[0].argmax(dim=-1).cpu().tolist()


def run_cell(config: CellularAutomatonConfig, t: int, k: int, device: torch.device) -> Dict[str, Any]:
    seed = config.experiment.seed
    d = config.data

    # Mirror experiments/cellular_automaton.py RNG order exactly.
    set_seed(seed)
    base_state = TesseractModel.from_config(config.model, 1).state_dict()
    set_seed(seed)
    train_ds, val_ds = make_datasets(config, t)
    set_seed(seed)
    model = TesseractModel.from_config(config.model, k).to(device)
    model.load_state_dict(base_state, strict=True)

    train_inputs, train_targets = train_ds.inputs.to(device), train_ds.targets.to(device)
    val_inputs, val_targets = val_ds.inputs.to(device), val_ds.targets.to(device)
    reported, _ = train_cell(model, train_inputs, train_targets, val_inputs, val_targets, config.training)

    train = independent_metrics(predict(model, train_inputs), train_ds.targets.tolist())
    val = independent_metrics(predict(model, val_inputs), val_ds.targets.tolist())
    regenerated = CellularAutomatonDataset(d.num_train, d.seq_len, d.rule_number, t, seed=seed)
    regen = independent_metrics(predict(model, regenerated.inputs.to(device)), regenerated.targets.tolist())

    val_preds = torch.tensor(predict(model, val_inputs))
    per_position = (val_preds == val_ds.targets).float().mean(dim=0).mul(100).tolist()

    return {
        "T": t,
        "K": k,
        "num_train": d.num_train,
        "num_val": d.num_val,
        "training_steps": reported["training_steps"],
        "stopped_reason": reported["stopped_reason"],
        "reported": {key: reported[key] for key in
                     ("train_token_accuracy", "train_exact_match", "val_token_accuracy", "val_exact_match")},
        "independent_train": {k_: v for k_, v in train.items() if k_ != "errors_per_sequence"},
        "independent_val": {k_: v for k_, v in val.items() if k_ != "errors_per_sequence"},
        "regenerated_train_matches": regen["token_accuracy"] == train["token_accuracy"]
        and regen["exact_match"] == train["exact_match"],
        "val_errors_per_sequence_histogram": dict(sorted(Counter(val["errors_per_sequence"]).items())),
        "val_per_position_accuracy_range": [min(per_position), max(per_position)],
        "independence_estimate_val_exact_match": 100.0 * (val["token_accuracy"] / 100.0) ** d.seq_len,
    }


def committed_row(config: CellularAutomatonConfig, t: int, k: int) -> Dict[str, str] | None:
    path = resolve_run_dir(config.experiment.output_dir) / "results.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        return next((r for r in csv.DictReader(f) if int(r["T"]) == t and int(r["K"]) == k), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", nargs="+", default=["1:4", "8:4"], help="T:K pairs")
    parser.add_argument("--control-train-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config("cellular_automaton", CellularAutomatonConfig)
    device = resolve_device(args.device)
    cells = [tuple(int(v) for v in cell.split(":")) for cell in args.cells]
    report: Dict[str, Any] = {"device": str(device), "cells": [], "controls": []}

    for t, k in cells:
        print(f"\n=== Phase 1 configuration: T={t}, K={k} ===")
        result = run_cell(config, t, k, device)
        row = committed_row(config, t, k)
        if row is not None:
            result["matches_results_csv"] = (
                float(row["val_exact_match"]) == result["reported"]["val_exact_match"]
                and float(row["train_exact_match"]) == result["reported"]["train_exact_match"]
                and int(row["training_steps"]) == result["training_steps"]
            )
        report["cells"].append(result)
        print(json.dumps(result, indent=2))

    control_config = replace(config, data=replace(config.data, num_train=args.control_train_size))
    for t, k in cells:
        print(f"\n=== Control: num_train={args.control_train_size}, T={t}, K={k} ===")
        result = run_cell(control_config, t, k, device)
        report["controls"].append(result)
        print(json.dumps(result, indent=2))

    out_dir = runs_root() / "ca_generalization_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out_dir / 'diagnostic.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
