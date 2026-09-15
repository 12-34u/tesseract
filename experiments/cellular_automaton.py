"""Tesseract Cellular Automaton experiment.

Task: predict the state of a 1D binary cellular automaton after T rule
applications, given only the initial state.

For each (T, K) in t_values × k_values a fresh model is trained from
identical initial weights; train and validation accuracy are recorded.

Audit note: for Rule 90 with T a power of two, target_i = x[i-T] XOR x[i+T]
exactly, so T in {1, 2, 4, 8} does not make targets depend on more input
cells. The summary re-checks this identity on the generated data.

Usage:
    python experiments/cellular_automaton.py [--config PATH] [--device auto|cpu|cuda] [--output-dir DIR]
"""

import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.toy_cellular_automaton import CellularAutomatonDataset, simulate_trajectory
from evaluation.metrics import EvalResult, evaluate, token_errors_per_sequence
from experiments.common import parse_setup
from models.tesseract import TesseractModel
from training.trainer import build_optimizer, train_step
from utils.config import CATrainingConfig, CellularAutomatonConfig
from utils.param_count import count_parameters
from utils.run_artifacts import RunRecorder
from utils.seed import set_seed

ARTIFACTS = [
    "results.csv",
    "train_em_heatmap.png",
    "val_em_heatmap.png",
    "val_token_acc_heatmap.png",
    "loss_heatmap.png",
    "accuracy_vs_k.png",
    "summary.txt",
    "predictions",  # directory: real validation predictions per (T, K)
    "examples",     # directory: ground-truth simulator trajectories per T
]
NUM_EXAMPLES_SHOWN = 4  # display only
OBSERVATION_MARGIN = 5.0  # percentage points: summary wording threshold only

CSV_FIELDS = [
    "T", "K", "parameter_count",
    "initial_loss", "final_loss",
    "train_token_accuracy", "train_exact_match",
    "val_token_accuracy", "val_exact_match",
    "best_val_exact_match", "training_steps",
    "stopped_reason", "eval_step",
    "val_mean_token_errors", "val_min_token_errors",
]


# ============================================================================
# Training
# ============================================================================


def train_cell(
    model: TesseractModel,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    tc: CATrainingConfig,
) -> Tuple[Dict[str, Any], Optional[EvalResult]]:
    """Train one (T, K) cell.

    Mini-batches are sampled with replacement from the global CPU torch RNG
    (identical index stream on CPU and GPU). ``final_loss`` is the last
    mini-batch loss. Train/val metrics are full-set eval-mode measurements
    at the most recent evaluation point (``eval_step``). Evaluation points are:
    steps 1..eval_first_steps, every eval_every steps, max_steps and the
    early-stop step. ``best_val_exact_match`` is the maximum over those points.
    """
    optimizer = build_optimizer(model, tc.optimizer)
    n_train = train_inputs.shape[0]

    initial_loss: Optional[float] = None
    loss_val = float("nan")
    train_eval: Optional[EvalResult] = None
    val_eval: Optional[EvalResult] = None
    eval_step: Optional[int] = None
    best_val_em: Optional[float] = None
    stopped_reason = "max_steps"

    for step in range(1, tc.max_steps + 1):
        if n_train > tc.batch_size:
            idx = torch.randint(0, n_train, (tc.batch_size,)).to(train_inputs.device)
            batch_inp, batch_tgt = train_inputs[idx], train_targets[idx]
        else:
            batch_inp, batch_tgt = train_inputs, train_targets

        loss_val = train_step(model, batch_inp, batch_tgt, optimizer)["loss"]
        if initial_loss is None:
            initial_loss = loss_val

        finite = math.isfinite(loss_val)
        early_stop = loss_val < tc.early_stop_loss
        if finite and (
            step <= tc.eval_first_steps or step % tc.eval_every == 0 or step == tc.max_steps or early_stop
        ):
            train_eval = evaluate(model, train_inputs, train_targets)
            val_eval = evaluate(model, val_inputs, val_targets)
            eval_step = step
            best_val_em = max(val_eval.exact_match_accuracy, best_val_em or 0.0)
            print(
                f"      step {step:>4d} | loss={loss_val:.6f} | "
                f"tok={train_eval.token_accuracy:.1f}% em={train_eval.exact_match_accuracy:.1f}% | "
                f"val_tok={val_eval.token_accuracy:.1f}% val_em={val_eval.exact_match_accuracy:.1f}%"
            )

        if early_stop:
            stopped_reason = "early_stop"
            print(f"      *** Early stop at step {step}: loss={loss_val:.6f} ***")
            break
        if not finite:
            stopped_reason = "non_finite_loss"
            print(f"      *** FAILURE: non-finite loss at step {step} ***")
            break

    val_errors = token_errors_per_sequence(val_eval.predictions, val_targets).float() if val_eval else None
    metrics = {
        "initial_loss": initial_loss,
        "final_loss": loss_val,
        "train_token_accuracy": train_eval.token_accuracy if train_eval else None,
        "train_exact_match": train_eval.exact_match_accuracy if train_eval else None,
        "val_token_accuracy": val_eval.token_accuracy if val_eval else None,
        "val_exact_match": val_eval.exact_match_accuracy if val_eval else None,
        "best_val_exact_match": best_val_em,
        "training_steps": step,
        "stopped_reason": stopped_reason,
        "eval_step": eval_step,
        "val_mean_token_errors": val_errors.mean().item() if val_errors is not None else None,
        "val_min_token_errors": int(val_errors.min().item()) if val_errors is not None else None,
    }
    return metrics, val_eval


# ============================================================================
# Experiment
# ============================================================================


def make_datasets(config: CellularAutomatonConfig, t: int) -> Tuple[CellularAutomatonDataset, CellularAutomatonDataset]:
    d = config.data
    seed = config.experiment.seed
    train_ds = CellularAutomatonDataset(d.num_train, d.seq_len, d.rule_number, t, seed=seed)
    val_ds = CellularAutomatonDataset(d.num_val, d.seq_len, d.rule_number, t, seed=seed + d.val_seed_offset)

    train_set = {tuple(row) for row in train_ds.inputs.tolist()}
    overlap = sum(tuple(row) in train_set for row in val_ds.inputs.tolist())
    if overlap:
        raise RuntimeError(f"T={t}: {overlap} validation initial states also appear in the training set")
    return train_ds, val_ds


def two_cell_xor_identity(dataset: CellularAutomatonDataset) -> bool:
    """True if every target equals x[i-T] XOR x[i+T] (periodic)."""
    t = dataset.steps
    x = dataset.inputs
    return torch.equal(dataset.targets, torch.roll(x, t, dims=1) ^ torch.roll(x, -t, dims=1))


def run_experiment(config: CellularAutomatonConfig, device: torch.device, run: RunRecorder) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    d, tc, seed = config.data, config.training, config.experiment.seed
    print("\n" + "=" * 64)
    print("  CELLULAR AUTOMATON EXPERIMENT")
    print(f"  Task: predict Rule {d.rule_number} after T steps")
    print("=" * 64)
    print(f"  Seed:       {seed}")
    print(f"  Device:     {device}")
    print(f"  Seq len:    {d.seq_len}")
    print(f"  Train/Val:  {d.num_train}/{d.num_val} (val seed {seed + d.val_seed_offset})")
    print(f"  T values:   {list(d.t_values)}")
    print(f"  K values:   {list(config.k_values)}")

    set_seed(seed)
    base_model = TesseractModel.from_config(config.model, 1).to(device)
    base_state = copy.deepcopy(base_model.state_dict())
    param_count = count_parameters(base_model)["trainable"]
    del base_model

    print("\n  --- Parameter Invariance Check ---")
    counts = {k: count_parameters(TesseractModel.from_config(config.model, k))["trainable"] for k in config.k_values}
    for k, c in counts.items():
        print(f"    K={k} → {c:,} parameters")
    if set(counts.values()) != {param_count}:
        raise RuntimeError(f"PARAMETER INVARIANCE VIOLATED: {counts}")

    examples_dir = run.path("examples")
    predictions_dir = run.path("predictions")
    examples_dir.mkdir()
    predictions_dir.mkdir()

    rows: List[Dict[str, Any]] = []
    identity: Dict[int, bool] = {}
    for t in d.t_values:
        print(f"\n{'=' * 64}\n  T = {t}\n{'=' * 64}")
        set_seed(seed)
        train_ds, val_ds = make_datasets(config, t)
        if d.rule_number == 90:
            identity[t] = two_cell_xor_identity(train_ds) and two_cell_xor_identity(val_ds)
        save_ground_truth_examples(train_ds, examples_dir / f"examples_T{t}.txt")

        train_inputs, train_targets = train_ds.inputs.to(device), train_ds.targets.to(device)
        val_inputs, val_targets = val_ds.inputs.to(device), val_ds.targets.to(device)
        print(f"\n  Example (T={t}):\n    Input:  {train_ds.inputs[0].tolist()}\n    Target: {train_ds.targets[0].tolist()}")

        for k in config.k_values:
            print(f"\n    --- T={t}, K={k} ---")
            # Seed before building so every cell starts from the same RNG state
            # (and therefore the same mini-batch index stream).
            set_seed(seed)
            model = TesseractModel.from_config(config.model, k).to(device)
            model.load_state_dict(base_state, strict=True)

            metrics, val_eval = train_cell(model, train_inputs, train_targets, val_inputs, val_targets, tc)
            rows.append({"T": t, "K": k, "parameter_count": param_count, **metrics})
            if val_eval is not None:
                save_predictions(val_ds, val_eval.predictions.cpu(), predictions_dir / f"T{t}_K{k}.txt", k)

            print(
                f"    Result: loss={metrics['final_loss']:.6f} "
                f"train_em={_fmt_pct(metrics['train_exact_match'])} val_em={_fmt_pct(metrics['val_exact_match'])} "
                f"val_tok={_fmt_pct(metrics['val_token_accuracy'])} ({metrics['stopped_reason']})"
            )
            del model

    facts = {"parameter_count": param_count, "train_val_overlap": 0, "two_cell_xor_identity": identity}
    return rows, facts


# ============================================================================
# Artifacts
# ============================================================================


def _fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1f}%"


def _bits(values: List[int]) -> str:
    return "".join(str(v) for v in values)


def save_ground_truth_examples(dataset: CellularAutomatonDataset, path: Path) -> None:
    lines = [
        f"Ground-truth simulator trajectories — Rule {dataset.rule_number}, T={dataset.steps}",
        "(training-set examples; these are NOT model predictions)",
        "=" * 50,
    ]
    for i in range(min(NUM_EXAMPLES_SHOWN, len(dataset))):
        inp, _ = dataset[i]
        lines.append(f"\nExample {i + 1}:")
        for step_idx, state in enumerate(simulate_trajectory(inp.tolist(), dataset.rule_number, dataset.steps)):
            label = f"t={step_idx}" + (" (target)" if step_idx == dataset.steps else "")
            lines.append(f"  {label:<14} {_bits(state)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_predictions(dataset: CellularAutomatonDataset, predictions: torch.Tensor, path: Path, k: int) -> None:
    lines = [
        f"Validation predictions — Rule {dataset.rule_number}, T={dataset.steps}, K={k}",
        "(model at the final evaluation point; '^' marks wrong tokens)",
        "=" * 50,
    ]
    for i in range(min(NUM_EXAMPLES_SHOWN, len(dataset))):
        inp, tgt = dataset[i]
        pred = predictions[i]
        wrong = (pred != tgt).tolist()
        lines.append(f"\nExample {i + 1}: {sum(wrong)} wrong token(s)")
        lines.append(f"  input:      {_bits(inp.tolist())}")
        lines.append(f"  target:     {_bits(tgt.tolist())}")
        lines.append(f"  prediction: {_bits(pred.tolist())}")
        lines.append(f"  errors:     {''.join('^' if w else ' ' for w in wrong)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved: {path}")


def _matrix(rows: List[Dict[str, Any]], key: str) -> Tuple[List[int], List[int], Dict[Tuple[int, int], Optional[float]]]:
    t_vals = sorted({r["T"] for r in rows})
    k_vals = sorted({r["K"] for r in rows})
    return t_vals, k_vals, {(r["T"], r["K"]): r[key] for r in rows}


def plot_heatmap(rows, key: str, title: str, path: Path, cmap: str, vmin: float, vmax: float, fmt: str = ".1f") -> None:
    t_vals, k_vals, matrix = _matrix(rows, key)
    data = [[matrix.get((t, k)) if matrix.get((t, k)) is not None else float("nan") for k in k_vals] for t in t_vals]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([str(k) for k in k_vals], fontsize=12)
    ax.set_yticks(range(len(t_vals)))
    ax.set_yticklabels([str(t) for t in t_vals], fontsize=12)
    ax.set_xlabel("Recursive Depth K", fontsize=13, fontweight="bold")
    ax.set_ylabel("Transformation Depth T", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    midpoint = (vmin + vmax) / 2
    for i, t in enumerate(t_vals):
        for j, k in enumerate(k_vals):
            val = matrix.get((t, k))
            if val is None or not math.isfinite(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=11, color="gray")
                continue
            dark = val > midpoint if cmap.endswith("_r") else val < midpoint
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if dark else "black")
            if t == k:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2.5, edgecolor="blue",
                                           facecolor="none", linestyle="--"))

    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_accuracy_vs_k(rows: List[Dict[str, Any]], path: Path, rule_number: int) -> None:
    t_vals, k_vals, train = _matrix(rows, "train_exact_match")
    _, _, val = _matrix(rows, "val_exact_match")
    colors = ["#E74C3C", "#F39C12", "#27AE60", "#3498DB", "#9B59B6", "#1ABC9C"]

    def series(matrix, t):
        return [matrix.get((t, k)) if matrix.get((t, k)) is not None else float("nan") for k in k_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for idx, t in enumerate(t_vals):
        color = colors[idx % len(colors)]
        ax1.plot(k_vals, series(train, t), "o-", color=color, linewidth=2, markersize=8, label=f"T={t}")
        ax2.plot(k_vals, series(val, t), "s--", color=color, linewidth=2, markersize=8, label=f"T={t}")
    for ax, title in ((ax1, "Train Exact Match (%)"), (ax2, "Val Exact Match (%)")):
        ax.set_xlabel("Recursive Depth K", fontsize=12, fontweight="bold")
        ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(k_vals)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
    fig.suptitle(f"Tesseract: K vs T on Cellular Automaton (Rule {rule_number})", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


def build_summary(config: CellularAutomatonConfig, device: torch.device, rows: List[Dict[str, Any]], facts: Dict[str, Any]) -> str:
    d = config.data
    t_vals, k_vals, _ = _matrix(rows, "T")
    header = "  T\\K  | " + " | ".join(f" K={k:<3}" for k in k_vals) + " |"
    sep = "  " + "-" * (len(header) - 2)

    def table(key: str, title: str, fmt: str) -> List[str]:
        _, _, matrix = _matrix(rows, key)
        out = ["", f"  {title}:", sep, header, sep]
        for t in t_vals:
            cells = []
            for k in k_vals:
                v = matrix.get((t, k))
                cells.append(f"{'N/A':>6}" if v is None else f"{v:>6{fmt}}")
            out.append(f"  T={t:<3} | " + " | ".join(cells) + " |")
        out.append(sep)
        return out

    lines = [
        "=" * 64,
        "  TESSERACT CELLULAR AUTOMATON EXPERIMENT — SUMMARY",
        "=" * 64,
        "",
        "  Research question:",
        "    When the task requires T sequential rule applications,",
        "    does increasing recursive depth K improve performance?",
        "",
        f"  Rule:          {d.rule_number}",
        f"  Seq length:    {d.seq_len}",
        f"  Train/Val:     {d.num_train}/{d.num_val} (overlap: {facts['train_val_overlap']})",
        f"  Parameters:    {facts['parameter_count']:,} (constant across K)",
        f"  Device:        {device}",
    ]
    lines += table("train_exact_match", "Train Exact Match Accuracy (%)", ".1f")
    lines += table("val_exact_match", "Val Exact Match Accuracy (%)", ".1f")
    lines += table("val_token_accuracy", "Val Token Accuracy (%)  [chance ≈ 50]", ".1f")
    lines += table("val_min_token_errors", f"Val: fewest wrong tokens in any sequence (of {d.seq_len})", ".0f")

    lines += ["", "  Observations (validation exact match):"]
    for t in t_vals:
        ems = [(r["K"], r["val_exact_match"]) for r in rows if r["T"] == t and r["val_exact_match"] is not None]
        if not ems:
            lines.append(f"    T={t}: no evaluated cells")
            continue
        best_k, best = max(ems, key=lambda e: e[1])
        worst_k, worst = min(ems, key=lambda e: e[1])
        if best > worst + OBSERVATION_MARGIN:
            lines.append(f"    T={t}: K={best_k} ({best:.1f}%) outperforms K={worst_k} ({worst:.1f}%)")
        else:
            lines.append(f"    T={t}: no K differs by more than {OBSERVATION_MARGIN:.0f} points (range {worst:.1f}%–{best:.1f}%)")

    if facts["two_cell_xor_identity"]:
        lines += ["", "  Task-structure check (Rule 90): target_i == x[i-T] XOR x[i+T] on all train+val data?"]
        for t, holds in facts["two_cell_xor_identity"].items():
            lines.append(f"    T={t}: {'yes' if holds else 'no'}")
        lines.append("    Where 'yes', the target depends on exactly two input cells regardless of T.")

    stops = sorted({r["stopped_reason"] for r in rows})
    lines += [
        "",
        f"  Stop reasons: {', '.join(stops)}. final_loss is the last mini-batch loss; train/val",
        "  metrics are full-set eval-mode measurements at each cell's last evaluation step.",
        "",
        "=" * 64,
    ]
    return "\n".join(lines)


def run_cellular_automaton(config: CellularAutomatonConfig, config_path: Path, device: torch.device, run_dir: Path) -> Dict[str, Any]:
    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, ARTIFACTS) as run:
        rows, facts = run_experiment(config, device, run)

        print("\n" + "=" * 64)
        print("  SAVING ARTIFACTS")
        print("=" * 64)
        save_results_csv(rows, run.path("results.csv"))
        rule = config.data.rule_number
        plot_heatmap(rows, "train_exact_match", f"Train Exact Match (%) — Rule {rule} CA", run.path("train_em_heatmap.png"), "RdYlGn", 0, 100)
        plot_heatmap(rows, "val_exact_match", f"Val Exact Match (%) — Rule {rule} CA", run.path("val_em_heatmap.png"), "RdYlGn", 0, 100)
        plot_heatmap(rows, "val_token_accuracy", f"Val Token Accuracy (%) — Rule {rule} CA", run.path("val_token_acc_heatmap.png"), "RdYlGn", 0, 100)
        plot_heatmap(rows, "final_loss", f"Final Mini-batch Loss — Rule {rule} CA", run.path("loss_heatmap.png"), "RdYlGn_r", 0, 1.0, fmt=".3f")
        plot_accuracy_vs_k(rows, run.path("accuracy_vs_k.png"), rule)

        summary_text = build_summary(config, device, rows, facts)
        print(summary_text)
        run.path("summary.txt").write_text(summary_text + "\n", encoding="utf-8")

        non_finite = [(r["T"], r["K"]) for r in rows if r["stopped_reason"] == "non_finite_loss"]
        verdict = "FAIL" if non_finite else "COMPLETED"
        results = {
            **facts,
            "cells": len(rows),
            "max_train_exact_match": max((r["train_exact_match"] or 0.0) for r in rows),
            "max_val_exact_match": max((r["val_exact_match"] or 0.0) for r in rows),
            "max_val_token_accuracy": max((r["val_token_accuracy"] or 0.0) for r in rows),
            "non_finite_cells": non_finite,
            "verdict": verdict,
        }
        run.finish(verdict=verdict, results=results)

    print(f"\n  Cellular automaton experiment: {verdict}")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup = parse_setup(argv, "Tesseract cellular automaton experiment", "cellular_automaton", CellularAutomatonConfig)
    results = run_cellular_automaton(setup.config, setup.config_path, setup.device, setup.run_dir)
    return 0 if results["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
