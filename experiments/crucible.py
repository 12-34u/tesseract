"""Tesseract Crucible experiment — GO / NO-GO gate.

Checks two things:
    A) The prototype can overfit a tiny copy dataset (full-batch training).
    B) Loss gradients reach every recursive state z_L^(1..K) (full BPTT).

Configuration: configs/crucible.yaml (model shape from configs/prototype_small.yaml).

Usage:
    python experiments/crucible.py [--config PATH] [--device auto|cpu|cuda] [--output-dir DIR]

The process exits with status 0 only if every pass criterion holds.
"""

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.toy_copy import CopyDataset
from evaluation.metrics import evaluate
from experiments.common import parse_setup
from models.tesseract import TesseractModel
from training.logger import ExperimentLogger
from training.trainer import (
    BPTTReport,
    build_optimizer,
    train_full_batch,
    verify_bptt,
    verify_parameter_update,
)
from utils.config import CrucibleConfig
from utils.param_count import count_parameters
from utils.run_artifacts import RunRecorder, write_json
from utils.seed import set_seed

ARTIFACTS = [
    "metrics.csv",
    "loss_curve.png",
    "gradient_norms.png",
    "accuracy_curve.png",
    "predictions.txt",
    "bptt_verification.json",
    "summary.json",
    "model.pt",
]
NUM_PREDICTIONS_SHOWN = 8  # display only: examples written to predictions.txt


def _fmt(value: Optional[float], spec: str = ".6f") -> str:
    return "None" if value is None else format(value, spec)


def print_banner(config: CrucibleConfig, device: torch.device, param_count: int) -> None:
    m, d, t = config.model, config.data, config.training
    rows = [
        ("Seed", config.experiment.seed),
        ("Device", device),
        ("Parameters", f"{param_count:,}"),
        ("K", config.k),
        ("Alpha", m.alpha),
        ("d_model", m.d_model),
        ("num_heads", m.num_heads),
        ("d_ff", m.d_ff),
        ("Batch", f"full ({d.num_examples} examples)"),
        ("Learning rate", t.optimizer.learning_rate),
        ("Weight decay", t.optimizer.weight_decay),
        ("Max steps", t.max_steps),
        ("Early stop", f"loss < {t.early_stop_loss}"),
        ("Dataset", f"{d.task} ({d.num_examples} examples, seq_len={d.seq_len})"),
        ("Vocab size", m.vocab_size),
    ]
    print("=" * 56)
    print("  TESSERACT CRUCIBLE")
    print("=" * 56)
    for label, value in rows:
        print(f"  {label + ':':<18}{value}")
    print("=" * 56)


def save_predictions(
    model: TesseractModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    path: Path,
    num_examples: int,
) -> None:
    n = min(num_examples, inputs.shape[0])
    result = evaluate(model, inputs[:n], targets[:n])
    lines = ["=" * 50, "  CRUCIBLE — Final Predictions (eval mode, after training)", "=" * 50]
    for i in range(n):
        inp, tgt, pred = inputs[i].tolist(), targets[i].tolist(), result.predictions[i].tolist()
        lines.append(f"\n  Example {i + 1} {'✓' if pred == tgt else '✗'}")
        lines.append(f"    Input:      {inp}")
        lines.append(f"    Target:     {tgt}")
        lines.append(f"    Predicted:  {pred}")
    lines.append("\n" + "=" * 50)
    text = "\n".join(lines)
    print(text)
    path.write_text(text + "\n", encoding="utf-8")


def print_bptt_report(report: BPTTReport) -> None:
    num_steps = len(report.steps)
    print(f"  Loss: {report.loss:.8f}")
    for s in report.steps:
        note = ""
        if s.step == num_steps and s.grad_z_H_norm is None:
            note = "  (z_H^(K) has no gradient: not consumed downstream — expected)"
        print(
            f"  Step {s.step}:"
            f"  ||dL/dz_L||={_fmt(s.grad_z_L_norm)}"
            f"  ||dL/dz_H||={_fmt(s.grad_z_H_norm)}"
            f"  ||z_L||={s.z_L_norm:.4f}"
            f"  ||z_H||={s.z_H_norm:.4f}{note}"
        )
    for failure in report.failures:
        print(f"  ✗ {failure}")
    print(f"  BPTT verification: {'PASS' if report.passed else 'FAIL'}")


def print_results(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 56)
    print("  CRUCIBLE RESULTS")
    print("=" * 56)
    print(f"  Initial loss:          {summary['initial_loss']:.6f}")
    print(f"  Final loss:            {summary['final_loss']:.8f}")
    print(f"  Final token accuracy:  {summary['final_token_accuracy']:.1f}%")
    print(f"  Final exact-match:     {summary['final_exact_match_accuracy']:.1f}%")
    print(f"  Training steps:        {summary['training_steps']} ({summary['stopped_reason']})")
    print(f"  Parameters:            {summary['parameter_count']:,}")
    print(f"  K:                     {summary['k']}")
    for name, ok in summary["checks"].items():
        print(f"  {name + ':':<23}{'✓' if ok else '✗'}")
    print(f"\n  CRUCIBLE: {'✓ PASS' if summary['verdict'] == 'PASS' else '✗ FAIL'}")
    print("=" * 56)


def run_crucible(
    config: CrucibleConfig, config_path: Path, device: torch.device, run_dir: Path
) -> Dict[str, Any]:
    seed = config.experiment.seed
    t = config.training

    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, ARTIFACTS) as run:
        set_seed(seed)
        dataset = CopyDataset(
            num_examples=config.data.num_examples,
            seq_len=config.data.seq_len,
            vocab_size=config.model.vocab_size,
            task=config.data.task,
            seed=seed,
        )
        inputs = dataset.inputs.to(device)
        targets = dataset.targets.to(device)

        model = TesseractModel.from_config(config.model, config.k).to(device)
        param_count = count_parameters(model)["trainable"]
        print_banner(config, device, param_count)

        print("\n--- Data Verification ---")
        print(f"  Sample input:  {dataset.inputs[0].tolist()}")
        print(f"  Sample target: {dataset.targets[0].tolist()}")

        # One-step sanity check on this instance; the real run then starts from
        # a freshly built, identically initialised model.
        print("\n--- Parameter Update Verification ---")
        params_changed, delta_norm = verify_parameter_update(
            model, inputs, targets, build_optimizer(model, t.optimizer)
        )
        print(f"  Parameters changed: {params_changed}")
        print(f"  Total delta norm:   {delta_norm:.6f}")

        set_seed(seed)
        model = TesseractModel.from_config(config.model, config.k).to(device)
        optimizer = build_optimizer(model, t.optimizer)
        logger = ExperimentLogger(run_dir, experiment_name="Tesseract Crucible")

        def instrument(step: int) -> bool:
            return step <= t.grad_instrument_first_steps or step % t.grad_instrument_every == 0

        def on_step(step: int, metrics: Dict[str, Any]) -> None:
            logger.log(metrics, step=step)
            if step <= t.grad_instrument_first_steps or step % t.log_every == 0 or step == t.max_steps:
                grad_str = ""
                if "grad_zL_step_1" in metrics:
                    grads = [_fmt(metrics[f"grad_zL_step_{i}"], ".4f") for i in range(1, config.k + 1)]
                    grad_str = f"  grad_zL=[{', '.join(grads)}]"
                print(
                    f"  step {step:>4d} | loss={metrics['loss']:.6f} | "
                    f"tok_acc={metrics['token_accuracy']:.1f}% | "
                    f"em_acc={metrics['exact_match_accuracy']:.1f}%{grad_str}"
                )

        print("\n--- Training ---")
        result = train_full_batch(
            model,
            inputs,
            targets,
            optimizer,
            max_steps=t.max_steps,
            early_stop_loss=t.early_stop_loss,
            instrument=instrument,
            on_step=on_step,
        )
        print(f"\n  *** Stopped at step {result.steps}: {result.stopped_reason} (loss={result.final_loss:.8f}) ***")

        print("\n--- Saving Artifacts ---")
        for name, writer in (
            ("metrics.csv", logger.save_csv),
            ("loss_curve.png", logger.plot_loss_curve),
            ("gradient_norms.png", logger.plot_gradient_norms),
            ("accuracy_curve.png", logger.plot_accuracy),
        ):
            print(f"  {writer(run.path(name))}")

        post_eval = evaluate(model, inputs, targets)
        print()
        save_predictions(model, inputs, targets, run.path("predictions.txt"), NUM_PREDICTIONS_SHOWN)

        print("\n--- BPTT Verification (trained model, full training set) ---")
        bptt = verify_bptt(model, inputs, targets)
        print_bptt_report(bptt)
        write_json(
            run.path("bptt_verification.json"),
            {
                "k": config.k,
                "passed": bptt.passed,
                "failures": bptt.failures,
                "loss": bptt.loss,
                "evaluated_on": "trained model after the final optimizer step; full training set; train mode",
                "steps": [asdict(s) for s in bptt.steps],
            },
        )

        torch.save(model.state_dict(), run.path("model.pt"))

        criteria = config.pass_criteria
        checks = {
            "parameters_updated": params_changed,
            "loss_finite": result.stopped_reason != "non_finite_loss",
            "loss_converged": result.final_loss < criteria.max_final_loss,
            "exact_match_reached": result.final_exact_match_accuracy >= criteria.min_exact_match,
            "bptt_gradients_ok": bptt.passed,
        }
        verdict = "PASS" if all(checks.values()) else "FAIL"
        summary = {
            "k": config.k,
            "parameter_count": param_count,
            "dataset": {
                "task": config.data.task,
                "num_examples": config.data.num_examples,
                "seq_len": config.data.seq_len,
                "vocab_size": config.model.vocab_size,
            },
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "final_token_accuracy": result.final_token_accuracy,
            "final_exact_match_accuracy": result.final_exact_match_accuracy,
            "final_metrics_note": "final_* come from the last training step's forward pass, before its optimizer update (train mode)",
            "post_training_eval": {
                "token_accuracy": post_eval.token_accuracy,
                "exact_match_accuracy": post_eval.exact_match_accuracy,
            },
            "training_steps": result.steps,
            "stopped_reason": result.stopped_reason,
            "early_stop_loss": t.early_stop_loss,
            "parameter_update_delta_norm": delta_norm,
            "pass_criteria": asdict(criteria),
            "checks": checks,
            "verdict": verdict,
        }
        write_json(run.path("summary.json"), summary)
        run.finish(verdict=verdict, results=summary)

    print_results(summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup = parse_setup(argv, "Tesseract Crucible experiment", "crucible", CrucibleConfig)
    summary = run_crucible(setup.config, setup.config_path, setup.device, setup.run_dir)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
