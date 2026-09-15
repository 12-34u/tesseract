"""Tesseract K-scaling experiment.

Central architectural property under test:

    Increasing recursive depth K increases computation while the number of
    trainable parameters stays constant.

Part A (structural, primary):
    For every K: build the same architecture, measure trainable parameters,
    count actual shared-block executions with a forward hook, check the
    state_dict layout is identical, and benchmark forward-pass inference
    latency (eval mode, no_grad, warmup, CUDA synchronisation).

Part B (optional diagnostic):
    Train every K on the copy task from identical initial weights. The copy
    task needs no iteration, and initial loss grows with K, so differences
    here are NOT evidence that recursion improves reasoning.

Usage:
    python experiments/k_scaling.py [--config PATH] [--device auto|cpu|cuda] [--output-dir DIR]
"""

import copy
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.toy_copy import CopyDataset
from experiments.common import parse_setup
from models.block import TransformerBlock
from models.tesseract import TesseractModel
from training.trainer import build_optimizer, train_full_batch
from utils.config import KScalingConfig
from utils.param_count import count_parameters
from utils.run_artifacts import RunRecorder
from utils.seed import set_seed

ARTIFACTS = ["results.csv", "parameter_count_vs_k.png", "latency_vs_k.png", "summary.txt"]


# ============================================================================
# Part A: Structural K-scaling test
# ============================================================================


def count_shared_block_calls(model: TesseractModel, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Run one eval-mode no_grad forward and count shared-block executions."""
    calls = 0

    def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
        nonlocal calls
        calls += 1

    handle = model.reasoner.shared_block.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, return_states=False)
    finally:
        handle.remove()
    return logits, calls


def measure_forward_latency(
    model: TesseractModel, x: torch.Tensor, warmup: int, runs: int
) -> Dict[str, float]:
    """Forward-pass (inference) latency in ms: eval mode, no_grad, warmup, CUDA sync."""
    model.eval()
    device = x.device

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x, return_states=False)
            sync()
        for _ in range(runs):
            sync()
            start = time.perf_counter()
            model(x, return_states=False)
            sync()
            latencies.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.pstdev(latencies),
    }


def measure_peak_gpu_memory_mb(model: TesseractModel, x: torch.Tensor) -> float:
    """Peak CUDA memory (MB) allocated during one inference forward pass."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(x.device)
    with torch.no_grad():
        model(x, return_states=False)
        torch.cuda.synchronize(x.device)
    return torch.cuda.max_memory_allocated(x.device) / (1024 * 1024)


def run_part_a(config: KScalingConfig, device: torch.device) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns per-K rows and verified structural facts. Raises on any violated invariant."""
    print("\n" + "=" * 60)
    print("  PART A: STRUCTURAL K-SCALING TEST")
    print("=" * 60)

    seed = config.experiment.seed
    bench = config.benchmark
    vocab = config.model.vocab_size

    set_seed(seed)
    x = torch.randint(0, vocab, (bench.batch_size, bench.seq_len), dtype=torch.long).to(device)

    # Global warmup so the first K is not penalised by cold caches.
    warmup_model = TesseractModel.from_config(config.model, bench.global_warmup_k).to(device)
    warmup_model.eval()
    with torch.no_grad():
        for _ in range(bench.global_warmup_runs):
            warmup_model(x, return_states=False)
    del warmup_model

    rows: List[Dict[str, Any]] = []
    layouts: Dict[int, Dict[str, tuple]] = {}
    block_instances: Dict[int, int] = {}

    for k in config.k_values:
        print(f"\n  --- K = {k} ---")
        set_seed(seed)
        model = TesseractModel.from_config(config.model, k).to(device)

        param_count = count_parameters(model)["trainable"]
        logits, calls = count_shared_block_calls(model, x)
        expected_shape = (bench.batch_size, bench.seq_len, vocab)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f"K={k}: logits shape {tuple(logits.shape)} != expected {expected_shape}")
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"K={k}: logits contain NaN/Inf")

        layouts[k] = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        block_instances[k] = sum(isinstance(m, TransformerBlock) for m in model.modules())

        latency = measure_forward_latency(model, x, bench.warmup_runs, bench.timing_runs)
        print(f"    Trainable parameters:   {param_count:,}")
        print(f"    Shared-block calls:     {calls} (measured)")
        print(f"    Block instances:        {block_instances[k]}")
        print(f"    Output shape / finite:  {list(logits.shape)} / ✓")
        print(
            f"    Latency:                mean={latency['mean_ms']:.2f}ms  "
            f"median={latency['median_ms']:.2f}ms  std={latency['std_ms']:.2f}ms"
        )

        row: Dict[str, Any] = {
            "K": k,
            "parameter_count": param_count,
            "recursive_calls": calls,
            "latency_mean_ms": latency["mean_ms"],
            "latency_median_ms": latency["median_ms"],
            "latency_std_ms": latency["std_ms"],
        }
        if device.type == "cuda":
            row["gpu_memory_mb"] = measure_peak_gpu_memory_mb(model, x)
        rows.append(row)

    # Invariants are checked with explicit raises (asserts vanish under python -O).
    counts = {r["K"]: r["parameter_count"] for r in rows}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"PARAMETER INVARIANCE VIOLATED: {counts}")
    wrong_calls = {r["K"]: r["recursive_calls"] for r in rows if r["recursive_calls"] != r["K"]}
    if wrong_calls:
        raise RuntimeError(f"Shared block call count != K: {wrong_calls}")
    reference_layout = layouts[config.k_values[0]]
    if any(layout != reference_layout for layout in layouts.values()):
        raise RuntimeError("state_dict layout (parameter names/shapes) differs across K")
    if set(block_instances.values()) != {1}:
        raise RuntimeError(f"Expected exactly one TransformerBlock per model, got {block_instances}")

    structure = {
        "parameter_count": rows[0]["parameter_count"],
        "transformer_block_instances": 1,
        "state_dict_tensors": len(reference_layout),
        "recursive_calls_equal_k": True,
    }
    print(f"\n    ✓ Parameter count identical for K={list(config.k_values)}: {structure['parameter_count']:,}")
    print("    ✓ Measured shared-block calls equal K for every model")
    print(f"    ✓ Identical state_dict layout ({structure['state_dict_tensors']} tensors), 1 TransformerBlock")
    return rows, structure


# ============================================================================
# Part B: Training comparison (diagnostic)
# ============================================================================


def run_part_b(config: KScalingConfig, device: torch.device, rows: List[Dict[str, Any]]) -> None:
    """Train each K from the same initial weights; adds training columns to ``rows``."""
    print("\n" + "=" * 60)
    print("  PART B: TRAINING COMPARISON (CONTROLLED)")
    print("=" * 60)

    seed = config.experiment.seed
    tr = config.training

    set_seed(seed)
    dataset = CopyDataset(
        num_examples=tr.data.num_examples,
        seq_len=tr.data.seq_len,
        vocab_size=config.model.vocab_size,
        task=tr.data.task,
        seed=seed,
    )
    inputs = dataset.inputs.to(device)
    targets = dataset.targets.to(device)

    # Parameters do not depend on K, so a K=1 state_dict initialises every K identically.
    set_seed(seed)
    base_state = copy.deepcopy(TesseractModel.from_config(config.model, 1).to(device).state_dict())

    for row in rows:
        k = row["K"]
        print(f"\n  --- Training K = {k} ---")
        set_seed(seed)
        model = TesseractModel.from_config(config.model, k).to(device)
        model.load_state_dict(base_state, strict=True)
        optimizer = build_optimizer(model, tr.optimizer)

        def on_step(step: int, metrics: Dict[str, Any]) -> None:
            if step <= 5 or step % tr.log_every == 0 or step == tr.max_steps:
                print(
                    f"    step {step:>4d} | loss={metrics['loss']:.6f} | "
                    f"tok_acc={metrics['token_accuracy']:.1f}% | em_acc={metrics['exact_match_accuracy']:.1f}%"
                )

        result = train_full_batch(
            model, inputs, targets, optimizer,
            max_steps=tr.max_steps, early_stop_loss=tr.early_stop_loss, on_step=on_step,
        )
        row.update(
            initial_loss=result.initial_loss,
            final_loss=result.final_loss,
            token_accuracy=result.final_token_accuracy,
            exact_match_accuracy=result.final_exact_match_accuracy,
            training_steps=result.steps,
            stopped_reason=result.stopped_reason,
        )
        print(
            f"    Final: loss={result.final_loss:.6f}  tok={result.final_token_accuracy:.1f}%  "
            f"em={result.final_exact_match_accuracy:.1f}%  steps={result.steps} ({result.stopped_reason})"
        )


# ============================================================================
# Saving and plotting
# ============================================================================


def save_results_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = ["K", "parameter_count", "recursive_calls", "latency_mean_ms", "latency_median_ms", "latency_std_ms"]
    if "gpu_memory_mb" in rows[0]:
        fieldnames.append("gpu_memory_mb")
    if "final_loss" in rows[0]:
        fieldnames += ["initial_loss", "final_loss", "token_accuracy", "exact_match_accuracy", "training_steps", "stopped_reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results CSV: {path}")


def plot_parameter_count(rows: List[Dict[str, Any]], path: Path) -> None:
    ks = [r["K"] for r in rows]
    params = [r["parameter_count"] for r in rows]
    p = params[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, params, "o-", color="#4A90D9", linewidth=2.5, markersize=10,
            markerfacecolor="#2C5F9E", markeredgecolor="white", markeredgewidth=2,
            label=f"Trainable Parameters = {p:,}")
    ax.set_xlabel("Recursive Depth K", fontsize=13, fontweight="bold")
    ax.set_ylabel("Trainable Parameters", fontsize=13, fontweight="bold")
    ax.set_title("Tesseract: Parameter Count is Independent of K", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(ks)
    margin = max(p * 0.05, 1000)
    ax.set_ylim(p - margin, p + margin)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Parameter plot: {path}")


def plot_latency(rows: List[Dict[str, Any]], path: Path, device: torch.device) -> None:
    ks = [r["K"] for r in rows]
    means = [r["latency_mean_ms"] for r in rows]
    medians = [r["latency_median_ms"] for r in rows]
    stds = [r["latency_std_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, medians, yerr=stds, fmt="s-", color="#E74C3C", linewidth=2.5, markersize=10,
                markerfacecolor="#C0392B", markeredgecolor="white", markeredgewidth=2,
                capsize=5, capthick=2, label="Median ± σ")
    ax.plot(ks, means, "o--", color="#F39C12", linewidth=1.5, markersize=7, alpha=0.7, label="Mean")
    ax.plot(ks, [medians[0] * k / ks[0] for k in ks], ":", color="#95A5A6", linewidth=1.5, alpha=0.6,
            label=f"Linear in K (from K={ks[0]})")
    ax.set_xlabel("Recursive Depth K", fontsize=13, fontweight="bold")
    ax.set_ylabel("Forward Latency (ms)", fontsize=13, fontweight="bold")
    ax.set_title(f"Tesseract: Forward Inference Latency vs K ({device.type.upper()})",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(ks)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    if len(ks) > 1 and medians[0] > 0:
        ax.annotate(
            f"K={ks[-1]} / K={ks[0]} = {medians[-1] / medians[0]:.1f}×",
            xy=(ks[-1], medians[-1]), xytext=(0.55, 0.15), textcoords="axes fraction", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#666"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor="#E74C3C"),
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Latency plot:  {path}")


def build_summary(config: KScalingConfig, device: torch.device, rows: List[Dict[str, Any]], structure: Dict[str, Any]) -> str:
    m = config.model
    ks = [r["K"] for r in rows]
    p = structure["parameter_count"]
    medians = [r["latency_median_ms"] for r in rows]
    monotonic = all(b > a for a, b in zip(medians, medians[1:]))

    lines = [
        "=" * 60,
        "  TESSERACT K-SCALING EXPERIMENT — SUMMARY",
        "=" * 60,
        "",
        "  Research question:",
        "    Does increasing K increase computational depth while",
        "    leaving trainable parameter count unchanged?",
        "",
        f"  Parameter invariance:   HOLDS — {p:,} trainable parameters for every K in {ks}",
        f"  Shared-block calls:     measured with a forward hook; equal to K for every model",
        f"  TransformerBlock count: {structure['transformer_block_instances']} (identical state_dict layout, "
        f"{structure['state_dict_tensors']} tensors)",
        f"  Device:                 {device} ({torch.get_num_threads()} torch threads)",
        f"  Architecture:           d_model={m.d_model}, heads={m.num_heads}, d_ff={m.d_ff}, "
        f"vocab={m.vocab_size}, alpha={m.alpha}",
        f"  Benchmark:              batch={config.benchmark.batch_size}, seq_len={config.benchmark.seq_len}, "
        f"{config.benchmark.warmup_runs} warmup + {config.benchmark.timing_runs} timed forward passes (no_grad)",
        "",
        "  " + "-" * 50,
        f"  {'K':>4} | {'Parameters':>15} | {'Block Calls':>12} | {'Median (ms)':>12}",
        "  " + "-" * 50,
    ]
    for r in rows:
        lines.append(f"  {r['K']:>4} | {r['parameter_count']:>15,} | {r['recursive_calls']:>12} | {r['latency_median_ms']:>12.2f}")
    lines.append("  " + "-" * 50)
    lines.append(f"  Median latency strictly increasing with K: {'yes' if monotonic else 'NO (system noise / overhead)'}")
    lines.append("  Latency is wall-clock time on this machine, not a FLOP count.")

    if "final_loss" in rows[0]:
        lines += [
            "",
            "  Training diagnostic (copy task, identical initial weights, full batch):",
            "  " + "-" * 72,
            f"  {'K':>4} | {'Init Loss':>10} | {'Final Loss':>10} | {'Tok Acc':>8} | {'EM Acc':>8} | {'Steps':>6} | Stop",
            "  " + "-" * 72,
        ]
        for r in rows:
            lines.append(
                f"  {r['K']:>4} | {r['initial_loss']:>10.4f} | {r['final_loss']:>10.6f} | "
                f"{r['token_accuracy']:>7.1f}% | {r['exact_match_accuracy']:>7.1f}% | "
                f"{r['training_steps']:>6} | {r['stopped_reason']}"
            )
        lines += [
            "  " + "-" * 72,
            "  Final loss/accuracy are from the last step's forward pass (before its update).",
            "  Initial loss grows with K (residual-stream growth, no final LayerNorm), so",
            "  steps-to-converge is confounded; the copy task needs no iteration. These are",
            "  secondary diagnostics, not evidence that recursion improves reasoning.",
        ]

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================


def run_k_scaling(config: KScalingConfig, config_path: Path, device: torch.device, run_dir: Path) -> Dict[str, Any]:
    print("=" * 60)
    print("  TESSERACT K-SCALING EXPERIMENT")
    print("=" * 60)
    print(f"  Seed:    {config.experiment.seed}")
    print(f"  Device:  {device}")
    print(f"  K vals:  {list(config.k_values)}")

    with RunRecorder(run_dir, config.experiment.name, config, config_path, device, ARTIFACTS) as run:
        rows, structure = run_part_a(config, device)
        if config.training.enabled:
            run_part_b(config, device, rows)

        print("\n" + "=" * 60)
        print("  SAVING ARTIFACTS")
        print("=" * 60)
        save_results_csv(rows, run.path("results.csv"))
        plot_parameter_count(rows, run.path("parameter_count_vs_k.png"))
        plot_latency(rows, run.path("latency_vs_k.png"), device)
        summary_text = build_summary(config, device, rows, structure)
        print(summary_text)
        run.path("summary.txt").write_text(summary_text + "\n", encoding="utf-8")

        non_finite = [r["K"] for r in rows if r.get("stopped_reason") == "non_finite_loss"]
        verdict = "FAIL" if non_finite else "PASS"
        results = {
            **structure,
            "k_values": list(config.k_values),
            "latency_median_ms": {str(r["K"]): r["latency_median_ms"] for r in rows},
            "non_finite_training_k": non_finite,
            "verdict": verdict,
        }
        run.finish(verdict=verdict, results=results)

    print(f"\n  K-scaling experiment: {verdict}")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup = parse_setup(argv, "Tesseract K-scaling experiment", "k_scaling", KScalingConfig)
    results = run_k_scaling(setup.config, setup.config_path, setup.device, setup.run_dir)
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
