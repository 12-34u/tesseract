"""Tesseract K-Scaling Experiment — Day 5.

Demonstrates the central architectural property of Tesseract:

    Increasing recursive depth K increases computation while keeping
    the number of trainable parameters constant.

Part A (Structural):
    - Instantiate the same architecture at K = 1, 2, 4, 8
    - Measure trainable parameter count (programmatic)
    - Assert parameter invariance across all K
    - Measure forward-pass latency with proper warmup and synchronization
    - Generate parameter-count and latency plots

Part B (Training Comparison — optional):
    - Train each K on the tiny copy task with identical initial weights
    - Record initial/final loss, token accuracy, exact-match accuracy
    - Compare learning behavior (secondary diagnostic only)

Usage:
    PYTHONPATH=. python experiments/k_scaling.py
"""

import copy
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.toy_copy import CopyDataset
from models.tesseract import TesseractModel
from utils.param_count import count_parameters
from utils.seed import set_seed

# Try matplotlib — fall back gracefully if unavailable
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# K values to test
K_VALUES = [1, 2, 4, 8]

# Model configuration (prototype_small — must match Crucible)
VOCAB_SIZE = 16
D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
MAX_SEQ_LEN = 64
ALPHA = 0.9
DROPOUT = 0.0

# Dataset (same as Crucible)
NUM_EXAMPLES = 16
SEQ_LEN = 16
TASK = "copy"

# Benchmarking
WARMUP_RUNS = 10
TIMING_RUNS = 30
BENCHMARK_BATCH_SIZE = 16

# Training (Part B)
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 3e-4
MAX_TRAIN_STEPS = 200
EARLY_STOP_LOSS = 0.01
LOG_EVERY = 50

# Output
RUN_DIR = Path("runs/k_scaling")


# ============================================================================
# Part A: Structural K-Scaling Test
# ============================================================================


def create_model(k: int) -> TesseractModel:
    """Create a TesseractModel with the given recursive depth K."""
    return TesseractModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        num_recursive_steps=k,
        alpha=ALPHA,
        dropout=DROPOUT,
    )


def measure_parameter_count(model: TesseractModel) -> int:
    """Measure the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_forward_latency(
    model: TesseractModel,
    x: torch.Tensor,
    warmup: int = WARMUP_RUNS,
    runs: int = TIMING_RUNS,
) -> Dict[str, float]:
    """Measure forward-pass latency with warmup and proper CUDA sync.

    Returns:
        Dict with 'mean_ms', 'median_ms', 'std_ms', and 'all_ms'.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(x, return_states=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed runs
        latencies = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model(x, return_states=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000.0)  # Convert to ms

    mean_ms = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    median_ms = (sorted_lat[n // 2] + sorted_lat[(n - 1) // 2]) / 2.0
    variance = sum((t - mean_ms) ** 2 for t in latencies) / len(latencies)
    std_ms = variance**0.5

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "all_ms": latencies,
    }


def measure_gpu_memory(model: TesseractModel, x: torch.Tensor) -> float:
    """Measure peak GPU memory for a forward pass (MB). Returns 0.0 on CPU."""
    if not torch.cuda.is_available():
        return 0.0

    device = next(model.parameters()).device
    if device.type != "cuda":
        return 0.0

    model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        model(x, return_states=False)
        torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 * 1024)


def run_part_a(results: List[Dict[str, Any]]) -> None:
    """Part A: Structural K-scaling — parameter invariance and latency."""
    print("\n" + "=" * 60)
    print("  PART A: STRUCTURAL K-SCALING TEST")
    print("=" * 60)

    set_seed(SEED)

    # Create fixed input for benchmarking
    x = torch.randint(
        0, VOCAB_SIZE, (BENCHMARK_BATCH_SIZE, SEQ_LEN), dtype=torch.long
    ).to(DEVICE)

    # Global warmup: run a throwaway model to warm CPU/GPU caches
    # This prevents the first K value from being unfairly penalized
    _warmup_model = create_model(k=4).to(DEVICE)
    _warmup_model.eval()
    with torch.no_grad():
        for _ in range(5):
            _warmup_model(x, return_states=False)
    del _warmup_model

    parameter_counts = []

    for k in K_VALUES:
        print(f"\n  --- K = {k} ---")

        set_seed(SEED)
        model = create_model(k).to(DEVICE)

        # 1. Measure parameter count
        param_count = measure_parameter_count(model)
        parameter_counts.append(param_count)
        print(f"    Trainable parameters: {param_count:,}")

        # 2. Verify output shape
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, return_states=False)
        B, N, V = logits.shape
        assert B == BENCHMARK_BATCH_SIZE and N == SEQ_LEN and V == VOCAB_SIZE, (
            f"Output shape mismatch at K={k}: got ({B},{N},{V}), "
            f"expected ({BENCHMARK_BATCH_SIZE},{SEQ_LEN},{VOCAB_SIZE})"
        )
        assert torch.isfinite(logits).all(), f"Logits contain NaN/Inf at K={k}"
        print(f"    Output shape: [{B}, {N}, {V}] ✓")
        print(f"    No NaN/Inf:   ✓")

        # 3. Measure forward latency
        latency = measure_forward_latency(model, x)
        print(
            f"    Latency:      mean={latency['mean_ms']:.2f}ms  "
            f"median={latency['median_ms']:.2f}ms  "
            f"std={latency['std_ms']:.2f}ms"
        )

        # 4. Measure GPU memory (if applicable)
        gpu_mem = measure_gpu_memory(model, x)

        # 5. Store results
        result = {
            "K": k,
            "parameter_count": param_count,
            "recursive_calls": k,
            "latency_mean_ms": latency["mean_ms"],
            "latency_median_ms": latency["median_ms"],
            "latency_std_ms": latency["std_ms"],
            "gpu_memory_mb": gpu_mem,
        }
        results.append(result)

    # === CRITICAL ASSERTION: Parameter invariance ===
    print("\n  --- Parameter Invariance Assertion ---")
    unique_counts = set(parameter_counts)
    for k, pc in zip(K_VALUES, parameter_counts):
        print(f"    K={k}  →  {pc:,} trainable parameters")

    assert len(unique_counts) == 1, (
        f"PARAMETER INVARIANCE VIOLATED! "
        f"Distinct parameter counts: {unique_counts}"
    )

    P = parameter_counts[0]
    print(f"\n    ✓ All K values share identical parameter count: {P:,}")
    print(f"    ✓ Parameter invariance assertion PASSED")

    # === Latency trend ===
    print("\n  --- Latency Trend ---")
    for r in results:
        k = r["K"]
        lat = r["latency_median_ms"]
        bar = "█" * max(1, int(lat / results[0]["latency_median_ms"] * 10))
        print(f"    K={k:>2}  {lat:>8.2f}ms  {bar}")

    # Verify latency generally increases
    latencies_median = [r["latency_median_ms"] for r in results]
    if latencies_median[-1] > latencies_median[0]:
        print(f"\n    ✓ Latency increases from K=1 to K=8 "
              f"({latencies_median[0]:.2f}ms → {latencies_median[-1]:.2f}ms)")
    else:
        print(f"\n    ⚠ Latency did NOT increase monotonically — "
              f"this may be due to system noise or framework overhead")


# ============================================================================
# Part B: Training Comparison (Optional)
# ============================================================================


def run_part_b(results: List[Dict[str, Any]]) -> None:
    """Part B: Optional training comparison across K values.

    Uses the same initial weights for all K to remove initialization
    as a confounding variable.
    """
    print("\n" + "=" * 60)
    print("  PART B: TRAINING COMPARISON (CONTROLLED)")
    print("=" * 60)

    # Create dataset
    set_seed(SEED)
    dataset = CopyDataset(
        num_examples=NUM_EXAMPLES,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        task=TASK,
        seed=SEED,
    )
    inputs = dataset.inputs.to(DEVICE)
    targets = dataset.targets.to(DEVICE)

    # Create a base model (K=1 to save the shared weights) and save its state
    set_seed(SEED)
    base_model = create_model(k=1).to(DEVICE)
    base_state_dict = copy.deepcopy(base_model.state_dict())

    for i, r in enumerate(results):
        k = r["K"]
        print(f"\n  --- Training K = {k} ---")

        # Create model with this K and load the SAME initial weights
        set_seed(SEED)
        model = create_model(k).to(DEVICE)
        model.load_state_dict(base_state_dict, strict=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        initial_loss = None
        final_loss = None
        final_token_acc = 0.0
        final_em_acc = 0.0

        for step in range(1, MAX_TRAIN_STEPS + 1):
            model.train()
            optimizer.zero_grad()

            logits, _ = model(inputs, return_states=False)
            loss = model.compute_loss(logits, targets)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val

            # Compute accuracies
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                token_correct = (preds == targets).float().mean().item() * 100.0
                em_correct = (preds == targets).all(dim=-1).float().mean().item() * 100.0
                final_token_acc = token_correct
                final_em_acc = em_correct

            if step <= 5 or step % LOG_EVERY == 0 or step == MAX_TRAIN_STEPS:
                print(
                    f"    step {step:>4d} | loss={loss_val:.6f} | "
                    f"tok_acc={token_correct:.1f}% | em_acc={em_correct:.1f}%"
                )

            if loss_val < EARLY_STOP_LOSS:
                print(
                    f"    *** Early stop at step {step}: loss={loss_val:.8f} ***"
                )
                break

            if not torch.isfinite(torch.tensor(loss_val)):
                print(f"    *** FAILURE: Loss became NaN/Inf at step {step} ***")
                break

        # Update results with training metrics
        r["initial_loss"] = initial_loss
        r["final_loss"] = final_loss
        r["token_accuracy"] = final_token_acc
        r["exact_match_accuracy"] = final_em_acc
        r["training_steps"] = step

        print(
            f"    Final: loss={final_loss:.6f}  "
            f"tok={final_token_acc:.1f}%  em={final_em_acc:.1f}%  "
            f"steps={step}"
        )


# ============================================================================
# Saving and Plotting
# ============================================================================


def save_results_csv(results: List[Dict[str, Any]], filepath: Path) -> None:
    """Save experiment results to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine columns based on what's in the results
    fieldnames = ["K", "parameter_count", "recursive_calls",
                  "latency_mean_ms", "latency_median_ms", "latency_std_ms"]

    if "gpu_memory_mb" in results[0] and results[0]["gpu_memory_mb"] > 0:
        fieldnames.append("gpu_memory_mb")

    if "final_loss" in results[0]:
        fieldnames.extend([
            "initial_loss", "final_loss",
            "token_accuracy", "exact_match_accuracy",
            "training_steps",
        ])

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"  Results CSV: {filepath}")


def save_config(filepath: Path) -> None:
    """Save experiment configuration to YAML."""
    config = {
        "experiment": {"name": "k_scaling", "seed": SEED, "device": DEVICE},
        "k_values": K_VALUES,
        "model": {
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "d_ff": D_FF,
            "max_seq_len": MAX_SEQ_LEN,
            "alpha": ALPHA,
            "dropout": DROPOUT,
        },
        "benchmark": {
            "warmup_runs": WARMUP_RUNS,
            "timing_runs": TIMING_RUNS,
            "batch_size": BENCHMARK_BATCH_SIZE,
            "seq_len": SEQ_LEN,
        },
        "training": {
            "enabled": True,
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_TRAIN_STEPS,
            "early_stop_loss": EARLY_STOP_LOSS,
            "dataset": TASK,
            "num_examples": NUM_EXAMPLES,
        },
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Config YAML: {filepath}")


def plot_parameter_count(results: List[Dict[str, Any]], filepath: Path) -> None:
    """Generate parameter count vs K plot (expected: flat line)."""
    if not HAS_MATPLOTLIB:
        print(f"  ⚠ matplotlib not available — skipping {filepath.name}")
        return

    ks = [r["K"] for r in results]
    params = [r["parameter_count"] for r in results]
    P = params[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot with markers
    ax.plot(ks, params, "o-", color="#4A90D9", linewidth=2.5, markersize=10,
            markerfacecolor="#2C5F9E", markeredgecolor="white", markeredgewidth=2,
            label=f"Trainable Parameters = {P:,}")

    # Style
    ax.set_xlabel("Recursive Depth K", fontsize=13, fontweight="bold")
    ax.set_ylabel("Trainable Parameters", fontsize=13, fontweight="bold")
    ax.set_title("Tesseract: Parameter Count is Independent of K",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=11)

    # Set y-axis range to emphasize flatness
    y_center = P
    y_margin = max(P * 0.05, 1000)
    ax.set_ylim(y_center - y_margin, y_center + y_margin)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add annotation
    ax.annotate(
        f"Constant: {P:,} parameters\nacross K = {ks[0]}..{ks[-1]}",
        xy=(ks[-1], P),
        xytext=(ks[-1] - 1.5, P + y_margin * 0.6),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#666"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F0FE", edgecolor="#4A90D9"),
    )

    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Parameter plot: {filepath}")


def plot_latency(results: List[Dict[str, Any]], filepath: Path) -> None:
    """Generate latency vs K plot (expected: increasing)."""
    if not HAS_MATPLOTLIB:
        print(f"  ⚠ matplotlib not available — skipping {filepath.name}")
        return

    ks = [r["K"] for r in results]
    means = [r["latency_mean_ms"] for r in results]
    medians = [r["latency_median_ms"] for r in results]
    stds = [r["latency_std_ms"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot median with error bars
    ax.errorbar(ks, medians, yerr=stds, fmt="s-", color="#E74C3C",
                linewidth=2.5, markersize=10, markerfacecolor="#C0392B",
                markeredgecolor="white", markeredgewidth=2,
                capsize=5, capthick=2, label="Median ± σ")
    ax.plot(ks, means, "o--", color="#F39C12", linewidth=1.5, markersize=7,
            alpha=0.7, label="Mean")

    # Ideal linear reference (dashed)
    base = medians[0]
    ideal = [base * k for k in ks]
    ax.plot(ks, ideal, ":", color="#95A5A6", linewidth=1.5, alpha=0.6,
            label="Ideal linear scaling")

    ax.set_xlabel("Recursive Depth K", fontsize=13, fontweight="bold")
    ax.set_ylabel("Forward Latency (ms)", fontsize=13, fontweight="bold")
    ax.set_title("Tesseract: Forward Latency Increases with K",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Annotate speedup
    if medians[-1] > 0 and medians[0] > 0:
        ratio = medians[-1] / medians[0]
        ax.annotate(
            f"K=8 / K=1 = {ratio:.1f}×",
            xy=(ks[-1], medians[-1]),
            xytext=(ks[-1] - 2, medians[-1] * 0.85),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#666"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC",
                      edgecolor="#E74C3C"),
        )

    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Latency plot:  {filepath}")


def save_summary(results: List[Dict[str, Any]], filepath: Path) -> None:
    """Save a human-readable summary of the experiment."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    P = results[0]["parameter_count"]

    lines = []
    lines.append("=" * 60)
    lines.append("  TESSERACT K-SCALING EXPERIMENT — SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  Research Question:")
    lines.append("    Does increasing K increase computational depth while")
    lines.append("    leaving trainable parameter count unchanged?")
    lines.append("")
    lines.append("  Answer: YES")
    lines.append("")
    lines.append(f"  Constant parameter count: {P:,}")
    lines.append(f"  Device: {DEVICE}")
    lines.append(f"  Architecture: d_model={D_MODEL}, heads={NUM_HEADS}, d_ff={D_FF}")
    lines.append("")

    # Parameter table
    lines.append("  " + "-" * 50)
    lines.append(f"  {'K':>4} | {'Parameters':>15} | {'Block Calls':>12} | {'Latency (ms)':>14}")
    lines.append("  " + "-" * 50)
    for r in results:
        lines.append(
            f"  {r['K']:>4} | {r['parameter_count']:>15,} | {r['recursive_calls']:>12} | "
            f"{r['latency_median_ms']:>14.2f}"
        )
    lines.append("  " + "-" * 50)

    # Training results if available
    if "final_loss" in results[0]:
        lines.append("")
        lines.append("  Training Results (copy task, same initial weights):")
        lines.append("  " + "-" * 58)
        lines.append(
            f"  {'K':>4} | {'Init Loss':>10} | {'Final Loss':>10} | "
            f"{'Tok Acc':>8} | {'EM Acc':>8} | {'Steps':>6}"
        )
        lines.append("  " + "-" * 58)
        for r in results:
            lines.append(
                f"  {r['K']:>4} | {r['initial_loss']:>10.4f} | "
                f"{r['final_loss']:>10.6f} | "
                f"{r['token_accuracy']:>7.1f}% | "
                f"{r['exact_match_accuracy']:>7.1f}% | "
                f"{r['training_steps']:>6}"
            )
        lines.append("  " + "-" * 58)
        lines.append("")
        lines.append("  Note: Loss/accuracy differences on the copy task are")
        lines.append("  secondary diagnostics, not evidence that recursion")
        lines.append("  improves reasoning. The copy task can be solved")
        lines.append("  without iterative reasoning.")

    lines.append("")
    lines.append("  Conclusion:")
    lines.append(
        "    Tesseract decouples trainable parameter count from"
    )
    lines.append(
        f"    recursive computation depth: the same {P:,} parameter"
    )
    lines.append(
        "    model can execute the shared transformer block 1, 2, 4,"
    )
    lines.append(
        "    or 8 times, increasing computation without introducing"
    )
    lines.append("    additional trainable parameters.")
    lines.append("")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)

    with open(filepath, "w") as f:
        f.write(text)
    print(f"\n  Summary saved: {filepath}")


def print_parameter_table(results: List[Dict[str, Any]]) -> None:
    """Print the formatted parameter count vs K table."""
    print("\n  Parameter Count vs K:")
    print("  " + "-" * 52)
    print(f"  | {'K':>3} | {'Trainable Parameters':>20} | {'Recursive Block Calls':>21} |")
    print("  " + "-" * 52)
    for r in results:
        print(
            f"  | {r['K']:>3} | {r['parameter_count']:>20,} | {r['recursive_calls']:>21} |"
        )
    print("  " + "-" * 52)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the K-scaling experiment."""
    print("=" * 60)
    print("  TESSERACT K-SCALING EXPERIMENT")
    print("  Day 5: Constant Parameters, Variable Computation")
    print("=" * 60)
    print(f"  Seed:    {SEED}")
    print(f"  Device:  {DEVICE}")
    print(f"  K vals:  {K_VALUES}")
    print(f"  Config:  d_model={D_MODEL}, heads={NUM_HEADS}, d_ff={D_FF}")

    results: List[Dict[str, Any]] = []

    # Part A: Structural test (primary Day-5 result)
    run_part_a(results)

    # Print the formatted table
    print_parameter_table(results)

    # Part B: Training comparison (secondary diagnostic)
    run_part_b(results)

    # Save all artifacts
    print("\n" + "=" * 60)
    print("  SAVING ARTIFACTS")
    print("=" * 60)

    save_results_csv(results, RUN_DIR / "results.csv")
    save_config(RUN_DIR / "config.yaml")
    plot_parameter_count(results, RUN_DIR / "parameter_count_vs_k.png")
    plot_latency(results, RUN_DIR / "latency_vs_k.png")
    save_summary(results, RUN_DIR / "summary.txt")

    print("\n  ✓ K-scaling experiment complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
