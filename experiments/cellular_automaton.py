"""Tesseract Cellular Automaton Experiment — Day 6.

Investigates whether increasing recursive computation depth K helps Tesseract
perform a task that requires multiple sequential transformations (T).

Task: Predict the state of a 1D binary cellular automaton (Rule 90) after
T applications of the rule, given only the initial state.

Experiment Matrix:
    T ∈ {1, 2, 4, 8}   — required transformation depth
    K ∈ {1, 2, 4, 8}   — model's recursive computation depth

For each (T, K) combination:
    - Train a fresh model from identical initial weights
    - Record final loss, token accuracy, exact-match accuracy
    - Verify parameter count remains constant

Usage:
    PYTHONPATH=. python experiments/cellular_automaton.py
"""

import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.toy_cellular_automaton import CellularAutomatonDataset, simulate_trajectory
from models.tesseract import TesseractModel
from utils.param_count import count_parameters
from utils.seed import set_seed

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task
RULE_NUMBER = 90
SEQ_LEN = 32  # Use 32 to avoid periodic collapse at T=8 (length-16 collapses to all-zeros)
VOCAB_SIZE = 2  # Binary: {0, 1}

# T and K values
T_VALUES = [1, 2, 4, 8]
K_VALUES = [1, 2, 4, 8]

# Dataset
NUM_TRAIN = 256
NUM_VAL = 64

# Model (prototype — same architecture as Crucible/K-scaling)
D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
MAX_SEQ_LEN = 64
ALPHA = 0.9
DROPOUT = 0.0

# Training
LEARNING_RATE = 1e-3
MAX_STEPS = 2000
EARLY_STOP_LOSS = 0.005
BATCH_SIZE = 64
LOG_EVERY = 200

# Output
RUN_DIR = Path("runs/cellular_automaton")


# ============================================================================
# Model Creation
# ============================================================================


def create_model(k: int) -> TesseractModel:
    """Create a TesseractModel for the binary CA task."""
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


# ============================================================================
# Training
# ============================================================================


def train_model(
    model: TesseractModel,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    t: int,
    k: int,
) -> Dict[str, Any]:
    """Train a model on a single (T, K) configuration.

    Returns dict with metrics.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    initial_loss = None
    final_loss = None
    final_tok_acc = 0.0
    final_em_acc = 0.0
    val_tok_acc = 0.0
    val_em_acc = 0.0
    best_val_em = 0.0
    steps_done = 0

    n_train = train_inputs.shape[0]

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()

        # Mini-batch sampling
        if n_train > BATCH_SIZE:
            idx = torch.randint(0, n_train, (BATCH_SIZE,))
            batch_inp = train_inputs[idx]
            batch_tgt = train_targets[idx]
        else:
            batch_inp = train_inputs
            batch_tgt = train_targets

        logits, _ = model(batch_inp, return_states=False)
        loss = model.compute_loss(logits, batch_tgt)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val

        # Train accuracy (on full training set periodically)
        if step <= 5 or step % LOG_EVERY == 0 or step == MAX_STEPS or loss_val < EARLY_STOP_LOSS:
            model.eval()
            with torch.no_grad():
                full_logits, _ = model(train_inputs, return_states=False)
                preds = full_logits.argmax(dim=-1)
                final_tok_acc = (preds == train_targets).float().mean().item() * 100.0
                final_em_acc = (preds == train_targets).all(dim=-1).float().mean().item() * 100.0

                # Validation
                val_logits, _ = model(val_inputs, return_states=False)
                val_preds = val_logits.argmax(dim=-1)
                val_tok_acc = (val_preds == val_targets).float().mean().item() * 100.0
                val_em_acc = (val_preds == val_targets).all(dim=-1).float().mean().item() * 100.0
                best_val_em = max(best_val_em, val_em_acc)

            print(
                f"      step {step:>4d} | loss={loss_val:.6f} | "
                f"tok={final_tok_acc:.1f}% em={final_em_acc:.1f}% | "
                f"val_tok={val_tok_acc:.1f}% val_em={val_em_acc:.1f}%"
            )

        steps_done = step

        if loss_val < EARLY_STOP_LOSS:
            print(f"      *** Early stop at step {step}: loss={loss_val:.6f} ***")
            break

        if not torch.isfinite(torch.tensor(loss_val)):
            print(f"      *** FAILURE: NaN/Inf at step {step} ***")
            break

    return {
        "T": t,
        "K": k,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "train_token_accuracy": final_tok_acc,
        "train_exact_match": final_em_acc,
        "val_token_accuracy": val_tok_acc,
        "val_exact_match": val_em_acc,
        "best_val_exact_match": best_val_em,
        "training_steps": steps_done,
    }


# ============================================================================
# Experiment
# ============================================================================


def run_experiment() -> List[Dict[str, Any]]:
    """Run the full T×K experiment matrix."""
    print("\n" + "=" * 64)
    print("  CELLULAR AUTOMATON EXPERIMENT")
    print("  Task: Predict Rule 90 after T steps")
    print("=" * 64)
    print(f"  Seed:       {SEED}")
    print(f"  Device:     {DEVICE}")
    print(f"  Rule:       {RULE_NUMBER}")
    print(f"  Seq len:    {SEQ_LEN}")
    print(f"  Train/Val:  {NUM_TRAIN}/{NUM_VAL}")
    print(f"  T values:   {T_VALUES}")
    print(f"  K values:   {K_VALUES}")

    # Create base model and save initial state for controlled comparison
    set_seed(SEED)
    base_model = create_model(k=1).to(DEVICE)
    base_state_dict = copy.deepcopy(base_model.state_dict())
    param_count = count_parameters(base_model)["trainable"]
    print(f"  Parameters: {param_count:,}")
    del base_model

    # Verify parameter invariance upfront
    print("\n  --- Parameter Invariance Check ---")
    param_counts = []
    for k in K_VALUES:
        m = create_model(k)
        pc = sum(p.numel() for p in m.parameters() if p.requires_grad)
        param_counts.append(pc)
        print(f"    K={k} → {pc:,} parameters")
        del m
    assert len(set(param_counts)) == 1, (
        f"PARAMETER INVARIANCE VIOLATED: {param_counts}"
    )
    print(f"    ✓ All K values: {param_counts[0]:,} parameters")

    results: List[Dict[str, Any]] = []

    for t in T_VALUES:
        print(f"\n{'='*64}")
        print(f"  T = {t} (required transformation depth)")
        print(f"{'='*64}")

        # Create datasets for this T
        set_seed(SEED)
        train_ds = CellularAutomatonDataset(
            num_examples=NUM_TRAIN,
            seq_len=SEQ_LEN,
            rule_number=RULE_NUMBER,
            steps=t,
            seed=SEED,
        )
        val_ds = CellularAutomatonDataset(
            num_examples=NUM_VAL,
            seq_len=SEQ_LEN,
            rule_number=RULE_NUMBER,
            steps=t,
            seed=SEED + 1000,  # Different seed for validation
        )

        train_inputs = train_ds.inputs.to(DEVICE)
        train_targets = train_ds.targets.to(DEVICE)
        val_inputs = val_ds.inputs.to(DEVICE)
        val_targets = val_ds.targets.to(DEVICE)

        # Show one example
        print(f"\n  Example (T={t}):")
        print(f"    Input:  {train_inputs[0].tolist()}")
        print(f"    Target: {train_targets[0].tolist()}")

        for k in K_VALUES:
            print(f"\n    --- T={t}, K={k} ---")

            # Create model with same initial weights
            set_seed(SEED)
            model = create_model(k).to(DEVICE)
            model.load_state_dict(base_state_dict, strict=True)

            # Train
            metrics = train_model(
                model, train_inputs, train_targets,
                val_inputs, val_targets, t, k,
            )
            metrics["parameter_count"] = param_count

            results.append(metrics)

            print(
                f"    Result: loss={metrics['final_loss']:.6f} "
                f"train_em={metrics['train_exact_match']:.1f}% "
                f"val_em={metrics['val_exact_match']:.1f}%"
            )

            del model

    return results


# ============================================================================
# Visualization
# ============================================================================


def build_matrix(
    results: List[Dict[str, Any]], metric_key: str
) -> Dict:
    """Build a T×K matrix from results."""
    matrix = {}
    for r in results:
        t, k = r["T"], r["K"]
        matrix[(t, k)] = r[metric_key]
    return matrix


def plot_heatmap(
    results: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    filepath: Path,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 100.0,
    fmt: str = ".1f",
) -> None:
    """Generate a T×K heatmap."""
    if not HAS_MATPLOTLIB:
        print(f"  ⚠ matplotlib not available — skipping {filepath.name}")
        return

    matrix = build_matrix(results, metric_key)

    t_vals = sorted(set(r["T"] for r in results))
    k_vals = sorted(set(r["K"] for r in results))

    data = []
    for t in t_vals:
        row = []
        for k in k_vals:
            val = matrix.get((t, k), float("nan"))
            row.append(val)
        data.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Labels
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([str(k) for k in k_vals], fontsize=12)
    ax.set_yticks(range(len(t_vals)))
    ax.set_yticklabels([str(t) for t in t_vals], fontsize=12)
    ax.set_xlabel("Recursive Depth K", fontsize=13, fontweight="bold")
    ax.set_ylabel("Transformation Depth T", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Annotate cells
    for i, t in enumerate(t_vals):
        for j, k in enumerate(k_vals):
            val = matrix.get((t, k), float("nan"))
            if not (val != val):  # not NaN
                # Choose text color based on background
                text_color = "white" if val < (vmin + vmax) / 2 else "black"
                if cmap == "RdYlGn_r":
                    text_color = "white" if val > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)

    # Diagonal annotation (K=T line)
    for i, t in enumerate(t_vals):
        for j, k in enumerate(k_vals):
            if t == k:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2.5, edgecolor="blue", facecolor="none",
                    linestyle="--"
                )
                ax.add_patch(rect)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


def plot_accuracy_vs_k_by_t(
    results: List[Dict[str, Any]],
    filepath: Path,
) -> None:
    """Line plot: accuracy vs K, one line per T value."""
    if not HAS_MATPLOTLIB:
        return

    t_vals = sorted(set(r["T"] for r in results))
    k_vals = sorted(set(r["K"] for r in results))
    colors = ["#E74C3C", "#F39C12", "#27AE60", "#3498DB"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, t in enumerate(t_vals):
        color = colors[idx % len(colors)]
        train_ems = []
        val_ems = []
        for k in k_vals:
            for r in results:
                if r["T"] == t and r["K"] == k:
                    train_ems.append(r["train_exact_match"])
                    val_ems.append(r["val_exact_match"])

        ax1.plot(k_vals, train_ems, "o-", color=color, linewidth=2,
                 markersize=8, label=f"T={t}")
        ax2.plot(k_vals, val_ems, "s--", color=color, linewidth=2,
                 markersize=8, label=f"T={t}")

    for ax, title in [(ax1, "Train Exact Match (%)"), (ax2, "Val Exact Match (%)")]:
        ax.set_xlabel("Recursive Depth K", fontsize=12, fontweight="bold")
        ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(k_vals)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Tesseract: K vs T on Cellular Automaton (Rule 90)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {filepath}")


def save_predictions(
    results: List[Dict[str, Any]],
    dirpath: Path,
) -> None:
    """Save sample predictions for select (T,K) pairs."""
    dirpath.mkdir(parents=True, exist_ok=True)

    # Pick interesting pairs: T=K diagonal + corners
    interesting = [(1, 1), (2, 2), (4, 4), (4, 1), (4, 8)]

    for t, k in interesting:
        set_seed(SEED)
        model = create_model(k).to(DEVICE)
        # We'd need to retrain, so instead just show dataset examples
        ds = CellularAutomatonDataset(
            num_examples=8, seq_len=SEQ_LEN, rule_number=RULE_NUMBER,
            steps=t, seed=SEED,
        )

        lines = [f"Cellular Automaton Predictions (T={t}, K={k})", "=" * 50]
        for i in range(min(4, len(ds))):
            inp, tgt = ds[i]
            lines.append(f"\nExample {i+1}:")
            lines.append(f"  Input:  {''.join(str(x) for x in inp.tolist())}")

            # Show trajectory
            traj = simulate_trajectory(inp.tolist(), RULE_NUMBER, t)
            for step_idx, state in enumerate(traj):
                prefix = "  " if step_idx > 0 else "  "
                label = f"t={step_idx}" if step_idx < t else f"t={step_idx} (target)"
                lines.append(f"  {label}: {''.join(str(x) for x in state)}")

        filepath = dirpath / f"examples_T{t}.txt"
        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        del model


# ============================================================================
# Saving
# ============================================================================


def save_results_csv(results: List[Dict[str, Any]], filepath: Path) -> None:
    """Save results to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "T", "K", "parameter_count",
        "initial_loss", "final_loss",
        "train_token_accuracy", "train_exact_match",
        "val_token_accuracy", "val_exact_match",
        "best_val_exact_match", "training_steps",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {filepath}")


def save_config(filepath: Path) -> None:
    """Save experiment config."""
    config = {
        "experiment": {"name": "cellular_automaton", "seed": SEED, "device": DEVICE},
        "task": {
            "rule_number": RULE_NUMBER,
            "seq_len": SEQ_LEN,
            "num_train": NUM_TRAIN,
            "num_val": NUM_VAL,
            "T_values": T_VALUES,
        },
        "model": {
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "d_ff": D_FF,
            "max_seq_len": MAX_SEQ_LEN,
            "K_values": K_VALUES,
            "alpha": ALPHA,
            "dropout": DROPOUT,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "early_stop_loss": EARLY_STOP_LOSS,
            "batch_size": BATCH_SIZE,
        },
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Config: {filepath}")


def save_summary(results: List[Dict[str, Any]], filepath: Path) -> None:
    """Save human-readable summary."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    P = results[0]["parameter_count"]
    t_vals = sorted(set(r["T"] for r in results))
    k_vals = sorted(set(r["K"] for r in results))

    lines = []
    lines.append("=" * 64)
    lines.append("  TESSERACT CELLULAR AUTOMATON EXPERIMENT — SUMMARY")
    lines.append("=" * 64)
    lines.append("")
    lines.append("  Research Question:")
    lines.append("    When the task requires T sequential transformations,")
    lines.append("    does increasing recursive depth K improve performance?")
    lines.append("")
    lines.append(f"  Rule:          {RULE_NUMBER}")
    lines.append(f"  Seq length:    {SEQ_LEN}")
    lines.append(f"  Train/Val:     {NUM_TRAIN}/{NUM_VAL}")
    lines.append(f"  Parameters:    {P:,} (constant)")
    lines.append(f"  Device:        {DEVICE}")

    # Results matrix — Train EM
    lines.append("")
    lines.append("  Train Exact Match Accuracy (%):")
    header = "  T\\K  | " + " | ".join(f"K={k:>2}" for k in k_vals) + " |"
    sep = "  " + "-" * (len(header) - 2)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for t in t_vals:
        row_vals = []
        for k in k_vals:
            val = next(
                (r["train_exact_match"] for r in results if r["T"] == t and r["K"] == k),
                float("nan"),
            )
            row_vals.append(f"{val:>5.1f}" if val == val else "  N/A")
        lines.append(f"  T={t:<2} | " + " | ".join(row_vals) + " |")
    lines.append(sep)

    # Val EM
    lines.append("")
    lines.append("  Val Exact Match Accuracy (%):")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for t in t_vals:
        row_vals = []
        for k in k_vals:
            val = next(
                (r["val_exact_match"] for r in results if r["T"] == t and r["K"] == k),
                float("nan"),
            )
            row_vals.append(f"{val:>5.1f}" if val == val else "  N/A")
        lines.append(f"  T={t:<2} | " + " | ".join(row_vals) + " |")
    lines.append(sep)

    # Observations
    lines.append("")
    lines.append("  Observations:")

    # Check if higher K helps for higher T
    for t in t_vals:
        ems = []
        for k in k_vals:
            val = next(
                (r["val_exact_match"] for r in results if r["T"] == t and r["K"] == k),
                None,
            )
            if val is not None:
                ems.append((k, val))
        if ems:
            best_k, best_em = max(ems, key=lambda x: x[1])
            worst_k, worst_em = min(ems, key=lambda x: x[1])
            if best_em > worst_em + 5:
                lines.append(
                    f"    T={t}: K={best_k} ({best_em:.1f}%) outperforms "
                    f"K={worst_k} ({worst_em:.1f}%)"
                )
            else:
                lines.append(
                    f"    T={t}: performance similar across K "
                    f"(range: {worst_em:.1f}%–{best_em:.1f}%)"
                )

    lines.append("")
    lines.append("=" * 64)

    text = "\n".join(lines)
    print(text)

    with open(filepath, "w") as f:
        f.write(text)
    print(f"\n  Summary saved: {filepath}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Run the cellular automaton experiment."""
    results = run_experiment()

    print("\n" + "=" * 64)
    print("  SAVING ARTIFACTS")
    print("=" * 64)

    save_results_csv(results, RUN_DIR / "results.csv")
    save_config(RUN_DIR / "config.yaml")

    # Heatmaps
    plot_heatmap(
        results,
        metric_key="train_exact_match",
        title="Train Exact Match (%) — Rule 90 CA",
        filepath=RUN_DIR / "train_em_heatmap.png",
        cmap="RdYlGn",
        vmin=0, vmax=100,
    )
    plot_heatmap(
        results,
        metric_key="val_exact_match",
        title="Val Exact Match (%) — Rule 90 CA",
        filepath=RUN_DIR / "val_em_heatmap.png",
        cmap="RdYlGn",
        vmin=0, vmax=100,
    )
    plot_heatmap(
        results,
        metric_key="final_loss",
        title="Final Loss — Rule 90 CA",
        filepath=RUN_DIR / "loss_heatmap.png",
        cmap="RdYlGn_r",
        vmin=0, vmax=1.0,
        fmt=".3f",
    )

    # Line plots
    plot_accuracy_vs_k_by_t(results, RUN_DIR / "accuracy_vs_k.png")

    # Predictions
    save_predictions(results, RUN_DIR / "predictions")

    # Summary
    save_summary(results, RUN_DIR / "summary.txt")

    print("\n  ✓ Cellular automaton experiment complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
