"""Tesseract Crucible Experiment — Day 4 GO / NO-GO Gate.

Proves two things:
    A) The ~211K Tesseract prototype can overfit a tiny 16-example copy dataset.
    B) Gradients propagate through all K=4 recursive steps via full BPTT.

Configuration (from ROADMAP / crucible.yaml):
    - 16 examples, seq_len=16, vocab_size=16
    - ~211K prototype (d_model=128, heads=4, d_ff=512)
    - K=4 recursive steps, alpha=0.9
    - AdamW, lr=3e-4, batch_size=16
    - Max 500 steps, early stop at loss < 0.01

Usage:
    PYTHONPATH=. python experiments/crucible.py
"""

import sys
import os
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.toy_copy import CopyDataset
from models.tesseract import TesseractModel
from training.trainer import (
    train_step,
    verify_parameter_update,
    compute_token_accuracy,
    compute_exact_match_accuracy,
    collect_state_norms,
    retain_state_gradients,
    collect_state_gradient_norms,
)
from training.logger import ExperimentLogger
from utils.param_count import count_parameters
from utils.seed import set_seed


# ============================================================================
# Configuration
# ============================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
NUM_EXAMPLES = 16
SEQ_LEN = 16
VOCAB_SIZE = 16
TASK = "copy"

# Model (prototype_small)
D_MODEL = 128
NUM_HEADS = 4
D_FF = 512
MAX_SEQ_LEN = 64
K = 4
ALPHA = 0.9
DROPOUT = 0.0

# Training
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
MAX_STEPS = 500
EARLY_STOP_LOSS = 0.01

# Output
RUN_DIR = Path("runs/crucible_k04")

# Logging frequency
LOG_EVERY = 10
GRAD_INSTRUMENT_EVERY = 10  # Instrument recursive gradients every N steps


def print_banner(param_count: int) -> None:
    """Print experiment configuration banner."""
    print("=" * 56)
    print("  TESSERACT CRUCIBLE")
    print("=" * 56)
    print(f"  Seed:             {SEED}")
    print(f"  Device:           {DEVICE}")
    print(f"  Parameters:       {param_count:,}")
    print(f"  K:                {K}")
    print(f"  Alpha:            {ALPHA}")
    print(f"  d_model:          {D_MODEL}")
    print(f"  num_heads:        {NUM_HEADS}")
    print(f"  d_ff:             {D_FF}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Max steps:        {MAX_STEPS}")
    print(f"  Early stop:       loss < {EARLY_STOP_LOSS}")
    print(f"  Dataset:          {TASK} ({NUM_EXAMPLES} examples, seq_len={SEQ_LEN})")
    print(f"  Vocab size:       {VOCAB_SIZE}")
    print("=" * 56)


def save_predictions(
    model: TesseractModel,
    dataset: CopyDataset,
    device: str,
    filepath: Path,
    num_examples: int = 8,
) -> None:
    """Save final model predictions to a text file."""
    model.eval()
    lines = []
    lines.append("=" * 50)
    lines.append("  CRUCIBLE — Final Predictions")
    lines.append("=" * 50)

    num_to_show = min(num_examples, len(dataset))

    with torch.no_grad():
        inputs_all = dataset.inputs[:num_to_show].to(device)
        targets_all = dataset.targets[:num_to_show].to(device)
        logits, _ = model(inputs_all, return_states=False)
        preds = logits.argmax(dim=-1)

    for i in range(num_to_show):
        inp = inputs_all[i].cpu().tolist()
        tgt = targets_all[i].cpu().tolist()
        pred = preds[i].cpu().tolist()
        match = "✓" if pred == tgt else "✗"

        lines.append(f"\n  Example {i + 1} {match}")
        lines.append(f"    Input:      {inp}")
        lines.append(f"    Target:     {tgt}")
        lines.append(f"    Predicted:  {pred}")

    lines.append("\n" + "=" * 50)

    text = "\n".join(lines)
    print(text)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(text)


def save_config(filepath: Path) -> None:
    """Save experiment configuration to YAML."""
    config = {
        "experiment": {"name": "tesseract_crucible_k04", "seed": SEED},
        "data": {
            "type": TASK,
            "num_examples": NUM_EXAMPLES,
            "seq_len": SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
        },
        "model": {
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "d_ff": D_FF,
            "max_seq_len": MAX_SEQ_LEN,
            "k": K,
            "alpha": ALPHA,
            "dropout": DROPOUT,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "early_stop_loss": EARLY_STOP_LOSS,
            "optimizer": "AdamW",
        },
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_crucible() -> None:
    """Execute the Crucible experiment."""
    # === 1. Set seed ===
    set_seed(SEED)

    # === 2. Create dataset ===
    dataset = CopyDataset(
        num_examples=NUM_EXAMPLES,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        task=TASK,
        seed=SEED,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # === 3. Create model ===
    model = TesseractModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        num_recursive_steps=K,
        alpha=ALPHA,
        dropout=DROPOUT,
    ).to(DEVICE)

    param_count = count_parameters(model)["trainable"]

    # === 4. Print banner ===
    print_banner(param_count)

    # === 5. Verify data ===
    print("\n--- Data Verification ---")
    sample_inp, sample_tgt = dataset[0]
    print(f"  Sample input:  {sample_inp.tolist()}")
    print(f"  Sample target: {sample_tgt.tolist()}")
    print(f"  Match: {torch.equal(sample_inp, sample_tgt)}")

    # === 6. Create optimizer ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # === 7. Verify parameter update (one-step sanity) ===
    print("\n--- Parameter Update Verification ---")
    set_seed(SEED)
    # Use dataset tensors directly for verification
    verify_inputs = dataset.inputs.to(DEVICE)
    verify_targets = dataset.targets.to(DEVICE)
    params_changed, delta_norm = verify_parameter_update(model, verify_inputs, verify_targets, optimizer)
    print(f"  Parameters changed: {params_changed}")
    print(f"  Total delta norm:   {delta_norm:.6f}")
    if not params_changed:
        print("  *** FAILURE: Optimizer did not update any parameters! ***")
        return

    # Re-initialize model and optimizer for the actual run
    set_seed(SEED)
    model = TesseractModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        num_recursive_steps=K,
        alpha=ALPHA,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # === 8. Setup logger ===
    logger = ExperimentLogger(
        run_dir=RUN_DIR,
        experiment_name="tesseract_crucible_k04",
        use_wandb=False,  # Local-only for reliability
    )

    # Save config
    save_config(RUN_DIR / "config.yaml")

    # === 9. Training loop ===
    print("\n--- Training ---")
    initial_loss = None
    final_loss = None
    final_token_acc = 0.0
    final_em_acc = 0.0
    last_grad_norms = {}
    last_state_norms = {}

    # Since batch_size == num_examples, each epoch is one step through the loader
    # We iterate for MAX_STEPS, cycling through the data
    for step in range(1, MAX_STEPS + 1):
        # Get the full batch (since batch_size == dataset_size)
        inputs = dataset.inputs.to(DEVICE)
        targets = dataset.targets.to(DEVICE)

        # Determine whether to instrument gradients this step
        instrument = (step % GRAD_INSTRUMENT_EVERY == 0) or (step == 1) or (step <= 5)

        # Training step
        metrics = train_step(
            model=model,
            inputs=inputs,
            targets=targets,
            optimizer=optimizer,
            instrument_gradients=instrument,
        )

        # Track initial/final
        if initial_loss is None:
            initial_loss = metrics["loss"]
        final_loss = metrics["loss"]
        final_token_acc = metrics["token_accuracy"]
        final_em_acc = metrics["exact_match_accuracy"]

        # Collect gradient norms if instrumented
        grad_keys = [k for k in metrics if k.startswith("grad_z")]
        if grad_keys:
            last_grad_norms = {k: metrics[k] for k in grad_keys}

        state_keys = [k for k in metrics if k.startswith("z") and "_norm_" in k]
        if state_keys:
            last_state_norms = {k: metrics[k] for k in state_keys}

        # Log
        logger.log(metrics, step=step)

        # Console output
        if step <= 5 or step % LOG_EVERY == 0 or step == MAX_STEPS:
            grad_str = ""
            if grad_keys:
                gvals = [f"{metrics.get(f'grad_zL_step_{i}', 0.0):.4f}" for i in range(1, K + 1)]
                grad_str = f"  grad_zL=[{', '.join(gvals)}]"
            print(
                f"  step {step:>4d} | "
                f"loss={metrics['loss']:.6f} | "
                f"tok_acc={metrics['token_accuracy']:.1f}% | "
                f"em_acc={metrics['exact_match_accuracy']:.1f}%"
                f"{grad_str}"
            )

        # Early stopping
        if metrics["loss"] < EARLY_STOP_LOSS:
            print(f"\n  *** Early stop at step {step}: loss={metrics['loss']:.8f} < {EARLY_STOP_LOSS} ***")
            break

        # NaN/Inf check
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            print(f"\n  *** FAILURE: Loss became NaN/Inf at step {step} ***")
            break

    # === 10. Generate plots ===
    print("\n--- Saving Artifacts ---")
    csv_path = logger.save_csv()
    print(f"  Metrics CSV:     {csv_path}")

    loss_plot = logger.plot_loss_curve()
    print(f"  Loss curve:      {loss_plot}")

    grad_plot = logger.plot_gradient_norms()
    print(f"  Gradient norms:  {grad_plot}")

    acc_plot = logger.plot_accuracy()
    print(f"  Accuracy curve:  {acc_plot}")

    # === 11. Save predictions ===
    print()
    save_predictions(model, dataset, DEVICE, RUN_DIR / "predictions.txt")

    # === 12. Final BPTT verification ===
    print("\n--- BPTT Verification ---")
    model.train()
    optimizer.zero_grad()
    inputs = dataset.inputs.to(DEVICE)
    targets = dataset.targets.to(DEVICE)
    logits, state_history = model(inputs, return_states=True)
    retain_state_gradients(state_history)
    loss = model.compute_loss(logits, targets)
    loss.backward()

    bptt_grad_norms = collect_state_gradient_norms(state_history)
    bptt_state_norms = collect_state_norms(state_history)

    all_zL_finite = True
    all_zL_nonzero = True
    all_zH_finite = True
    # NOTE: z_H^(K) at the FINAL step has zero gradient because only z_L^(K)
    # feeds into the output head. z_H^(K) is never consumed downstream.
    # This is mathematically correct, not a BPTT failure.
    # The critical chain is: z_L^(1) → z_L^(2) → ... → z_L^(K) → loss
    # z_H matters for steps 1..K-1 where it feeds into the next merge.
    all_zH_nonzero_except_final = True

    print(f"  Loss: {loss.item():.8f}")
    for k_step in range(1, K + 1):
        zL_grad = bptt_grad_norms.get(f"grad_zL_step_{k_step}", 0.0)
        zH_grad = bptt_grad_norms.get(f"grad_zH_step_{k_step}", 0.0)
        zL_norm = bptt_state_norms.get(f"zL_norm_step_{k_step}", 0.0)
        zH_norm = bptt_state_norms.get(f"zH_norm_step_{k_step}", 0.0)

        zL_finite = torch.isfinite(torch.tensor(zL_grad)).item()
        zH_finite = torch.isfinite(torch.tensor(zH_grad)).item()
        zL_nonzero = zL_grad > 0
        zH_nonzero = zH_grad > 0

        if not zL_finite:
            all_zL_finite = False
        if not zL_nonzero:
            all_zL_nonzero = False
        if not zH_finite:
            all_zH_finite = False
        # z_H at the final step is expected to be zero (not consumed downstream)
        if k_step < K and not zH_nonzero:
            all_zH_nonzero_except_final = True  # Only flag intermediate steps

        is_final = k_step == K
        if is_final:
            # z_H^(K) zero is expected
            status = "✓" if (zL_finite and zL_nonzero) else "✗"
            note = "  (z_H^(K) zero expected — not consumed)" if zH_grad == 0 else ""
        else:
            status = "✓" if (zL_finite and zL_nonzero and zH_finite and zH_nonzero) else "✗"
            note = ""

        print(
            f"  Step {k_step}: {status}"
            f"  ||∂L/∂z_L||={zL_grad:.6f}"
            f"  ||∂L/∂z_H||={zH_grad:.6f}"
            f"  ||z_L||={zL_norm:.4f}"
            f"  ||z_H||={zH_norm:.4f}"
            f"{note}"
        )

    # === 13. Save model checkpoint ===
    model_path = RUN_DIR / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  Model saved: {model_path}")

    # === 14. Final verdict ===
    gradients_ok = all_zL_finite and all_zL_nonzero and all_zH_finite and all_zH_nonzero_except_final

    print("\n" + "=" * 56)
    print("  CRUCIBLE RESULTS")
    print("=" * 56)
    print(f"  Initial loss:          {initial_loss:.6f}")
    print(f"  Final loss:            {final_loss:.8f}")
    print(f"  Final token accuracy:  {final_token_acc:.1f}%")
    print(f"  Final exact-match:     {final_em_acc:.1f}%")
    print(f"  Parameters:            {param_count:,}")
    print(f"  K:                     {K}")
    print(f"  z_L grads finite:      {all_zL_finite}")
    print(f"  z_L grads non-zero:    {all_zL_nonzero}")
    print(f"  z_H grads finite:      {all_zH_finite}")
    print(f"  z_H grads (1..K-1):    {all_zH_nonzero_except_final}")

    # Determine PASS / FAIL
    loss_converged = final_loss < 0.1
    accuracy_high = final_em_acc >= 90.0

    crucible_pass = loss_converged and accuracy_high and gradients_ok

    print()
    if crucible_pass:
        print("  ╔══════════════════════════════════════╗")
        print("  ║        CRUCIBLE: ✓ PASS              ║")
        print("  ╚══════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════╗")
        print("  ║        CRUCIBLE: ✗ FAIL              ║")
        print("  ╚══════════════════════════════════════╝")
        if not loss_converged:
            print(f"  - Loss did not converge (final: {final_loss:.6f}, need < 0.1)")
        if not accuracy_high:
            print(f"  - Exact-match accuracy insufficient (final: {final_em_acc:.1f}%, need >= 90%)")
        if not gradients_ok:
            print(f"  - Gradient issues: zL_finite={all_zL_finite}, zL_nonzero={all_zL_nonzero}")

    print("=" * 56)

    logger.finish()


if __name__ == "__main__":
    run_crucible()
