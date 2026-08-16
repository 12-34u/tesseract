"""Day 3 — Tesseract Recursive Architecture Demonstration.

Prints parameter counts for K=1,2,4,8 and z_H/z_L norms across recursive steps.
Uses actual computed values from the model.
"""

import torch
from models.tesseract import TesseractModel
from utils.param_count import count_parameters
from utils.seed import set_seed


def main():
    set_seed(42)

    # Prototype configuration
    config = dict(
        vocab_size=16,
        d_model=128,
        num_heads=4,
        d_ff=512,
        max_seq_len=64,
        alpha=0.9,
        dropout=0.0,
    )

    print("=" * 55)
    print("  Tesseract Prototype — Day 3 Recursive Architecture")
    print("=" * 55)
    print(f"  d_model:      {config['d_model']}")
    print(f"  num_heads:    {config['num_heads']}")
    print(f"  d_ff:         {config['d_ff']}")
    print(f"  vocab_size:   {config['vocab_size']}")
    print(f"  max_seq_len:  {config['max_seq_len']}")
    print(f"  alpha (EMA):  {config['alpha']}")
    print()

    # --- Parameter count across K ---
    print("-" * 55)
    print("  Weight Sharing Verification: Parameter Count vs K")
    print("-" * 55)

    for k in [1, 2, 4, 8]:
        model = TesseractModel(**config, num_recursive_steps=k)
        counts = count_parameters(model)
        print(f"  K={k:<3}  →  {counts['trainable']:>7,} trainable parameters")

    print()

    # --- State behavior with K=4 ---
    print("-" * 55)
    print("  State Behavior: z_H / z_L Norms (K=4)")
    print("-" * 55)

    set_seed(42)
    model = TesseractModel(**config, num_recursive_steps=4)
    tokens = torch.randint(0, 16, (2, 8))

    logits, state_history = model(tokens, return_states=True)

    for k, states in enumerate(state_history, start=1):
        z_H_norm = torch.norm(states["z_H"]).item()
        z_L_norm = torch.norm(states["z_L"]).item()
        diff_norm = torch.norm(states["z_H"] - states["z_L"]).item()
        print(f"  Step {k}:")
        print(f"    z_H norm:       {z_H_norm:.6f}")
        print(f"    z_L norm:       {z_L_norm:.6f}")
        print(f"    |z_H - z_L|:    {diff_norm:.6f}")

    print()

    # --- Tensor shape trace ---
    print("-" * 55)
    print("  Tensor Shape Trace (end-to-end)")
    print("-" * 55)
    print(f"  Input tokens:       {list(tokens.shape)}")
    print(f"  After embedding:    [{tokens.shape[0]}, {tokens.shape[1]}, {config['d_model']}]")
    print(f"  z_H^(0):            [{tokens.shape[0]}, {tokens.shape[1]}, {config['d_model']}]")
    print(f"  z_L^(0):            [{tokens.shape[0]}, {tokens.shape[1]}, {config['d_model']}]")
    print(f"  z_L^(K) final:      {list(state_history[-1]['z_L'].shape)}")
    print(f"  Output logits:      {list(logits.shape)}")

    print()

    # --- Forward + loss + backward sanity ---
    print("-" * 55)
    print("  Forward + Loss + Backward Sanity Check")
    print("-" * 55)

    set_seed(42)
    model = TesseractModel(**config, num_recursive_steps=4)
    tokens = torch.randint(0, 16, (2, 8))
    targets = torch.randint(0, 16, (2, 8))

    logits, _ = model(tokens)
    loss = model.compute_loss(logits, targets)
    loss.backward()

    print(f"  Loss value:         {loss.item():.6f}")
    print(f"  Loss is finite:     {torch.isfinite(loss).item()}")

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = torch.norm(param.grad).item()

    total_params_with_grad = sum(1 for v in grad_norms.values() if v > 0)
    total_params = len(grad_norms)
    print(f"  Params with grad:   {total_params_with_grad}/{total_params}")
    print(f"  All grads finite:   {all(torch.isfinite(torch.tensor(v)).item() for v in grad_norms.values())}")

    print()
    print("=" * 55)
    print("  Day 3 complete. Tesseract recursive graph verified.")
    print("=" * 55)


if __name__ == "__main__":
    main()
