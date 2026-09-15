#!/usr/bin/env python3
"""Tesseract viva demonstration.

Prints diagnostics computed live from the model (nothing is typed in):
1. End-to-end tensor shape trace
2. Weight sharing: block instances vs. measured block executions
3. Parameter invariance across K
4. BPTT gradient flow through every recursive state

Usage:
    python scripts/viva_demo.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.block import TransformerBlock
from models.tesseract import TesseractModel
from training.trainer import verify_bptt
from utils.config import load_prototype_config
from utils.param_count import count_parameters
from utils.seed import set_seed

K_VALUES = (1, 2, 4, 8)
DEMO_BATCH, DEMO_SEQ_LEN = 2, 8  # small, readable demo input


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proto = load_prototype_config()
    cfg, k = proto.model, proto.default_k
    set_seed(args.seed)
    model = TesseractModel.from_config(cfg, k)
    tokens = torch.randint(0, cfg.vocab_size, (DEMO_BATCH, DEMO_SEQ_LEN))
    targets = torch.randint(0, cfg.vocab_size, (DEMO_BATCH, DEMO_SEQ_LEN))

    print("=" * 56)
    print("  TESSERACT PROTOTYPE — VIVA DEMONSTRATION")
    print("=" * 56)

    print(f"\n--- 1. END-TO-END RECURSIVE FORWARD TRACE (K={k}) ---")
    calls = []
    handle = model.reasoner.shared_block.register_forward_hook(lambda *_: calls.append(1))
    with torch.no_grad():
        x_emb = model.embedding(tokens)
        logits, history = model(tokens, return_states=True)
    handle.remove()
    print(f"Input tokens:            {list(tokens.shape)}")
    print(f"Embedding x_emb:         {list(x_emb.shape)}")
    print(f"z_H^(0), z_L^(0):        learnable {list(model.reasoner.learnable_init_H.shape)} vectors broadcast to {list(x_emb.shape)}")
    for step, states in enumerate(history, start=1):
        print(f"Step {step}: z_H {list(states['z_H'].shape)}  z_L {list(states['z_L'].shape)}")
    print(f"Logits:                  {list(logits.shape)}")
    print(f"Loss:                    {model.compute_loss(logits, targets).item():.6f}")

    print("\n--- 2. WEIGHT SHARING ---")
    blocks = sum(isinstance(m, TransformerBlock) for m in model.modules())
    print(f"TransformerBlock instances:          {blocks}")
    print(f"Shared-block executions (measured):  {len(calls)}")
    print(f"Execution model: {blocks} block × {len(calls)} executions")

    print("\n--- 3. PARAMETER INVARIANCE ---")
    for k_i in K_VALUES:
        print(f"K={k_i} → {count_parameters(TesseractModel.from_config(cfg, k_i))['trainable']:,} trainable parameters")

    print(f"\n--- 4. BPTT GRADIENT FLOW (K={k}) ---")
    report = verify_bptt(model, tokens, targets)
    for s in report.steps:
        g_h = "absent (z_H^(K) is not consumed)" if s.grad_z_H_norm is None else f"{s.grad_z_H_norm:.6f}"
        print(f"  Step {s.step}: ||dL/dz_L|| = {s.grad_z_L_norm:.6f}   ||dL/dz_H|| = {g_h}")
    print(f"BPTT verification: {'PASS' if report.passed else 'FAIL ' + str(report.failures)}")
    print("\nThe recursive states stay connected in the computation graph, so the loss")
    print("sends gradients back through every recursive state update.")
    print("=" * 56)
    return 0 if report.passed and len(calls) == k else 1


if __name__ == "__main__":
    sys.exit(main())
