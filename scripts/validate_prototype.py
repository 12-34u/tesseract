#!/usr/bin/env python3
"""Tesseract Prototype Validation Script — Day 7 Milestone.

Automated health check verifying model initialization, K-scaling, weight sharing,
numerical stability, backward pass, and parameter invariance.
"""

import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn

from models.tesseract import TesseractModel
from models.block import TransformerBlock
from utils.param_count import count_parameters
from utils.seed import set_seed


def main():
    set_seed(42)

    config = dict(
        vocab_size=16,
        d_model=128,
        num_heads=4,
        d_ff=512,
        max_seq_len=64,
        alpha=0.9,
        dropout=0.0,
    )

    results = {}

    # 1. Environment Check
    try:
        cuda_avail = torch.cuda.is_available()
        results["Environment"] = "PASS"
    except Exception as e:
        print(f"Environment check failed: {e}")
        results["Environment"] = "FAIL"

    # 2. Model Check
    try:
        model_init = TesseractModel(**config, num_recursive_steps=4)
        param_counts = count_parameters(model_init)
        if param_counts["trainable"] > 0 and torch.isfinite(torch.tensor(param_counts["trainable"])):
            results["Model"] = "PASS"
        else:
            results["Model"] = "FAIL"
    except Exception as e:
        print(f"Model check failed: {e}")
        results["Model"] = "FAIL"

    # 3. Forward Pass Check
    try:
        tokens = torch.randint(0, config["vocab_size"], (4, 16))
        logits, state_history = model_init(tokens, return_states=True)
        if logits.shape == (4, 16, config["vocab_size"]):
            results["Forward"] = "PASS"
        else:
            results["Forward"] = "FAIL"
    except Exception as e:
        print(f"Forward check failed: {e}")
        results["Forward"] = "FAIL"

    # 4. K=1, 2, 4, 8 Checks
    k_params = {}
    k_results = {}
    for k in [1, 2, 4, 8]:
        try:
            m = TesseractModel(**config, num_recursive_steps=k)
            k_params[k] = count_parameters(m)["trainable"]
            out, _ = m(tokens)
            if out.shape == (4, 16, config["vocab_size"]) and not torch.isnan(out).any() and not torch.isinf(out).any():
                results[f"K={k}"] = "PASS"
            else:
                results[f"K={k}"] = "FAIL"
        except Exception as e:
            print(f"K={k} check failed: {e}")
            results[f"K={k}"] = "FAIL"

    # 5. Parameter Sharing Check
    try:
        # Verify single TransformerBlock instance
        tb_count = sum(1 for m in model_init.modules() if isinstance(m, TransformerBlock))
        if tb_count == 1:
            results["Parameter sharing"] = "PASS"
        else:
            print(f"Parameter sharing check failed: TransformerBlock count is {tb_count}, expected 1")
            results["Parameter sharing"] = "FAIL"
    except Exception as e:
        print(f"Parameter sharing check failed: {e}")
        results["Parameter sharing"] = "FAIL"

    # 6. Parameter Count Invariance Check
    try:
        p1 = k_params.get(1)
        p2 = k_params.get(2)
        p4 = k_params.get(4)
        p8 = k_params.get(8)
        if p1 == p2 == p4 == p8 and p1 is not None and p1 > 0:
            results["Parameter count"] = "PASS"
        else:
            print(f"Parameter count invariance failed: K=1:{p1}, K=2:{p2}, K=4:{p4}, K=8:{p8}")
            results["Parameter count"] = "FAIL"
    except Exception as e:
        print(f"Parameter count check failed: {e}")
        results["Parameter count"] = "FAIL"

    # 7. Numerical Stability Check
    try:
        targets = torch.randint(0, config["vocab_size"], (4, 16))
        logits, _ = model_init(tokens)
        loss = model_init.compute_loss(logits, targets)
        if not torch.isnan(logits).any() and not torch.isinf(logits).any() and torch.isfinite(loss).item():
            results["Numerical stability"] = "PASS"
        else:
            results["Numerical stability"] = "FAIL"
    except Exception as e:
        print(f"Numerical stability check failed: {e}")
        results["Numerical stability"] = "FAIL"

    # 8. Backward Pass Check
    try:
        model_init.zero_grad()
        loss.backward()
        grads_exist = True
        grads_finite = True
        grads_nonzero = False
        for param in model_init.parameters():
            if param.requires_grad:
                if param.grad is None:
                    grads_exist = False
                else:
                    if not torch.isfinite(param.grad).all():
                        grads_finite = False
                    if torch.abs(param.grad).sum() > 0:
                        grads_nonzero = True

        if grads_exist and grads_finite and grads_nonzero:
            results["Backward"] = "PASS"
        else:
            print(f"Backward check failed: exist={grads_exist}, finite={grads_finite}, nonzero={grads_nonzero}")
            results["Backward"] = "FAIL"
    except Exception as e:
        print(f"Backward check failed: {e}")
        results["Backward"] = "FAIL"

    # Print validation table
    print("========================================")
    print("TESSERACT PROTOTYPE VALIDATION")
    print("========================================")
    print(f"{'Environment':<18} {results.get('Environment', 'FAIL')}")
    print(f"{'Model':<18} {results.get('Model', 'FAIL')}")
    print(f"{'Forward':<18} {results.get('Forward', 'FAIL')}")
    print(f"{'K=1':<18} {results.get('K=1', 'FAIL')}")
    print(f"{'K=2':<18} {results.get('K=2', 'FAIL')}")
    print(f"{'K=4':<18} {results.get('K=4', 'FAIL')}")
    print(f"{'K=8':<18} {results.get('K=8', 'FAIL')}")
    print(f"{'Parameter sharing':<18} {results.get('Parameter sharing', 'FAIL')}")
    print(f"{'Parameter count':<18} {results.get('Parameter count', 'FAIL')}")
    print(f"{'Backward':<18} {results.get('Backward', 'FAIL')}")
    print(f"{'Numerical stability':<18} {results.get('Numerical stability', 'FAIL')}")
    print()

    overall = "PASS" if all(v == "PASS" for v in results.values()) else "FAIL"
    print(f"OVERALL: {overall}")
    print("========================================")

    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
