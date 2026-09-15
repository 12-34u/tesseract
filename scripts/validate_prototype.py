#!/usr/bin/env python3
"""Tesseract prototype health check.

Builds the prototype from configs/prototype_small.yaml and verifies, for each
Phase 1 recursion depth K: forward shape, finite outputs, a working backward
pass, the measured shared-block call count, BPTT gradient flow, parameter
invariance and weight sharing. Every check runs, even if an earlier one
failed; the script exits 1 if any check fails.

Usage:
    python scripts/validate_prototype.py [--device auto|cpu|cuda] [--seed 42]
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.block import TransformerBlock
from models.tesseract import TesseractModel
from training.trainer import verify_bptt
from utils.config import load_prototype_config
from utils.device import resolve_device
from utils.param_count import count_parameters
from utils.seed import set_seed

K_VALUES = (1, 2, 4, 8)  # Phase 1 recursion depths under validation
PROBE_BATCH, PROBE_SEQ_LEN = 4, 16  # probe input size; any valid size works


class CheckFailed(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proto = load_prototype_config()
    cfg = proto.model
    device = resolve_device(args.device)
    set_seed(args.seed)
    tokens = torch.randint(0, cfg.vocab_size, (PROBE_BATCH, PROBE_SEQ_LEN), device=device)
    targets = torch.randint(0, cfg.vocab_size, (PROBE_BATCH, PROBE_SEQ_LEN), device=device)

    models: Dict[int, TesseractModel] = {}
    for k in K_VALUES:
        set_seed(args.seed)
        models[k] = TesseractModel.from_config(cfg, k).to(device)

    def forward(k: int) -> None:
        with torch.no_grad():
            logits, _ = models[k].eval()(tokens)
        require(logits.shape == (PROBE_BATCH, PROBE_SEQ_LEN, cfg.vocab_size), f"logits shape {tuple(logits.shape)}")
        require(bool(torch.isfinite(logits).all()), "logits contain NaN/Inf")

    def block_calls(k: int) -> None:
        calls = []
        handle = models[k].reasoner.shared_block.register_forward_hook(lambda *_: calls.append(1))
        try:
            with torch.no_grad():
                models[k].eval()(tokens)
        finally:
            handle.remove()
        require(len(calls) == k, f"shared block executed {len(calls)} times, expected {k}")

    def backward(k: int) -> None:
        model = models[k].train()
        model.zero_grad(set_to_none=True)
        logits, _ = model(tokens)
        loss = model.compute_loss(logits, targets)
        require(bool(torch.isfinite(loss)), f"loss is {loss.item()}")
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        require(not missing, f"parameters without gradient: {missing}")
        require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "non-finite gradients")
        require(any(bool((p.grad != 0).any()) for p in model.parameters()), "all gradients are zero")
        model.zero_grad(set_to_none=True)

    def bptt(k: int) -> None:
        report = verify_bptt(models[k], tokens, targets)
        require(report.passed, "; ".join(report.failures))

    def invariance() -> None:
        counts = {k: count_parameters(m)["trainable"] for k, m in models.items()}
        require(len(set(counts.values())) == 1, f"parameter counts differ across K: {counts}")

    def weight_sharing() -> None:
        blocks = {k: sum(isinstance(mod, TransformerBlock) for mod in m.modules()) for k, m in models.items()}
        require(set(blocks.values()) == {1}, f"TransformerBlock instances per model: {blocks}")
        layouts = {k: [(n, tuple(p.shape)) for n, p in m.named_parameters()] for k, m in models.items()}
        require(len({tuple(v) for v in layouts.values()}) == 1, "parameter layout differs across K")

    checks: List[Tuple[str, Callable[[], None]]] = []
    for k in K_VALUES:
        checks += [
            (f"K={k} forward", lambda k=k: forward(k)),
            (f"K={k} block calls", lambda k=k: block_calls(k)),
            (f"K={k} backward", lambda k=k: backward(k)),
            (f"K={k} BPTT", lambda k=k: bptt(k)),
        ]
    checks += [("Parameter invariance", invariance), ("Weight sharing", weight_sharing)]

    results: Dict[str, str] = {}
    for name, check in checks:
        try:
            check()
            results[name] = "PASS"
        except Exception:  # report every failing check with its traceback, then keep going
            traceback.print_exc()
            results[name] = "FAIL"

    params = count_parameters(models[K_VALUES[0]])["trainable"]
    print("=" * 44)
    print("TESSERACT PROTOTYPE VALIDATION")
    print("=" * 44)
    print(f"Device: {device}   Trainable parameters: {params:,}")
    print(f"Config: d_model={cfg.d_model} heads={cfg.num_heads} d_ff={cfg.d_ff} vocab={cfg.vocab_size}")
    for name, status in results.items():
        print(f"{name:<24} {status}")
    overall = "PASS" if all(s == "PASS" for s in results.values()) else "FAIL"
    print(f"\nOVERALL: {overall}")
    print("=" * 44)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
