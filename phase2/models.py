"""Model families for Phase 2 (PHASE2_BENCHMARK_DESIGN.md §4.14).

* ``tesseract``: Phase 1 ``TesseractModel``, unchanged. One shared block
  executed K times.
* ``unrolled``: the same pipeline and z_H/z_L update rule, but step k has its
  own block. L distinct blocks, so compute matches Tesseract at K = L. It
  isolates the effect of weight sharing.
* ``param_matched``: unrolled with L blocks, and d_model/d_ff shrunk so that
  trainable parameters ≈ Tesseract's.
* ``width_scaled``: Tesseract with K = 1 and d_ff enlarged so forward compute
  ≈ Tesseract at K = depth. The compute goes into width instead of depth.

Phase 1 building blocks (embedding, TransformerBlock, TesseractModel) are
imported, not modified.
"""

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.block import TransformerBlock
from models.embeddings import TokenPositionalEmbedding
from models.tesseract import TesseractModel
from utils.config import ModelConfig
from utils.param_count import count_parameters


class UnrolledTransformer(nn.Module):
    """Tesseract's recurrence with L distinct (non-shared) blocks."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, max_seq_len: int,
                 num_layers: int, alpha: float = 0.9, dropout: float = 0.0) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.d_model = d_model
        self.alpha = alpha
        self.embedding = TokenPositionalEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.learnable_init_H = nn.Parameter(torch.zeros(d_model))
        self.learnable_init_L = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.learnable_init_H, mean=0.0, std=0.02)
        nn.init.normal_(self.learnable_init_L, mean=0.0, std=0.02)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    @classmethod
    def from_config(cls, config: ModelConfig, num_layers: int) -> "UnrolledTransformer":
        return cls(config.vocab_size, config.d_model, config.num_heads, config.d_ff, config.max_seq_len,
                   num_layers, config.alpha, config.dropout)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def forward(self, tokens: torch.Tensor, return_states: bool = False):
        x_emb = self.embedding(tokens)
        B, N, D = x_emb.shape
        z_H = self.learnable_init_H.expand(B, N, D)
        z_L = self.learnable_init_L.expand(B, N, D)
        history = [] if return_states else None
        for block in self.blocks:
            z_L = block(z_L + z_H + x_emb)
            z_H = self.alpha * z_H + (1 - self.alpha) * z_L
            if return_states:
                history.append({"z_H": z_H, "z_L": z_L})
        return self.output_head(z_L), history

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or targets.shape != logits.shape[:2]:
            raise ValueError(f"Expected logits [B, N, V] and targets [B, N]; got {list(logits.shape)}, {list(targets.shape)}")
        B, N, V = logits.shape
        return self.loss_fn(logits.reshape(B * N, V), targets.reshape(B * N))


# ============================================================================
# Size and compute accounting
# ============================================================================


def model_parameter_count(vocab_size: int, d_model: int, d_ff: int, max_seq_len: int, num_blocks: int) -> int:
    """Trainable parameters of Tesseract (num_blocks = 1) or UnrolledTransformer (num_blocks = L)."""
    d, f = d_model, d_ff
    block = 4 * d * d + 9 * d + f * (2 * d + 1)  # 2 LayerNorms, QKV + out projections, FFN
    return vocab_size * d + max_seq_len * d + num_blocks * block + 2 * d + d * vocab_size + vocab_size


def forward_flops(d_model: int, d_ff: int, vocab_size: int, n: int, block_executions: int) -> int:
    """Approximate forward FLOPs for one length-n sequence.

    Counts 2 × weights × tokens for every matrix multiply (QKV, output
    projection, FFN, head) and 2·n²·d for attention scores plus weighted sum.
    Embedding lookups, LayerNorm, biases and softmax are ignored.
    """
    d = d_model
    per_block = 2 * n * (4 * d * d + 2 * d * d_ff) + 2 * n * n * d
    return block_executions * per_block + 2 * n * d * vocab_size


def match_parameters(config: ModelConfig, num_layers: int, target: int) -> Tuple[int, int]:
    """(d_model, d_ff) for an L-layer unrolled model with ≈ ``target`` parameters.

    For each d_model (a multiple of num_heads, up to the prototype's), d_ff is
    solved exactly. The pair whose d_ff/d_model ratio is closest to the
    prototype's is chosen (larger d_model on ties).
    """
    ratio = config.d_ff / config.d_model
    best = None
    for d in range(config.num_heads, config.d_model + 1, config.num_heads):
        base = model_parameter_count(config.vocab_size, d, 0, config.max_seq_len, num_layers)
        f = round((target - base) / (num_layers * (2 * d + 1)))
        if f < 1:
            continue
        shape_error = abs(math.log(f / (ratio * d)))
        if best is None or shape_error < best[0] or (shape_error == best[0] and d > best[1]):
            best = (shape_error, d, f)
    if best is None:
        raise ValueError(f"cannot match {target} parameters with {num_layers} layers")
    return best[1], best[2]


def width_scaled_d_ff(config: ModelConfig, matched_k: int, n: int) -> int:
    """d_ff for a single executed block whose forward FLOPs match Tesseract at K = matched_k."""
    d = config.d_model
    target = forward_flops(d, config.d_ff, config.vocab_size, n, matched_k)
    fixed = 2 * n * 4 * d * d + 2 * n * n * d + 2 * n * d * config.vocab_size
    return max(1, round((target - fixed) / (4 * n * d)))


@dataclass(frozen=True)
class BuiltModel:
    model: nn.Module
    family: str
    depth: int
    block_executions: int
    d_model: int
    d_ff: int
    num_heads: int
    head_dim: int
    parameter_count: int
    forward_flops_per_sequence: int
    matched: Optional[Dict]

    def describe(self) -> Dict:
        return {k: getattr(self, k) for k in ("family", "depth", "block_executions", "d_model", "d_ff", "num_heads",
                                              "head_dim", "parameter_count", "forward_flops_per_sequence", "matched")}


def build_model(family: str, depth: int, config: ModelConfig, n: int) -> BuiltModel:
    """Instantiate a model family. Seed the RNG beforehand for reproducible initialisation."""
    vocab = config.vocab_size
    matched = None
    if family == "tesseract":
        model, cfg, executions = TesseractModel.from_config(config, depth), config, depth
    elif family == "unrolled":
        model, cfg, executions = UnrolledTransformer.from_config(config, depth), config, depth
    elif family == "param_matched":
        target = model_parameter_count(vocab, config.d_model, config.d_ff, config.max_seq_len, 1)
        d, f = match_parameters(config, depth, target)
        cfg = replace(config, d_model=d, d_ff=f)
        model, executions = UnrolledTransformer.from_config(cfg, depth), depth
        achieved = model_parameter_count(vocab, d, f, config.max_seq_len, depth)
        matched = {"target_parameter_count": target, "relative_error": (achieved - target) / target,
                   "head_dim": d // config.num_heads, "reference_head_dim": config.d_model // config.num_heads,
                   "known_confound": "attention head dimension differs from Tesseract (accepted, Amendment 02 D3)"}
    elif family == "width_scaled":
        cfg = replace(config, d_ff=width_scaled_d_ff(config, depth, n))
        model, executions = TesseractModel.from_config(cfg, 1), 1
        target = forward_flops(config.d_model, config.d_ff, vocab, n, depth)
        achieved = forward_flops(cfg.d_model, cfg.d_ff, vocab, n, 1)
        matched = {"target_forward_flops": target, "matched_tesseract_k": depth, "relative_error": (achieved - target) / target}
    else:
        raise ValueError(f"unknown model family {family!r}")

    return BuiltModel(
        model=model,
        family=family,
        depth=depth,
        block_executions=executions,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_heads=cfg.num_heads,
        head_dim=cfg.d_model // cfg.num_heads,
        parameter_count=count_parameters(model)["trainable"],
        forward_flops_per_sequence=forward_flops(cfg.d_model, cfg.d_ff, vocab, n, executions),
        matched=matched,
    )
