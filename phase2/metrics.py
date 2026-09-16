"""Phase 2 evaluation metrics and statistics (PHASE2_BENCHMARK_DESIGN.md §4.12).

Token accuracy and sequence exact match reuse Phase 1's implementations in
``evaluation/metrics.py``, unchanged.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from evaluation.metrics import exact_match_accuracy, token_accuracy


def _as_tensor(values) -> torch.Tensor:
    return values if isinstance(values, torch.Tensor) else torch.as_tensor(np.asarray(values), dtype=torch.long)


def chance_normalised(token_accuracy_percent: float, vocab_size: int) -> float:
    """(acc − 1/V) / (1 − 1/V): 0 at chance, 1 when perfect."""
    chance = 1.0 / vocab_size
    return (token_accuracy_percent / 100.0 - chance) / (1.0 - chance)


def per_position_accuracy(predictions, targets) -> List[float]:
    predictions, targets = _as_tensor(predictions), _as_tensor(targets)
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError(f"expected matching [B, N] tensors, got {list(predictions.shape)} and {list(targets.shape)}")
    return ((predictions == targets).float().mean(dim=0) * 100.0).tolist()


@dataclass(frozen=True)
class SplitMetrics:
    num_examples: int
    exact_match: float  # percent
    token_accuracy: float  # percent
    chance_normalised_token_accuracy: float  # fraction, 0 = chance
    per_position_accuracy: List[float]  # percent per cell

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_metrics(predictions, targets, vocab_size: int) -> SplitMetrics:
    predictions, targets = _as_tensor(predictions), _as_tensor(targets)
    tok = token_accuracy(predictions, targets)
    return SplitMetrics(
        num_examples=int(targets.shape[0]),
        exact_match=exact_match_accuracy(predictions, targets),
        token_accuracy=tok,
        chance_normalised_token_accuracy=chance_normalised(tok, vocab_size),
        per_position_accuracy=per_position_accuracy(predictions, targets),
    )


# ============================================================================
# Depth statistics
# ============================================================================


def k_star(score_by_k: Mapping[int, Optional[float]], tau: float) -> Optional[int]:
    """Smallest K whose score reaches tau; None if no K does."""
    for k in sorted(score_by_k):
        score = score_by_k[k]
        if score is not None and score >= tau:
            return k
    return None


def delta(score_by_k: Mapping[int, Optional[float]], k_high: int, k_low: int) -> Optional[float]:
    """Δ(T) = score(K=k_high) − score(K=k_low); None if either is missing."""
    high, low = score_by_k.get(k_high), score_by_k.get(k_low)
    return None if high is None or low is None else high - low


# ============================================================================
# Uncertainty
# ============================================================================

# Two-sided 95 % Student-t critical values by degrees of freedom.
_T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
                  10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def _t_critical(df: int) -> float:
    """Two-sided 95 % critical value, rounded down to the nearest tabulated df.

    Rounding df down always picks a *larger* critical value, so an unlisted df
    is never given a narrower interval than it deserves. Only beyond df = 120
    is the normal limit used, where the difference from t is < 0.011.
    """
    if df > 120:
        return 1.96
    return _T_CRITICAL_95[max(k for k in _T_CRITICAL_95 if k <= df)]


def mean_ci95(values: Iterable[float]) -> Dict[str, Optional[float]]:
    values = [float(v) for v in values]
    if not values:
        raise ValueError("mean_ci95 needs at least one value")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "low": None, "high": None}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    half = _t_critical(len(values) - 1) * sd / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "low": mean - half, "high": mean + half}


def bootstrap_ci(per_example: np.ndarray, num_resamples: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Percentile 95 % CI of the mean over examples."""
    values = np.asarray(per_example, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(num_resamples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}


def interaction_regression(rows: Iterable[Mapping]) -> Dict:
    """OLS of score on log2K, log2T and log2K·log2T, with a fixed intercept per seed.

    Deviation from the design: the proposal specifies a seed *random* effect.
    No mixed-model library is installed, so fixed seed intercepts are used,
    and this is reported alongside the coefficient.
    """
    rows = list(rows)
    seeds = sorted({r["seed"] for r in rows})
    x, y = [], []
    for r in rows:
        lk, lt = math.log2(r["k"]), math.log2(r["t"])
        x.append([lk, lt, lk * lt] + [1.0 if r["seed"] == s else 0.0 for s in seeds])
        y.append(r["score"])
    coef, *_ = np.linalg.lstsq(np.asarray(x), np.asarray(y), rcond=None)
    return {
        "model": "score ~ log2K + log2T + log2K:log2T + seed (fixed intercepts)",
        "n": len(rows),
        "log2K": float(coef[0]),
        "log2T": float(coef[1]),
        "log2K_x_log2T": float(coef[2]),
        "note": "fixed seed intercepts used in place of the pre-registered random effect",
    }
