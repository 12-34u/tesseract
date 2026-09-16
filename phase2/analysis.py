"""Aggregation of Phase 2 cell results into the pre-registered statistics (§4.12, amended by §8).

Amendment 01: the primary inference about increasing computational depth uses
only the primary depth T values (4, 8). Anchor (T=1) and diagnostic (T=2)
conditions are reported, labelled with their role, and excluded from the
primary regression.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from phase2.metrics import delta, interaction_regression, k_star, mean_ci95


def _role(t: int, primary: Sequence[int], diagnostic: Sequence[int], anchor: Sequence[int]) -> str:
    if t in primary:
        return "primary_depth"
    if t in diagnostic:
        return "diagnostic"
    if t in anchor:
        return "anchor"
    return "unassigned"


def _regressions(rows_by_key: Dict[tuple, List[Dict]], allowed_t) -> Dict:
    out = {}
    for (family, group, task), rows in sorted(rows_by_key.items()):
        rows = [r for r in rows if allowed_t is None or r["t"] in allowed_t]
        if len({r["k"] for r in rows}) > 1 and len({r["t"] for r in rows}) > 1:
            out[f"{family}/{group}/{task}"] = interaction_regression(rows)
    return out


def aggregate(results: Iterable[Dict], tau: float, k_values, primary_t_values: Sequence[int],
              diagnostic_t_values: Sequence[int] = (), anchor_t_values: Sequence[int] = ()) -> Dict:
    """Test-split statistics per (family, group, task, T, depth) over model seeds.

    K*(T) and Δ(T) use the seed-mean chance-normalised test token accuracy.
    """
    by_cell: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in results:
        c = r["cell"]
        if r.get("test") is not None:
            by_cell[(c["family"], c["group"], c["task"], c["t"], c["depth"])].append(r)

    cells = []
    mean_score: Dict[tuple, Dict[int, float]] = defaultdict(dict)
    regression_rows: Dict[tuple, List[Dict]] = defaultdict(list)
    for (family, group, task, t, depth), runs in sorted(by_cell.items()):
        cn = mean_ci95(r["test"]["chance_normalised_token_accuracy"] for r in runs)
        em = mean_ci95(r["test"]["exact_match"] for r in runs)
        cells.append({"family": family, "group": group, "task": task, "t": t, "depth": depth,
                      "role": _role(t, primary_t_values, diagnostic_t_values, anchor_t_values),
                      "seeds": sorted(r["cell"]["seed"] for r in runs),
                      "test_chance_normalised_token_accuracy": cn, "test_exact_match": em})
        mean_score[(family, group, task, t)][depth] = cn["mean"]
        for r in runs:
            regression_rows[(family, group, task)].append(
                {"seed": r["cell"]["seed"], "k": depth, "t": t, "score": r["test"]["chance_normalised_token_accuracy"]})

    k_high, k_low = max(k_values), min(k_values)
    depth_stats = [
        {"family": family, "group": group, "task": task, "t": t,
         "role": _role(t, primary_t_values, diagnostic_t_values, anchor_t_values),
         "k_star": k_star(scores, tau), "delta": delta(scores, k_high, k_low), "score_by_depth": dict(sorted(scores.items()))}
        for (family, group, task, t), scores in sorted(mean_score.items())
    ]
    return {
        "tau": tau,
        "t_roles": {"primary_depth": list(primary_t_values), "diagnostic": list(diagnostic_t_values),
                    "anchor": list(anchor_t_values)},
        "cells": cells,
        "depth_statistics": depth_stats,
        "primary_depth_statistics": [s for s in depth_stats if s["role"] == "primary_depth"],
        "interaction_regressions_primary": _regressions(regression_rows, set(primary_t_values)),
        "interaction_regressions_all_t_reported_only": _regressions(regression_rows, None),
    }
