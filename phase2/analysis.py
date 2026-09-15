"""Aggregation of Phase 2 cell results into the pre-registered statistics (§4.12)."""

from collections import defaultdict
from typing import Dict, Iterable, List

from phase2.metrics import delta, interaction_regression, k_star, mean_ci95


def aggregate(results: Iterable[Dict], tau: float, k_values) -> Dict:
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
                      "seeds": sorted(r["cell"]["seed"] for r in runs),
                      "test_chance_normalised_token_accuracy": cn, "test_exact_match": em})
        mean_score[(family, group, task, t)][depth] = cn["mean"]
        for r in runs:
            regression_rows[(family, group, task)].append(
                {"seed": r["cell"]["seed"], "k": depth, "t": t, "score": r["test"]["chance_normalised_token_accuracy"]})

    k_high, k_low = max(k_values), min(k_values)
    depth_stats = [
        {"family": family, "group": group, "task": task, "t": t,
         "k_star": k_star(scores, tau), "delta": delta(scores, k_high, k_low), "score_by_depth": dict(sorted(scores.items()))}
        for (family, group, task, t), scores in sorted(mean_score.items())
    ]
    regressions = {
        f"{family}/{group}/{task}": interaction_regression(rows)
        for (family, group, task), rows in sorted(regression_rows.items())
        if len({row["k"] for row in rows}) > 1 and len({row["t"] for row in rows}) > 1
    }
    return {"tau": tau, "cells": cells, "depth_statistics": depth_stats, "interaction_regressions": regressions}
