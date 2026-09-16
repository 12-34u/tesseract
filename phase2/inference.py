"""Phase 2 statistical analysis.

Pre-registered (PHASE2_BENCHMARK_DESIGN.md §4.12, §8 Amendment 01, §10 Amendment 02 draft):
* score: test chance-normalised token accuracy. The best-validation checkpoint
  is primary; the final checkpoint is a sensitivity check;
* paired, seed-level K effect Δ_s(T) = score(K_high) − score(K_low), and
  DiD_s = Δ_s(T_high) − Δ_s(T_low);
* mean and 95 % t-interval across seeds;
* regression score ~ log2K + log2T + log2K·log2T + seed fixed effects, on the
  primary T only;
* floor / ceiling detection, the amended §5 decision rule (D1), baseline
  differences (with the D3 head-dimension confound) and compute-normalised
  comparisons.

Inputs are the ``result.json`` dicts written by ``phase2.runner.run_cell``.
"""

from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from phase2.config import AmendmentConfig, DecisionRuleConfig, Stage1ReviewConfig
from phase2.metrics import interaction_regression, k_star, mean_ci95

SCORE = "chance_normalised_token_accuracy"
CHECKPOINT_KEYS = {"best": "test", "final": "test_final"}
Key = Tuple[str, str, int, int, int]  # family, task, T, depth, seed


def index_results(results: Iterable[Dict]) -> Dict[Key, Dict]:
    index: Dict[Key, Dict] = {}
    for r in results:
        c = r["cell"]
        key = (c["family"], c["task"], c["t"], c["depth"], c["seed"])
        if key in index:
            raise ValueError(f"duplicate result for {key}")
        index[key] = r
    return index


def test_score(result: Dict, checkpoint: str, metric: str = SCORE) -> Optional[float]:
    block = result.get(CHECKPOINT_KEYS[checkpoint])
    return None if block is None else block.get(metric)


def best_val_score(result: Dict) -> Optional[float]:
    return (result.get("best_val") or {}).get(SCORE)


def _summary(values: Iterable[Optional[float]]) -> Optional[Dict]:
    values = [v for v in values if v is not None]
    return mean_ci95(values) if values else None


def ci_excludes_zero(summary: Optional[Dict]) -> bool:
    return bool(summary) and summary.get("low") is not None and (summary["low"] > 0 or summary["high"] < 0)


def _positive_and_excludes_zero(summary: Optional[Dict]) -> bool:
    return bool(summary) and summary["mean"] > 0 and summary.get("low") is not None and summary["low"] > 0


def _pairing_coverage(expected_seeds: Iterable[int], per_seed: Dict[int, float]) -> Dict:
    """Report seeds that were expected but produced no paired value.

    A cell that diverged, or was never run, silently disappears from a paired
    comparison and shrinks n without changing any number that is printed.
    Recording the loss keeps a degraded seed set visible. This is reporting
    only: it is deliberately not consumed by the decision rule, whose outcome
    logic is pre-registered (Amendment 02 D1).
    """
    expected = sorted(expected_seeds)
    dropped = [s for s in expected if s not in per_seed]
    return {"seeds_expected": expected, "seeds_used": sorted(per_seed),
            "seeds_dropped": dropped, "complete_pairing": not dropped}


# ============================================================================
# Paired effects
# ============================================================================


def paired_delta(index: Dict[Key, Dict], family: str, task: str, t: int, k_high: int, k_low: int,
                 checkpoint: str) -> Dict:
    per_seed = {}
    for seed in sorted({k[4] for k in index if k[:3] == (family, task, t)}):
        high, low = index.get((family, task, t, k_high, seed)), index.get((family, task, t, k_low, seed))
        if high is not None and low is not None:
            a, b = test_score(high, checkpoint), test_score(low, checkpoint)
            if a is not None and b is not None:
                per_seed[seed] = a - b
    expected = sorted({k[4] for k in index if k[:3] == (family, task, t)})
    return {"family": family, "task": task, "t": t, "k_high": k_high, "k_low": k_low,
            "per_seed": per_seed, "summary": _summary(per_seed.values()),
            **_pairing_coverage(expected, per_seed)}


def paired_did(index: Dict[Key, Dict], family: str, task: str, t_high: int, t_low: int, k_high: int, k_low: int,
               checkpoint: str) -> Dict:
    high = paired_delta(index, family, task, t_high, k_high, k_low, checkpoint)["per_seed"]
    low = paired_delta(index, family, task, t_low, k_high, k_low, checkpoint)["per_seed"]
    per_seed = {s: high[s] - low[s] for s in high if s in low}
    return {"family": family, "task": task, "t_high": t_high, "t_low": t_low, "per_seed": per_seed,
            "summary": _summary(per_seed.values()),
            **_pairing_coverage(sorted(set(high) | set(low)), per_seed)}


def paired_cross(index: Dict[Key, Dict], a: Tuple[str, str, int, int], b: Tuple[str, str, int, int],
                 checkpoint: str) -> Dict:
    """Seed-paired score(a) − score(b) for two (family, task, T, depth) cells."""
    seeds = sorted({k[4] for k in index if k[:4] == a} & {k[4] for k in index if k[:4] == b})
    per_seed = {}
    for s in seeds:
        x, y = test_score(index[a + (s,)], checkpoint), test_score(index[b + (s,)], checkpoint)
        if x is not None and y is not None:
            per_seed[s] = x - y
    return {"a": list(a), "b": list(b), "per_seed": per_seed, "summary": _summary(per_seed.values()),
            **_pairing_coverage(seeds, per_seed)}


# ============================================================================
# Tables, floors, regression
# ============================================================================


def cell_table(results: Iterable[Dict], checkpoint: str, roles: Dict[int, str]) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in results:
        c = r["cell"]
        groups[(c["family"], c["task"], c["t"], c["depth"])].append(r)
    rows = []
    for (family, task, t, depth), runs in sorted(groups.items()):
        model = runs[0]["model"]
        rows.append({
            "family": family, "task": task, "t": t, "depth": depth, "role": roles.get(t, "unassigned"),
            "seeds": sorted(r["cell"]["seed"] for r in runs),
            "score": _summary(test_score(r, checkpoint) for r in runs),
            "exact_match": _summary(test_score(r, checkpoint, "exact_match") for r in runs),
            "parameter_count": model["parameter_count"],
            "forward_flops_per_sequence": model["forward_flops_per_sequence"],
            "head_dim": model.get("head_dim"),
            "seconds_per_train_step": _summary(r.get("seconds_per_train_step") for r in runs),
        })
    return rows


def floor_ceiling(index: Dict[Key, Dict], family: str, task: str, t: int, floor: float, tau: float, k_low: int,
                  scorer: Callable[[Dict], Optional[float]]) -> Dict:
    runs = {k: r for k, r in index.items() if k[:3] == (family, task, t)}
    scores = [scorer(r) for r in runs.values()]
    low_scores = [scorer(r) for k, r in runs.items() if k[3] == k_low]
    return {
        "runs": len(runs),
        "floor": bool(scores) and all(s is None or s < floor for s in scores),
        "ceiling": bool(low_scores) and all(s is not None and s >= tau for s in low_scores),
    }


def primary_regression(index: Dict[Key, Dict], family: str, task: str, primary_t: Sequence[int],
                       checkpoint: str) -> Optional[Dict]:
    rows = []
    for (fam, tk, t, depth, seed), r in index.items():
        if fam == family and tk == task and t in primary_t:
            s = test_score(r, checkpoint)
            if s is not None:
                rows.append({"seed": seed, "k": depth, "t": t, "score": s})
    seeds = {row["seed"] for row in rows}
    if len({row["k"] for row in rows}) < 2 or len({row["t"] for row in rows}) < 2 or len(rows) <= 3 + len(seeds):
        return None
    return interaction_regression(rows)


# ============================================================================
# Decision rule (Amendment 02 D1), baselines, compute
# ============================================================================


def decision_rule(index: Dict[Key, Dict], rule: DecisionRuleConfig, checkpoint: str) -> Dict:
    t_low, t_high = min(rule.primary_t_values), max(rule.primary_t_values)
    kh, kl = rule.k_high, rule.k_low
    main_delta_high_full = paired_delta(index, "tesseract", "main", t_high, kh, kl, checkpoint)
    main_did_full = paired_did(index, "tesseract", "main", t_high, t_low, kh, kl, checkpoint)
    c2_did_full = paired_did(index, "tesseract", "c2_word", t_high, t_low, kh, kl, checkpoint)
    c1_delta_low_full = paired_delta(index, "tesseract", "c1_span", t_low, kh, kl, checkpoint)
    c1_delta_high_full = paired_delta(index, "tesseract", "c1_span", t_high, kh, kl, checkpoint)
    main_delta_high = main_delta_high_full["summary"]
    main_did = main_did_full["summary"]
    c2_did = c2_did_full["summary"]
    c1_delta_low = c1_delta_low_full["summary"]
    c1_delta_high = c1_delta_high_full["summary"]

    def scorer(r):
        return test_score(r, checkpoint)

    main_floor_t_high = floor_ceiling(index, "tesseract", "main", t_high, rule.floor_threshold, rule.tau, kl, scorer)["floor"]
    main_ceiling_t_low = floor_ceiling(index, "tesseract", "main", t_low, rule.floor_threshold, rule.tau, kl, scorer)["ceiling"]

    needed = [main_delta_high, main_did, c2_did, c1_delta_low, c1_delta_high]
    complete = all(s is not None and s["n"] >= 2 for s in needed)
    conditions = {
        "1_main_delta_T_high_positive_ci_excludes_0": _positive_and_excludes_zero(main_delta_high),
        "2_main_DiD_positive_ci_excludes_0": _positive_and_excludes_zero(main_did),
        "3a_main_DiD_mean_exceeds_C2_DiD_mean": complete and main_did["mean"] > c2_did["mean"],
        "3b_C1_delta_T_low_ci_includes_0": complete and not ci_excludes_zero(c1_delta_low),
        "3c_C1_delta_T_high_ci_includes_0": complete and not ci_excludes_zero(c1_delta_high),
    }
    span = ci_excludes_zero(c1_delta_low) or ci_excludes_zero(c1_delta_high)
    composition = complete and c2_did["mean"] >= main_did["mean"]

    if not complete:
        outcome = "incomplete_data"
    elif main_floor_t_high or main_ceiling_t_low:
        outcome = "inconclusive"
    elif all(conditions.values()):
        outcome = "supports_sequential_depth_effect"
    elif span:
        outcome = "span_or_receptive_field_explanation"
    elif composition:
        outcome = "composition_without_sequential_depth"
    else:
        outcome = "inconclusive"
    return {
        "checkpoint": checkpoint,
        "outcome": outcome,
        "conditions": conditions,
        "flags": {"main_floor_at_T_high": main_floor_t_high, "main_ceiling_K_low_at_T_low": main_ceiling_t_low,
                  "C1_span_effect": span, "C2_composition_at_least_main": composition, "complete": complete,
                  # Reporting only; not consumed by the outcome logic above.
                  "all_comparisons_fully_paired": all(
                      c["complete_pairing"] for c in (main_delta_high_full, main_did_full, c2_did_full,
                                                      c1_delta_low_full, c1_delta_high_full))},
        "summaries": {"main_delta_T_high": main_delta_high, "main_DiD": main_did, "C2_DiD": c2_did,
                      "C1_delta_T_low": c1_delta_low, "C1_delta_T_high": c1_delta_high},
        "pairing": {name: {k: full[k] for k in ("seeds_expected", "seeds_used", "seeds_dropped", "complete_pairing")}
                    for name, full in (("main_delta_T_high", main_delta_high_full), ("main_DiD", main_did_full),
                                       ("C2_DiD", c2_did_full), ("C1_delta_T_low", c1_delta_low_full),
                                       ("C1_delta_T_high", c1_delta_high_full))},
        "note": "baselines are interpretive only and do not enter this rule (Amendment 02 D1)",
    }


def baseline_comparisons(index: Dict[Key, Dict], primary_t: Sequence[int], checkpoint: str) -> Dict:
    out = {}
    for t in primary_t:
        entry = {}
        for family in ("unrolled", "param_matched"):
            for depth in sorted({k[3] for k in index if k[0] == family and k[1] == "main" and k[2] == t}):
                comparison = paired_cross(index, (family, "main", t, depth), ("tesseract", "main", t, depth), checkpoint)
                if family == "param_matched":
                    sample = next(r for k, r in index.items() if k[:4] == (family, "main", t, depth))
                    reference = next((r for k, r in index.items() if k[:4] == ("tesseract", "main", t, depth)), None)
                    comparison["head_dim"] = sample["model"].get("head_dim")
                    comparison["tesseract_head_dim"] = reference["model"].get("head_dim") if reference else None
                    comparison["known_confound"] = "head dimension differs (Amendment 02 D3)"
                entry[f"{family}_L{depth}_minus_tesseract_K{depth}"] = comparison
        for depth in sorted({k[3] for k in index if k[0] == "width_scaled" and k[1] == "main" and k[2] == t}):
            entry[f"width_scaled_matchedK{depth}_minus_tesseract_K{depth}"] = paired_cross(
                index, ("width_scaled", "main", t, depth), ("tesseract", "main", t, depth), checkpoint)
        out[t] = entry
    return out


def compute_normalised(cells: List[Dict], tau: float) -> Dict:
    per_t: Dict[int, Dict] = {}
    for t in sorted({c["t"] for c in cells if c["task"] == "main"}):
        rows = [c for c in cells if c["task"] == "main" and c["t"] == t]
        points = [{"family": c["family"], "depth": c["depth"], "forward_flops_per_sequence": c["forward_flops_per_sequence"],
                   "parameter_count": c["parameter_count"], "score": c["score"],
                   "seconds_per_train_step": c["seconds_per_train_step"]}
                  for c in sorted(rows, key=lambda c: c["forward_flops_per_sequence"])]
        min_flops = {}
        for family in sorted({c["family"] for c in rows}):
            reaching = [c["forward_flops_per_sequence"] for c in rows
                        if c["family"] == family and c["score"] is not None and c["score"]["mean"] >= tau]
            min_flops[family] = min(reaching) if reaching else None
        equal_flops = [p for p in points if (p["family"], p["depth"]) in {("tesseract", 8), ("unrolled", 8), ("width_scaled", 8)}]
        per_t[t] = {"points": points, "min_flops_reaching_tau": min_flops, "equal_flops_at_tesseract_k8": equal_flops}
    return per_t


# ============================================================================
# Stage review and full analysis
# ============================================================================


def stage_review(results: Iterable[Dict], review: Stage1ReviewConfig, k_low: int = 1) -> Dict:
    """Stage t4 review, declared in advance. Uses best-checkpoint VALIDATION scores only."""
    index = index_results(results)
    fc = floor_ceiling(index, "tesseract", "main", review.t, review.floor_threshold, review.ceiling_tau, k_low,
                       best_val_score)
    status = "floor" if fc["floor"] else ("ceiling" if fc["ceiling"] else "informative")
    actions = {
        "floor": "STOP AND DIAGNOSE: every K and seed is below the floor at this T",
        "ceiling": "REPORT: K=1 is at the ceiling on all seeds (no room for a K effect); continue only with approval",
        "informative": "STOP FOR REVIEW before the next stage",
    }
    return {"t": review.t, "basis": "best-checkpoint validation chance-normalised token accuracy",
            **fc, "status": status, "action": actions[status]}


def analyze(results: List[Dict], amendment01: AmendmentConfig, rule: DecisionRuleConfig) -> Dict:
    roles = {**{t: "anchor" for t in amendment01.anchor_t_values},
             **{t: "diagnostic" for t in amendment01.diagnostic_t_values},
             **{t: "primary_depth" for t in amendment01.primary_depth_t_values}}
    index = index_results(results)
    tasks = sorted({k[1] for k in index if k[0] == "tesseract"})
    analysis = {
        "t_roles": roles,
        "primary_checkpoint": "best",
        "sensitivity_checkpoint": "final",
        "decision_rule_config": {"primary_t_values": list(rule.primary_t_values), "k_low": rule.k_low,
                                 "k_high": rule.k_high, "tau": rule.tau, "floor": rule.floor_threshold},
        "known_confounds": ["parameter-matched baseline: reduced attention head dimension (Amendment 02 D3)",
                            "shared learning rate across families, no tuning (Amendment 02 D2)"],
        "n_results": len(results),
        "checkpoints": {},
    }
    t_low, t_high = min(rule.primary_t_values), max(rule.primary_t_values)
    for checkpoint in ("best", "final"):
        cells = cell_table(results, checkpoint, roles)
        delta, kstar = {}, {}
        for task in tasks:
            ts = sorted({k[2] for k in index if k[0] == "tesseract" and k[1] == task})
            delta[task] = {t: {**paired_delta(index, "tesseract", task, t, rule.k_high, rule.k_low, checkpoint),
                               "role": roles.get(t, "unassigned")} for t in ts}
            kstar[task] = {t: k_star({c["depth"]: c["score"]["mean"] if c["score"] else None for c in cells
                                      if c["family"] == "tesseract" and c["task"] == task and c["t"] == t}, rule.tau)
                           for t in ts}

        def scorer(r, cp=checkpoint):
            return test_score(r, cp)

        analysis["checkpoints"][checkpoint] = {
            "cells": cells,
            "delta": delta,
            "did_primary": {task: paired_did(index, "tesseract", task, t_high, t_low, rule.k_high, rule.k_low, checkpoint)
                            for task in tasks},
            "k_star": kstar,
            "regression_primary_t": {task: primary_regression(index, "tesseract", task, rule.primary_t_values, checkpoint)
                                     for task in tasks},
            "floor_ceiling": {f"{fam}/{task}/T{t}": floor_ceiling(index, fam, task, t, rule.floor_threshold, rule.tau,
                                                                 rule.k_low, scorer)
                              for fam, task, t in sorted({k[:3] for k in index})},
            "decision_rule": decision_rule(index, rule, checkpoint),
            "baselines": baseline_comparisons(index, rule.primary_t_values, checkpoint),
            "compute_normalised": compute_normalised(cells, rule.tau),
        }
    return analysis
