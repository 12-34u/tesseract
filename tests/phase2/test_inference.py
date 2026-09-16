"""Phase 2 statistics: paired Δ and DiD, CIs, regression on primary T, decision rule, baselines, compute, floors."""

import math

import pytest

from phase2.config import load_amendment, load_amendment_02
from phase2.inference import (
    SCORE,
    analyze,
    ci_excludes_zero,
    decision_rule,
    index_results,
    paired_delta,
    paired_did,
    stage_review,
)

RULE = load_amendment_02().decision_rule
A1 = load_amendment()


def result(family, task, t, depth, seed, score, final=None, val=None, head_dim=32, flops=None, params=837_436):
    return {
        "cell": {"family": family, "group": "A5", "task": task, "t": t, "depth": depth, "seed": seed},
        "test": {SCORE: score, "exact_match": 100.0 * score},
        "test_final": {SCORE: score if final is None else final, "exact_match": 0.0},
        "best_val": {SCORE: score if val is None else val},
        "model": {"parameter_count": params, "forward_flops_per_sequence": flops or 1_000_000 * depth,
                  "head_dim": head_dim},
        "seconds_per_train_step": 0.01 * depth,
    }


def tesseract_cells(effects, base=0.3):
    """effects[(task, T)] = per-seed Δ(K8−K1); score = base + Δ·log2(K)/3."""
    out = []
    for (task, t), per_seed in effects.items():
        b = base[(task, t)] if isinstance(base, dict) else base
        for seed, d in enumerate(per_seed):
            for k in (1, 2, 4, 8):
                out.append(result("tesseract", task, t, k, seed, b + d * math.log2(k) / 3))
    return out


SUPPORT = {("main", 4): (0.09, 0.10, 0.11), ("main", 8): (0.39, 0.40, 0.41),
           ("c1_span", 4): (-0.01, 0.0, 0.01), ("c1_span", 8): (-0.01, 0.0, 0.01),
           ("c2_word", 4): (0.05, 0.05, 0.05), ("c2_word", 8): (0.10, 0.11, 0.12)}


def outcome(results):
    return decision_rule(index_results(results), RULE, "best")


def test_paired_delta_and_did_by_hand():
    index = index_results(tesseract_cells(SUPPORT))
    delta = paired_delta(index, "tesseract", "main", 8, 8, 1, "best")
    assert delta["per_seed"] == pytest.approx({0: 0.39, 1: 0.40, 2: 0.41})
    assert delta["summary"]["mean"] == pytest.approx(0.40)
    half = 4.303 * 0.01 / math.sqrt(3)
    assert delta["summary"]["low"] == pytest.approx(0.40 - half) and delta["summary"]["high"] == pytest.approx(0.40 + half)
    did = paired_did(index, "tesseract", "main", 8, 4, 8, 1, "best")
    assert did["per_seed"] == pytest.approx({0: 0.30, 1: 0.30, 2: 0.30})


def test_ci_excludes_zero():
    assert ci_excludes_zero({"mean": 1, "low": 0.1, "high": 2, "n": 3})
    assert ci_excludes_zero({"mean": -1, "low": -2, "high": -0.1, "n": 3})
    assert not ci_excludes_zero({"mean": 0.1, "low": -0.1, "high": 0.3, "n": 3})
    assert not ci_excludes_zero({"mean": 0.1, "low": None, "high": None, "n": 1})
    assert not ci_excludes_zero(None)


def test_rule_supports_depth_effect():
    r = outcome(tesseract_cells(SUPPORT))
    assert r["outcome"] == "supports_sequential_depth_effect" and all(r["conditions"].values())


def test_rule_span_explanation():
    effects = {**SUPPORT, ("c1_span", 8): (0.29, 0.30, 0.31)}
    r = outcome(tesseract_cells(effects))
    assert r["outcome"] == "span_or_receptive_field_explanation" and r["flags"]["C1_span_effect"]


def test_rule_composition_without_depth():
    effects = {**SUPPORT, ("c2_word", 4): (0.0, 0.0, 0.0), ("c2_word", 8): (0.39, 0.40, 0.41)}
    assert outcome(tesseract_cells(effects))["outcome"] == "composition_without_sequential_depth"


def test_rule_inconclusive_on_floor_and_ceiling():
    floor = tesseract_cells(SUPPORT, base={**{k: 0.3 for k in SUPPORT}})
    floor = [r if (r["cell"]["task"], r["cell"]["t"]) != ("main", 8) else {**r, "test": {SCORE: 0.01, "exact_match": 0}}
             for r in floor]
    r = outcome(floor)
    assert r["flags"]["main_floor_at_T_high"] and r["outcome"] in ("inconclusive", "incomplete_data")

    ceiling = tesseract_cells(SUPPORT, base={**{k: 0.3 for k in SUPPORT}, ("main", 4): 0.95})
    r = outcome(ceiling)
    assert r["flags"]["main_ceiling_K_low_at_T_low"] and r["outcome"] == "inconclusive"


def test_rule_incomplete_without_controls():
    effects = {k: v for k, v in SUPPORT.items() if k[0] != "c2_word"}
    assert outcome(tesseract_cells(effects))["outcome"] == "incomplete_data"


def test_analysis_primary_regression_excludes_diagnostic_t():
    effects = {**SUPPORT, ("main", 1): (0.0, 0.0, 0.0), ("main", 2): (0.8, 0.8, 0.8)}
    analysis = analyze(tesseract_cells(effects), A1, RULE)
    block = analysis["checkpoints"]["best"]
    assert block["delta"]["main"][2]["role"] == "diagnostic" and block["delta"]["main"][1]["role"] == "anchor"
    regression = block["regression_primary_t"]["main"]
    assert regression["n"] == 3 * 4 * 2  # T ∈ {4, 8} only
    assert regression["log2K_x_log2T"] > 0
    assert block["k_star"]["main"][8] is None  # max score 0.71 < tau


def test_baselines_and_compute_normalised():
    cells = tesseract_cells(SUPPORT)
    for seed in range(3):
        for t in (4, 8):
            tess8 = next(r for r in cells if r["cell"] == {"family": "tesseract", "group": "A5", "task": "main", "t": t,
                                                           "depth": 8, "seed": seed})
            s = tess8["test"][SCORE]
            cells.append(result("unrolled", "main", t, 8, seed, s + 0.05, flops=8_000_000, params=6_365_756))
            cells.append(result("param_matched", "main", t, 8, seed, s - 0.05, head_dim=11, flops=1_100_000))
            cells.append(result("width_scaled", "main", t, 8, seed, 0.95, flops=8_000_100, params=6_383_992))
    analysis = analyze(cells, A1, RULE)
    baselines = analysis["checkpoints"]["best"]["baselines"]
    assert baselines[8]["unrolled_L8_minus_tesseract_K8"]["summary"]["mean"] == pytest.approx(0.05)
    pm = baselines[8]["param_matched_L8_minus_tesseract_K8"]
    assert pm["summary"]["mean"] == pytest.approx(-0.05) and (pm["head_dim"], pm["tesseract_head_dim"]) == (11, 32)
    compute = analysis["checkpoints"]["best"]["compute_normalised"][8]
    assert compute["min_flops_reaching_tau"]["width_scaled"] == 8_000_100
    assert compute["min_flops_reaching_tau"]["tesseract"] is None
    assert {p["family"] for p in compute["equal_flops_at_tesseract_k8"]} == {"tesseract", "unrolled", "width_scaled"}


def test_final_checkpoint_sensitivity_is_separate():
    cells = [{**r, "test_final": {SCORE: 0.0, "exact_match": 0.0}} for r in tesseract_cells(SUPPORT)]
    analysis = analyze(cells, A1, RULE)
    assert analysis["checkpoints"]["best"]["decision_rule"]["outcome"] == "supports_sequential_depth_effect"
    assert analysis["checkpoints"]["final"]["decision_rule"]["outcome"] != "supports_sequential_depth_effect"


def test_stage_review_uses_validation_scores():
    review = load_amendment_02().stage1_review
    floor = [result("tesseract", "main", 4, k, s, score=0.9, val=0.01) for k in (1, 2, 4, 8) for s in range(3)]
    assert stage_review(floor, review)["status"] == "floor"  # test scores ignored
    ceiling = [result("tesseract", "main", 4, k, s, score=0.0, val=0.95) for k in (1, 2, 4, 8) for s in range(3)]
    assert stage_review(ceiling, review)["status"] == "ceiling"
    mixed = [result("tesseract", "main", 4, k, s, score=0.0, val=0.1 * k) for k in (1, 2, 4, 8) for s in range(3)]
    assert stage_review(mixed, review)["status"] == "informative"


def test_duplicate_results_rejected():
    r = result("tesseract", "main", 4, 1, 0, 0.5)
    with pytest.raises(ValueError, match="duplicate"):
        index_results([r, r])
