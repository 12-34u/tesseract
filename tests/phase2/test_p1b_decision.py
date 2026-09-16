"""Amendment 02 D5 evaluator: width and budget selection from a P1b report.

Every report here is synthetic. No real P1b data is read, and nothing trains.
"""

import json
import math

import pytest

from phase2.p1b_decision import (
    CURVE_SCORE,
    DO_NOT_PROCEED,
    LN_VOCAB,
    SELECT,
    D5DataError,
    evaluate_d5,
    evaluate_d5_file,
    render_d5_decision,
)

EVAL_EVERY = 250
MAX_STEPS = 20000
NUM_EVALS = MAX_STEPS // EVAL_EVERY  # 80


def curve(scores):
    """A validation curve of len(scores) evaluations, every 250 steps."""
    return [{"step": float((i + 1) * EVAL_EVERY), "val_chance_normalised_token_accuracy": float(s),
             "val_loss": 1.0} for i, s in enumerate(scores)]


def flat(value, n=NUM_EVALS):
    """A curve that reaches `value` immediately and stays there (levelled off)."""
    return [value] * n


def rising(start, end, n=NUM_EVALS):
    """A curve rising linearly throughout, i.e. never levelling off."""
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def saturating(final, knee_eval, n=NUM_EVALS):
    """Rises linearly to `final` by evaluation `knee_eval`, then holds."""
    return [final * min(1.0, (i + 1) / knee_eval) for i in range(n)]


def run(scores, final_loss, params, model_base="prototype_small", final_score=None):
    final = scores[-1] if final_score is None else final_score
    return {
        "model_variant": "x", "model_base": model_base, "curve": curve(scores),
        "headline": {"parameter_count": params, "final_val_chance_normalised": final,
                     "final_val_loss": final_loss, "best_val_chance_normalised": max(scores) if scores else None,
                     "steps_run": MAX_STEPS},
    }


def report(runs, t=8, eval_every=EVAL_EVERY, max_steps=MAX_STEPS, k_values=(1, 8)):
    return {
        "settings": {"t": t, "k_values": list(k_values), "max_steps": max_steps, "eval_every": eval_every,
                     "seed": 0, "floor_threshold": 0.02,
                     "chance_level_val_loss_ln_vocab": math.log(60)},
        "model_configs": {name: {"d_model": 128} for name in runs},
        "runs": runs,
        "single_seed_note": "one seed per cell: differences are descriptive, not statistical claims",
    }


SMALL, MEDIUM = 222140, 837436
GOOD_LOSS = LN_VOCAB - 0.5
BAD_LOSS = LN_VOCAB + 0.5


def learnable_run(params, model_base, knee=20):
    return run(saturating(0.50, knee), GOOD_LOSS, params, model_base)


def dead_run(params, model_base):
    return run(flat(0.001), BAD_LOSS, params, model_base, final_score=0.001)


# ---------------------------------------------------------------------------
# Width selection
# ---------------------------------------------------------------------------


def test_both_widths_learnable_selects_the_smaller():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")},
                "medium": {"1": learnable_run(MEDIUM, "phase2/prototype_medium"),
                           "8": learnable_run(MEDIUM, "phase2/prototype_medium")}})
    d = evaluate_d5(r)
    assert d["width_decision"]["outcome"] == SELECT
    assert d["width_decision"]["selected_width"] == "small"
    assert d["width_decision"]["parameter_count"] == SMALL
    assert [w["width"] for w in d["widths"]] == ["small", "medium"]  # ordered by size


def test_only_medium_learnable_selects_medium():
    r = report({"small": {"1": dead_run(SMALL, "prototype_small"),
                          "8": dead_run(SMALL, "prototype_small")},
                "medium": {"1": dead_run(MEDIUM, "phase2/prototype_medium"),
                           "8": learnable_run(MEDIUM, "phase2/prototype_medium")}})
    d = evaluate_d5(r)
    assert d["width_decision"]["outcome"] == SELECT
    assert d["width_decision"]["selected_width"] == "medium"
    assert d["width_decision"]["learnable_k"] == [8]


def test_neither_width_learnable_returns_do_not_proceed():
    r = report({"small": {"1": dead_run(SMALL, "prototype_small"), "8": dead_run(SMALL, "prototype_small")},
                "medium": {"1": dead_run(MEDIUM, "phase2/prototype_medium"),
                           "8": dead_run(MEDIUM, "phase2/prototype_medium")}})
    d = evaluate_d5(r)
    assert d["width_decision"]["outcome"] == DO_NOT_PROCEED
    assert d["width_decision"]["selected_width"] is None
    assert d["budget_decision"]["outcome"] == DO_NOT_PROCEED
    assert d["budget_decision"]["recommended_max_steps"] is None


def test_k1_still_at_floor_at_the_selected_width_is_reported():
    """The width is selected on K=8 alone; K=1 at the floor must be stated, not hidden."""
    r = report({"small": {"1": run(flat(0.005), GOOD_LOSS, SMALL), "8": learnable_run(SMALL, "prototype_small")}})
    d = evaluate_d5(r)
    w = d["width_decision"]
    assert w["outcome"] == SELECT and w["selected_width"] == "small"
    assert w["learnable_k"] == [8]
    assert w["k_low"] == 1
    assert w["k_low_at_floor"] is True
    assert w["k_low_learnable"] is False
    assert "AT FLOOR" in render_d5_decision(d)


def test_k1_above_the_floor_is_reported_as_such():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    w = evaluate_d5(r)["width_decision"]
    assert w["k_low_at_floor"] is False and w["k_low_learnable"] is True


# ---------------------------------------------------------------------------
# The three learnability criteria, including partial satisfaction
# ---------------------------------------------------------------------------


def test_tail_mean_passes_but_final_value_fails():
    """Criterion 2 satisfied, criterion 1 not: the K is not learnable."""
    scores = flat(0.50)
    r = report({"small": {"1": dead_run(SMALL, "prototype_small"),
                          "8": run(scores, GOOD_LOSS, SMALL, final_score=0.001)}})
    d = evaluate_d5(r)
    k8 = d["widths"][0]["per_k"][8]
    assert k8["criteria"]["2_final_20_evaluation_mean_at_least_0.02"] is True
    assert k8["criteria"]["1_final_chance_normalised_at_least_0.02"] is False
    assert k8["learnable"] is False
    assert d["width_decision"]["outcome"] == DO_NOT_PROCEED


def test_final_value_passes_but_tail_mean_fails():
    """Criterion 1 satisfied by a last-moment spike, criterion 2 not.

    The spike must be small enough not to drag the 20-evaluation mean over the
    threshold on its own: 19 evaluations at 0.001 plus one at 0.05 averages
    0.00345. This is the case criterion 2 exists to catch.
    """
    scores = flat(0.001)[:-1] + [0.05]
    r = report({"small": {"1": dead_run(SMALL, "prototype_small"),
                          "8": run(scores, GOOD_LOSS, SMALL)}})
    d = evaluate_d5(r)
    k8 = d["widths"][0]["per_k"][8]
    assert k8["criteria"]["1_final_chance_normalised_at_least_0.02"] is True
    assert k8["final_evaluations_mean"] == pytest.approx((19 * 0.001 + 0.05) / 20)
    assert k8["criteria"]["2_final_20_evaluation_mean_at_least_0.02"] is False
    assert k8["learnable"] is False
    assert d["width_decision"]["outcome"] == DO_NOT_PROCEED


def test_final_loss_exactly_at_ln_60_fails_the_strict_inequality():
    """Criterion 3 is '< ln 60'; the boundary itself does not pass."""
    r = report({"small": {"1": dead_run(SMALL, "prototype_small"),
                          "8": run(flat(0.50), LN_VOCAB, SMALL)}})
    d = evaluate_d5(r)
    k8 = d["widths"][0]["per_k"][8]
    assert k8["criteria"]["3_final_val_loss_below_ln_60"] is False
    assert k8["learnable"] is False
    assert d["width_decision"]["outcome"] == DO_NOT_PROCEED


def test_final_loss_just_below_ln_60_passes():
    r = report({"small": {"1": dead_run(SMALL, "prototype_small"),
                          "8": run(flat(0.50), math.nextafter(LN_VOCAB, 0.0), SMALL)}})
    k8 = evaluate_d5(r)["widths"][0]["per_k"][8]
    assert k8["criteria"]["3_final_val_loss_below_ln_60"] is True and k8["learnable"] is True


def test_criterion_2_window_is_the_last_twenty_evaluations():
    d = evaluate_d5(report({"small": {"1": dead_run(SMALL, "prototype_small"),
                                      "8": learnable_run(SMALL, "prototype_small")}}))
    span = d["widths"][0]["per_k"][8]["final_evaluations_span_steps"]
    assert span == [MAX_STEPS - 19 * EVAL_EVERY, MAX_STEPS]
    assert span[1] - span[0] == 4750  # 20 evaluations inclusive == the last 5,000 steps


def test_diverged_run_with_null_metrics_is_not_learnable_and_does_not_raise():
    diverged = {"model_variant": "x", "model_base": "prototype_small", "curve": curve(flat(0.0, 3)),
                "headline": {"parameter_count": SMALL, "final_val_chance_normalised": None,
                             "final_val_loss": None, "best_val_chance_normalised": None, "steps_run": 3}}
    d = evaluate_d5(report({"small": {"1": diverged, "8": learnable_run(SMALL, "prototype_small")}}))
    k1 = d["widths"][0]["per_k"][1]
    assert k1["learnable"] is False
    assert "3 of the required 20 evaluations" in k1["criterion_2_note"]
    assert d["width_decision"]["k_low_at_floor"] is None  # unknown, not silently "at floor"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_levelled_off_curve_picks_an_early_multiple_of_5000():
    """Saturates by evaluation 20 (step 5,000), so the budget is well under 20,000."""
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small", knee=20),
                          "8": learnable_run(SMALL, "prototype_small", knee=20)}})
    b = evaluate_d5(r)["budget_decision"]
    assert b["outcome"] == SELECT
    assert b["recommended_max_steps"] % 5000 == 0
    assert b["recommended_max_steps"] < MAX_STEPS
    assert b["not_converged"] is False


def test_still_rising_curve_recommends_20000_marked_not_converged():
    r = report({"small": {"1": run(rising(0.05, 0.60), GOOD_LOSS, SMALL),
                          "8": run(rising(0.05, 0.60), GOOD_LOSS, SMALL)}})
    b = evaluate_d5(r)["budget_decision"]
    assert b["recommended_max_steps"] == MAX_STEPS
    assert b["not_converged"] is True
    assert "NOT CONVERGED" in render_d5_decision(evaluate_d5(r))


def test_budget_is_the_maximum_over_learnable_k():
    """A budget short for any learnable K would under-train it."""
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small", knee=4),      # levels off very early
                          "8": run(rising(0.05, 0.60), GOOD_LOSS, SMALL)}})          # still rising
    b = evaluate_d5(r)["budget_decision"]
    assert set(b["per_k"]) == {1, 8}
    assert b["recommended_max_steps"] == max(x["recommended_max_steps"] for x in b["per_k"].values())
    assert b["recommended_max_steps"] == MAX_STEPS
    assert b["k_budgets_agree"] is False
    assert b["not_converged"] is True


def test_budget_never_exceeds_the_observed_steps():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    b = evaluate_d5(r)["budget_decision"]
    assert b["recommended_max_steps"] <= MAX_STEPS
    assert all(c["step"] <= MAX_STEPS for x in b["per_k"].values() for c in x["candidates"])


def test_budget_target_is_90_percent_of_the_final_rolling_mean():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    per_k = evaluate_d5(r)["budget_decision"]["per_k"][8]
    assert per_k["target"] == pytest.approx(0.9 * per_k["final_rolling_mean"])


# ---------------------------------------------------------------------------
# Malformed / incomplete data must fail loudly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.pop("runs"), "missing 'runs'"),
    (lambda r: r.pop("settings"), "missing 'settings'"),
    (lambda r: r.pop("single_seed_note"), "missing 'single_seed_note'"),
    (lambda r: r["settings"].pop("max_steps"), "missing 'max_steps'"),
    (lambda r: r["settings"].update(t=4), "D5 is stated for the T=8"),
    (lambda r: r["settings"].update(chance_level_val_loss_ln_vocab=math.log(16)), "does not match the pre-registered"),
    (lambda r: r["settings"].update(eval_every=100), "halves of the pre-registered definition disagree"),
    (lambda r: r["runs"]["small"].pop("1"), "missing K \\[1\\]"),
    (lambda r: r["runs"]["small"]["8"].pop("curve"), "missing 'curve'"),
    (lambda r: r["runs"]["small"]["8"].pop("headline"), "missing 'headline'"),
    (lambda r: r["runs"]["small"]["8"]["headline"].pop("final_val_loss"), "missing 'final_val_loss'"),
    (lambda r: r["runs"]["small"]["8"]["headline"].update(parameter_count=None), "must not be null"),
    (lambda r: r["runs"]["small"]["8"]["headline"].update(final_val_chance_normalised="0.5"), "expected a number"),
    (lambda r: r["runs"]["small"]["8"]["curve"].reverse(), "must be ordered by step"),
    (lambda r: r.update(runs={}), "non-empty mapping"),
])
def test_malformed_reports_raise(mutate, match):
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    mutate(r)
    with pytest.raises(D5DataError, match=match):
        evaluate_d5(r)


def test_parameter_count_must_be_constant_across_k_for_one_width():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL + 1, "prototype_small")}})
    with pytest.raises(D5DataError, match="parameter count differs across K"):
        evaluate_d5(r)


def test_two_widths_of_equal_size_are_refused_rather_than_guessed():
    r = report({"a": {"1": learnable_run(SMALL, "a"), "8": learnable_run(SMALL, "a")},
                "b": {"1": learnable_run(SMALL, "b"), "8": learnable_run(SMALL, "b")}})
    with pytest.raises(D5DataError, match="same parameter count"):
        evaluate_d5(r)


def test_missing_report_file_raises_instead_of_deciding(tmp_path):
    with pytest.raises(D5DataError, match="No P1b diagnostic report"):
        evaluate_d5_file(tmp_path / "diagnostic_report.json")


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "diagnostic_report.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(D5DataError, match="not valid JSON"):
        evaluate_d5_file(path)


def test_file_wrapper_matches_the_pure_evaluator(tmp_path):
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    path = tmp_path / "diagnostic_report.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    assert evaluate_d5_file(path) == evaluate_d5(r)


# ---------------------------------------------------------------------------
# Purity and reporting
# ---------------------------------------------------------------------------


def test_evaluator_is_deterministic_and_does_not_mutate_its_input():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    before = json.dumps(r, sort_keys=True)
    first, second = evaluate_d5(r), evaluate_d5(r)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(r, sort_keys=True) == before


def test_single_seed_fact_is_preserved_in_the_decision():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    d = evaluate_d5(r)
    assert d["p1b"]["seed"] == 0
    assert "one seed per cell" in d["p1b"]["single_seed_note"]
    assert any("single seed" in c for c in d["caveats"])
    assert "single seed" in render_d5_decision(d)


def test_rendered_output_is_complete_for_both_outcomes():
    selected = evaluate_d5(report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                                             "8": learnable_run(SMALL, "prototype_small")}}))
    text = render_d5_decision(selected)
    assert "AMENDMENT 02 D5" in text and "WIDTH: SELECT" in text and "BUDGET: SELECT" in text
    assert "not frozen" in text

    refused = evaluate_d5(report({"small": {"1": dead_run(SMALL, "prototype_small"),
                                            "8": dead_run(SMALL, "prototype_small")}}))
    text = render_d5_decision(refused)
    assert f"WIDTH: {DO_NOT_PROCEED}" in text and f"BUDGET: {DO_NOT_PROCEED}" in text


def test_decision_is_json_serialisable():
    d = evaluate_d5(report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                                      "8": learnable_run(SMALL, "prototype_small")}}))
    assert json.loads(json.dumps(d))["width_decision"]["selected_width"] == "small"


# ---------------------------------------------------------------------------
# Budget aggregation: maximum over the learnable K (approved interpretation)
# ---------------------------------------------------------------------------


def test_two_learnable_k_with_different_budgets_selects_the_maximum():
    """Both K learnable, both levelled off, different knees -> the larger budget wins.

    The selected width has to support the whole K in {1,2,4,8} comparison, so a
    budget short for any learnable K would under-train it.
    """
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small", knee=8),    # levels off early
                          "8": learnable_run(SMALL, "prototype_small", knee=44)}})  # levels off later
    b = evaluate_d5(r)["budget_decision"]

    per_k = {k: v["recommended_max_steps"] for k, v in b["per_k"].items()}
    assert set(per_k) == {1, 8}
    assert per_k[1] < per_k[8], f"fixture must give different per-K budgets, got {per_k}"
    assert b["recommended_max_steps"] == max(per_k.values()) == per_k[8]
    assert b["k_budgets_agree"] is False
    assert b["not_converged"] is False          # the deciding run did level off
    assert b["recommended_max_steps"] < MAX_STEPS
    assert "maximum over the learnable K" in b["aggregation"]


def test_non_learnable_k_does_not_influence_the_budget():
    """A K that fails the criteria contributes no budget, even a larger one."""
    still_rising = run(rising(0.05, 0.60), BAD_LOSS, SMALL)   # fails criterion 3 -> not learnable
    r = report({"small": {"1": still_rising,
                          "8": learnable_run(SMALL, "prototype_small", knee=8)}})
    d = evaluate_d5(r)
    assert d["widths"][0]["per_k"][1]["learnable"] is False
    b = d["budget_decision"]

    assert set(b["per_k"]) == {8}, "only learnable K may contribute a budget"
    assert b["recommended_max_steps"] == b["per_k"][8]["recommended_max_steps"]
    assert b["not_converged"] is False
    # The excluded K, had it counted, would have forced the full 20,000.
    assert b["recommended_max_steps"] < MAX_STEPS


def test_single_learnable_k_uses_its_budget_directly():
    r = report({"small": {"1": dead_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small", knee=8)}})
    d = evaluate_d5(r)
    assert d["width_decision"]["learnable_k"] == [8]
    b = d["budget_decision"]
    assert set(b["per_k"]) == {8}
    assert b["k_budgets_agree"] is True
    assert b["recommended_max_steps"] == b["per_k"][8]["recommended_max_steps"]
    assert b["not_converged"] == b["per_k"][8]["not_converged"]


def test_agreeing_per_k_budgets_report_agreement():
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small", knee=8),
                          "8": learnable_run(SMALL, "prototype_small", knee=8)}})
    b = evaluate_d5(r)["budget_decision"]
    assert b["k_budgets_agree"] is True
    assert len(set(v["recommended_max_steps"] for v in b["per_k"].values())) == 1


def test_not_converged_reflects_the_deciding_run_not_an_early_one():
    """One K levels off early, the other is still rising: the max governs."""
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small", knee=4),
                          "8": run(rising(0.05, 0.60), GOOD_LOSS, SMALL)}})
    b = evaluate_d5(r)["budget_decision"]
    assert b["recommended_max_steps"] == MAX_STEPS
    assert b["not_converged"] is True
    assert b["per_k"][1]["not_converged"] is False
    assert b["per_k"][8]["not_converged"] is True


# ---------------------------------------------------------------------------
# Contract with the real P1b writer: no synthetic-only fields
# ---------------------------------------------------------------------------


def _write_and_read_curve(history, tmp_path):
    """Round-trip a history through the real CSV writer and the real _read_curve."""
    import csv as _csv

    from phase2.runner import _read_curve

    fieldnames = sorted({k for row in history for k in row}, key=lambda k: (k != "step", k))
    with open(tmp_path / "val_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return _read_curve(tmp_path)


def _eval_row(step, score):
    """An evaluation row exactly as phase2.training.train_fixed_steps appends it."""
    return {"step": step, "train_loss_last_batch": 1.0, "train_loss_interval_mean": 1.0,
            "gradient_norm_last": 0.5, "gradient_norm_interval_mean": 0.5, "gradient_norm_interval_max": 0.9,
            "val_loss": 1.0, "val_exact_match": 0.0, "val_token_accuracy": 5.0,
            "val_chance_normalised_token_accuracy": score}


def test_curve_from_the_real_writer_is_accepted(tmp_path):
    """The evaluator must consume a curve produced by the real code path."""
    history = [_eval_row((i + 1) * EVAL_EVERY, 0.5) for i in range(NUM_EVALS)]
    real_curve = _write_and_read_curve(history, tmp_path)
    assert all(CURVE_SCORE in row for row in real_curve)

    run_dict = {"model_base": "prototype_small", "curve": real_curve,
                "headline": {"parameter_count": SMALL, "final_val_chance_normalised": 0.5,
                             "final_val_loss": GOOD_LOSS}}
    d = evaluate_d5(report({"small": {"1": run_dict, "8": run_dict}}))
    assert d["width_decision"]["outcome"] == SELECT
    assert d["widths"][0]["per_k"][8]["learnable"] is True


def test_diverged_curve_from_the_real_writer_does_not_raise(tmp_path):
    """A real non-finite-loss run ends with a row that has no validation columns.

    _read_curve drops the absent columns, so that row arrives without
    val_chance_normalised_token_accuracy. It is a divergence marker, not an
    evaluation, and must make the K not learnable rather than blow up the
    whole evaluation.
    """
    history = [_eval_row(250, 0.01), _eval_row(500, 0.01),
               {"step": 512, "train_loss_last_batch": float("nan"), "gradient_norm_last": float("inf")}]
    real_curve = _write_and_read_curve(history, tmp_path)
    assert CURVE_SCORE not in real_curve[-1], "fixture must reproduce the marker row"

    diverged = {"model_base": "prototype_small", "curve": real_curve,
                "headline": {"parameter_count": SMALL, "final_val_chance_normalised": None,
                             "final_val_loss": None}}
    d = evaluate_d5(report({"small": {"1": diverged, "8": learnable_run(SMALL, "prototype_small", knee=8)}}))
    k1 = d["widths"][0]["per_k"][1]
    assert k1["learnable"] is False
    assert "2 of the required 20 evaluations" in k1["criterion_2_note"]
    assert d["width_decision"]["outcome"] == SELECT   # K=8 still carries the width
    assert set(d["budget_decision"]["per_k"]) == {8}


def test_headline_from_the_real_builder_is_accepted(tmp_path):
    """Every headline field D5 reads is produced by phase2.runner._headline."""
    from phase2.runner import _headline

    result = {
        "model": {"parameter_count": SMALL, "forward_flops_per_sequence": 54330368},
        "block_calls_per_forward": 8, "steps_run": MAX_STEPS, "best_step": MAX_STEPS,
        "best_val": {"chance_normalised_token_accuracy": 0.5, "token_accuracy": 50.0, "exact_match": 1.0},
        "final_val": {"chance_normalised_token_accuracy": 0.5, "token_accuracy": 50.0, "exact_match": 1.0},
        "best_val_loss": GOOD_LOSS, "final_val_loss": GOOD_LOSS,
    }
    headline = _headline(result)
    for field in ("parameter_count", "final_val_chance_normalised", "final_val_loss"):
        assert field in headline, f"_headline must provide {field!r}"

    run_dict = {"model_base": "prototype_small", "curve": curve(flat(0.5)), "headline": headline}
    assert evaluate_d5(report({"small": {"1": run_dict, "8": run_dict}}))["width_decision"]["outcome"] == SELECT


def test_diverged_headline_from_the_real_builder_is_accepted():
    """_headline on a diverged result yields nulls, not missing keys."""
    from phase2.runner import _headline

    headline = _headline({
        "model": {"parameter_count": SMALL, "forward_flops_per_sequence": 1},
        "block_calls_per_forward": 8, "steps_run": 3, "best_step": None,
        "best_val": None, "final_val": None, "best_val_loss": None, "final_val_loss": None,
    })
    assert headline["final_val_chance_normalised"] is None and headline["final_val_loss"] is None
    run_dict = {"model_base": "prototype_small", "curve": curve(flat(0.5)), "headline": headline}
    k = evaluate_d5(report({"small": {"1": run_dict, "8": run_dict}}))["widths"][0]["per_k"][1]
    assert k["learnable"] is False


def test_settings_keys_match_the_real_diagnostic_writer():
    """Guard against D5 reading a settings field the real report never writes."""
    import inspect

    from phase2 import runner

    source = inspect.getsource(runner.run_capacity_diagnostic)
    for key in ("t", "k_values", "max_steps", "eval_every", "floor_threshold",
                "chance_level_val_loss_ln_vocab"):
        assert f'"{key}"' in source, f"settings.{key} is read by D5 but not written by run_capacity_diagnostic"
    for key in ("runs", "model_configs", "single_seed_note", "curve", "headline", "model_base"):
        assert key in source, f"report field {key!r} is read by D5 but not written by run_capacity_diagnostic"


# ---------------------------------------------------------------------------
# Purity: no mutation, no writes
# ---------------------------------------------------------------------------


def test_no_d5_function_writes_anything(monkeypatch, tmp_path):
    """Any attempt to open a file for writing during evaluation fails the test."""
    import builtins
    import io
    import os

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"evaluate_d5 attempted to write to {file!r} (mode {mode!r})")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: pytest.fail("evaluate_d5 attempted to create a directory"))

    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    d = evaluate_d5(r)
    assert render_d5_decision(d)
    assert isinstance(io.StringIO(), io.StringIO)  # sanity: monkeypatch did not break the module


def test_evaluate_d5_does_not_mutate_nested_structures():
    """Deep equality before and after, not just top-level identity."""
    import copy

    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    untouched = copy.deepcopy(r)
    evaluate_d5(r)
    assert r == untouched


def test_decision_does_not_alias_the_input_report():
    """Mutating the returned decision must not reach back into the report."""
    r = report({"small": {"1": learnable_run(SMALL, "prototype_small"),
                          "8": learnable_run(SMALL, "prototype_small")}})
    d = evaluate_d5(r)
    d["widths"][0]["per_k"][8]["criteria"]["1_final_chance_normalised_at_least_0.02"] = "tampered"
    assert r["runs"]["small"]["8"]["headline"]["final_val_chance_normalised"] == 0.5


def test_evaluator_module_imports_no_heavy_or_stateful_dependencies():
    import inspect

    from phase2 import p1b_decision

    source = inspect.getsource(p1b_decision)
    for banned in ("import torch", "import numpy", "import random", "from phase2.runner", "from phase2.training"):
        assert banned not in source, f"{banned!r} would break the evaluator's purity"
