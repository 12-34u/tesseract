"""Amendment 02 D5 — width and budget selection from a completed P1b report.

This module is a pure, deterministic evaluator of the rules recorded in
``configs/phase2/amendment_02_draft.yaml`` (decision D5). It reads a P1b
``diagnostic_report.json`` and returns the width and budget those rules imply.

It never trains, never writes, never reads a checkpoint, and never touches a
run directory. ``evaluate_d5`` takes a already-parsed report dict and depends on
nothing but the standard library, so the same input always produces the same
output.

The pre-registered rules, quoted from D5:

    Learnable means some K has final chance-normalised validation accuracy
    >= 0.02, a mean over the last 20 evaluations >= 0.02, and final validation
    loss < ln 60. Width: the smallest learnable width, stating whether K=1 is at
    the floor. Budget: the smallest multiple of 5,000 steps at which the
    8-evaluation rolling mean reaches 90 % of its final value; or 20,000 marked
    "not converged" if still rising. If neither width is learnable, do not
    proceed.

Two points in that text need a reading, and both are resolved here without
introducing a threshold or a preference:

* **"still rising"** is not given a separate test, and none is invented. The
  budget search already returns the smallest qualifying multiple of 5,000. If
  that smallest value is the last observed step, the curve never reached 90 % of
  its final level earlier, which is exactly what "still rising at 20,000" means.
  So ``not_converged`` is set when the selected budget equals ``max_steps``.
  Nothing is extrapolated past the observed steps.

* **"the relevant learnable run"** is singular, but a width may have more than
  one learnable K. Rather than pick one, the budget is computed for *every*
  learnable K and the recommendation is their **maximum**, because a budget that
  is short for any learnable configuration would under-train it. Every per-K
  budget is reported, and ``k_budgets_agree`` says whether the choice mattered.
  This aggregation is the one place the evaluator goes beyond D5's literal text;
  it is deterministic and flagged in the rendered output.

P1b uses a single seed, so every number here is descriptive of one run per cell.
That fact is carried through to the output and is never silently dropped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# --- Pre-registered constants. Do not change without amending D5. -----------
MIN_CHANCE_NORMALISED = 0.02   # criteria 1 and 2
FINAL_EVALUATIONS = 20         # criterion 2 window, in evaluations
FINAL_EVALUATION_STEPS = 5000  # the same window expressed in steps
ROLLING_WINDOW = 8             # budget rule: evaluations per rolling mean
BUDGET_FRACTION = 0.90         # budget rule: fraction of the final rolling mean
BUDGET_STEP_MULTIPLE = 5000    # budget rule: candidate steps
EXPECTED_VOCAB_SIZE = 60       # criterion 3 uses ln(60)
LN_VOCAB = math.log(EXPECTED_VOCAB_SIZE)
EXPECTED_T = 8                 # D5 is stated for the T = 8 diagnostic

CURVE_SCORE = "val_chance_normalised_token_accuracy"

DO_NOT_PROCEED = "DO_NOT_PROCEED"
SELECT = "SELECT"


class D5DataError(ValueError):
    """The P1b report is absent, malformed, or incomplete.

    Raised instead of returning a decision. A missing or mistyped *key* is
    malformation and raises. A key that is present but null because the run
    diverged is data, not malformation: that K simply fails the criterion and
    the reason is recorded.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D5DataError(message)


def _number(container: Mapping, key: str, where: str) -> Optional[float]:
    """Read a numeric field that is allowed to be null (a diverged run)."""
    _require(key in container, f"{where}: missing {key!r}")
    value = container[key]
    if value is None:
        return None
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{where}.{key}: expected a number or null, got {value!r}")
    return float(value)


def _required_number(container: Mapping, key: str, where: str) -> float:
    value = _number(container, key, where)
    _require(value is not None, f"{where}.{key} must not be null")
    return value


# ============================================================================
# Structural validation
# ============================================================================


def _validate(report: Mapping) -> Dict:
    _require(isinstance(report, Mapping), f"diagnostic report must be a mapping, got {type(report).__name__}")
    for key in ("settings", "runs", "model_configs", "single_seed_note"):
        _require(key in report, f"diagnostic report: missing {key!r}")

    settings = report["settings"]
    _require(isinstance(settings, Mapping), "settings must be a mapping")
    for key in ("t", "k_values", "max_steps", "eval_every", "seed", "floor_threshold",
                "chance_level_val_loss_ln_vocab"):
        _require(key in settings, f"settings: missing {key!r}")

    t = settings["t"]
    _require(t == EXPECTED_T, f"D5 is stated for the T={EXPECTED_T} diagnostic; this report is T={t}")

    ln_vocab = _required_number(settings, "chance_level_val_loss_ln_vocab", "settings")
    _require(abs(ln_vocab - LN_VOCAB) < 1e-9,
             f"settings.chance_level_val_loss_ln_vocab is {ln_vocab!r}; D5's criterion 3 is ln({EXPECTED_VOCAB_SIZE}) "
             f"= {LN_VOCAB!r}. The benchmark vocabulary does not match the pre-registered rule.")

    k_values = settings["k_values"]
    _require(isinstance(k_values, Sequence) and not isinstance(k_values, str) and len(k_values) > 0,
             "settings.k_values must be a non-empty list")
    k_values = [int(k) for k in k_values]

    max_steps = _required_number(settings, "max_steps", "settings")
    eval_every = _required_number(settings, "eval_every", "settings")
    _require(max_steps > 0 and eval_every > 0, "settings.max_steps and settings.eval_every must be positive")
    _require(FINAL_EVALUATIONS * eval_every == FINAL_EVALUATION_STEPS,
             f"D5's criterion 2 window is the last {FINAL_EVALUATIONS} evaluations, i.e. the last "
             f"{FINAL_EVALUATION_STEPS} steps. This report evaluates every {eval_every:g} steps, so "
             f"{FINAL_EVALUATIONS} evaluations span {FINAL_EVALUATIONS * eval_every:g} steps and the two halves "
             "of the pre-registered definition disagree.")

    runs = report["runs"]
    _require(isinstance(runs, Mapping) and runs, "runs must be a non-empty mapping of width name -> K -> report")
    for width, per_k in runs.items():
        _require(isinstance(per_k, Mapping), f"runs.{width} must be a mapping of K -> report")
        present = {int(k) for k in per_k}
        missing = sorted(set(k_values) - present)
        _require(not missing, f"runs.{width}: incomplete P1b data, missing K {missing}")

    return {"t": int(t), "k_values": k_values, "max_steps": int(max_steps), "eval_every": int(eval_every),
            "seed": settings["seed"], "floor_threshold": _required_number(settings, "floor_threshold", "settings"),
            "ln_vocab": ln_vocab}


def _curve_evaluations(run: Mapping, where: str) -> List[Dict[str, float]]:
    """The validation *evaluations* in a run's curve, in step order.

    Not every row of a real ``val_curve.csv`` is an evaluation. When training
    stops on a non-finite loss, ``phase2.training`` appends one final row
    carrying only the step and the offending train loss, and ``_read_curve``
    drops the absent validation columns. Such a row is a divergence marker, not
    an evaluation, so it is excluded here rather than treated as malformed
    data: a P1b run that diverged is a legitimate observation, and D5 answers it
    by finding that K not learnable.

    Every row must still carry a numeric ``step``; that is structure, and a
    report without it is malformed.
    """
    _require("curve" in run, f"{where}: missing 'curve'")
    curve = run["curve"]
    _require(isinstance(curve, Sequence) and not isinstance(curve, str), f"{where}.curve must be a list")
    steps, evaluations = [], []
    for i, row in enumerate(curve):
        _require(isinstance(row, Mapping), f"{where}.curve[{i}] must be a mapping")
        step = _required_number(row, "step", f"{where}.curve[{i}]")
        steps.append(step)
        if CURVE_SCORE not in row:
            continue  # not an evaluation (e.g. the non-finite-loss marker row)
        score = _number(row, CURVE_SCORE, f"{where}.curve[{i}]")
        if score is None:
            continue
        evaluations.append({"step": step, "score": score})
    _require(steps == sorted(steps), f"{where}.curve must be ordered by step")
    return evaluations


# ============================================================================
# LEARNABLE(width)
# ============================================================================


def _evaluate_k(run: Mapping, where: str) -> Dict:
    """The three D5 criteria for one K. All three must hold for this K to count."""
    _require("headline" in run, f"{where}: missing 'headline'")
    headline = run["headline"]
    _require(isinstance(headline, Mapping), f"{where}.headline must be a mapping")

    final_score = _number(headline, "final_val_chance_normalised", f"{where}.headline")
    final_loss = _number(headline, "final_val_loss", f"{where}.headline")
    points = _curve_evaluations(run, where)

    # Criterion 1: final-checkpoint chance-normalised validation accuracy.
    c1 = final_score is not None and final_score >= MIN_CHANCE_NORMALISED

    # Criterion 2: mean over the final 20 evaluations.
    tail = points[-FINAL_EVALUATIONS:]
    if len(tail) < FINAL_EVALUATIONS:
        c2, tail_mean, tail_note = False, None, (
            f"only {len(points)} of the required {FINAL_EVALUATIONS} evaluations are present")
    else:
        tail_mean = sum(p["score"] for p in tail) / len(tail)
        c2, tail_note = tail_mean >= MIN_CHANCE_NORMALISED, None

    # Criterion 3: final validation loss strictly below ln(60).
    c3 = final_loss is not None and final_loss < LN_VOCAB

    return {
        "final_val_chance_normalised": final_score,
        "final_evaluations_mean": tail_mean,
        "final_evaluations_span_steps": [tail[0]["step"], tail[-1]["step"]] if tail else None,
        "final_val_loss": final_loss,
        "criteria": {
            "1_final_chance_normalised_at_least_0.02": c1,
            "2_final_20_evaluation_mean_at_least_0.02": c2,
            "3_final_val_loss_below_ln_60": c3,
        },
        "criterion_2_note": tail_note,
        "learnable": c1 and c2 and c3,
    }


# ============================================================================
# BUDGET
# ============================================================================


def _rolling_means(points: Sequence[Mapping]) -> List[Optional[float]]:
    """Trailing mean of ROLLING_WINDOW scores, aligned to each evaluation."""
    means: List[Optional[float]] = []
    for i in range(len(points)):
        window = points[max(0, i - ROLLING_WINDOW + 1): i + 1]
        means.append(sum(p["score"] for p in window) / len(window) if len(window) == ROLLING_WINDOW else None)
    return means


def _budget_for_run(run: Mapping, where: str, max_steps: int) -> Dict:
    """Smallest multiple of 5,000 whose rolling mean reaches 90 % of the final rolling mean."""
    points = _curve_evaluations(run, where)
    means = _rolling_means(points)
    final_mean = means[-1] if means else None
    if final_mean is None:
        return {"recommended_max_steps": max_steps, "not_converged": True, "final_rolling_mean": None,
                "target": None, "candidates": [],
                "reason": f"fewer than {ROLLING_WINDOW} usable evaluations at the end of the curve; "
                          "no level-off can be observed, so the full observed budget is recommended"}

    target = BUDGET_FRACTION * final_mean
    candidates = []
    for point, mean in zip(points, means):
        step = int(point["step"])
        if step % BUDGET_STEP_MULTIPLE == 0 and step <= max_steps and mean is not None:
            candidates.append({"step": step, "rolling_mean": mean, "reaches_target": mean >= target})

    qualifying = [c["step"] for c in candidates if c["reaches_target"]]
    if qualifying:
        chosen = min(qualifying)
        # "Still rising at max_steps" is exactly the case where nothing earlier qualified.
        not_converged = chosen >= max_steps
        reason = (f"rolling mean reached {BUDGET_FRACTION:.0%} of its final value first at step {chosen}"
                  if not not_converged else
                  f"rolling mean only reached {BUDGET_FRACTION:.0%} of its final value at the last observed step")
    else:
        chosen, not_converged = max_steps, True
        reason = (f"no multiple of {BUDGET_STEP_MULTIPLE} reached {BUDGET_FRACTION:.0%} of the final rolling mean; "
                  "the full observed budget is recommended")

    return {"recommended_max_steps": chosen, "not_converged": not_converged, "final_rolling_mean": final_mean,
            "target": target, "candidates": candidates, "reason": reason}


# ============================================================================
# Evaluator
# ============================================================================


def evaluate_d5(report: Mapping) -> Dict:
    """Apply Amendment 02 D5 to a completed P1b diagnostic report.

    Pure: no I/O, no randomness, no global state. Raises :class:`D5DataError`
    if the report is malformed or incomplete, rather than returning a decision.
    """
    settings = _validate(report)
    runs, max_steps = report["runs"], settings["max_steps"]

    widths = []
    for name in sorted(runs):
        per_k = runs[name]
        ks = sorted(int(k) for k in per_k)
        by_k, sizes, bases = {}, set(), set()
        for k in ks:
            run = per_k[str(k)] if str(k) in per_k else per_k[k]
            where = f"runs.{name}.{k}"
            _require(isinstance(run, Mapping), f"{where} must be a mapping")
            by_k[k] = _evaluate_k(run, where)
            headline = run["headline"]
            sizes.add(int(_required_number(headline, "parameter_count", f"{where}.headline")))
            if "model_base" in run:
                bases.add(run["model_base"])
        _require(len(sizes) == 1,
                 f"runs.{name}: parameter count differs across K ({sorted(sizes)}); a Tesseract width must have the "
                 "same parameter count at every K")
        widths.append({
            "width": name,
            "model_base": sorted(bases)[0] if len(bases) == 1 else None,
            "parameter_count": sizes.pop(),
            "per_k": by_k,
            "learnable_k": [k for k in ks if by_k[k]["learnable"]],
            "learnable": any(by_k[k]["learnable"] for k in ks),
        })

    counts = [w["parameter_count"] for w in widths]
    _require(len(set(counts)) == len(counts),
             f"two widths have the same parameter count ({sorted(counts)}); 'the smallest learnable width' is not "
             "defined and the evaluator will not choose between them")
    widths.sort(key=lambda w: w["parameter_count"])

    learnable = [w for w in widths if w["learnable"]]
    decision: Dict = {
        "rule": "Amendment 02 D5 (width and budget selection)",
        "evaluator": "phase2.p1b_decision.evaluate_d5",
        "thresholds": {
            "min_chance_normalised": MIN_CHANCE_NORMALISED,
            "final_evaluations": FINAL_EVALUATIONS,
            "final_evaluation_steps": FINAL_EVALUATION_STEPS,
            "final_val_loss_below": LN_VOCAB,
            "rolling_window_evaluations": ROLLING_WINDOW,
            "budget_fraction_of_final": BUDGET_FRACTION,
            "budget_step_multiple": BUDGET_STEP_MULTIPLE,
        },
        "p1b": {"t": settings["t"], "seed": settings["seed"], "k_values": settings["k_values"],
                "max_steps": max_steps, "eval_every": settings["eval_every"],
                "floor_threshold": settings["floor_threshold"],
                "single_seed_note": report["single_seed_note"]},
        "widths": widths,
        "caveats": [
            "P1b uses a single seed per cell: every quantity here is descriptive of one run, not a statistical claim.",
            "Selection is advisory. The width and the budget are approved by the researcher before Amendment 02 is "
            "frozen; this evaluator only applies the pre-registered rule to the observed data.",
            f"Nothing is inferred beyond the {max_steps} observed steps.",
        ],
    }

    if not learnable:
        decision["width_decision"] = {
            "outcome": DO_NOT_PROCEED,
            "selected_width": None,
            "reason": "no width has any K satisfying all three D5 learnability criteria at "
                      f"T={settings['t']}; D5 says do not proceed",
        }
        decision["budget_decision"] = {
            "outcome": DO_NOT_PROCEED, "recommended_max_steps": None, "not_converged": None,
            "reason": "no width was selected, so no budget is recommended",
        }
        return decision

    selected = learnable[0]
    k_low = min(settings["k_values"])
    k_low_report = selected["per_k"].get(k_low)
    k_low_score = k_low_report["final_val_chance_normalised"] if k_low_report else None
    decision["width_decision"] = {
        "outcome": SELECT,
        "selected_width": selected["width"],
        "model_base": selected["model_base"],
        "parameter_count": selected["parameter_count"],
        "learnable_k": selected["learnable_k"],
        "reason": f"smallest width with at least one learnable K (K={selected['learnable_k']})",
        "k_low": k_low,
        "k_low_final_chance_normalised": k_low_score,
        "k_low_at_floor": None if k_low_score is None else k_low_score < settings["floor_threshold"],
        "k_low_learnable": bool(k_low_report and k_low_report["learnable"]),
        "rejected_widths": [{"width": w["width"], "parameter_count": w["parameter_count"], "learnable": w["learnable"]}
                            for w in widths if w["width"] != selected["width"]],
    }

    per_k_budget = {}
    for k in selected["learnable_k"]:
        run = runs[selected["width"]][str(k)] if str(k) in runs[selected["width"]] else runs[selected["width"]][k]
        per_k_budget[k] = _budget_for_run(run, f"runs.{selected['width']}.{k}", max_steps)

    recommended = max(b["recommended_max_steps"] for b in per_k_budget.values())
    agree = len({b["recommended_max_steps"] for b in per_k_budget.values()}) == 1
    decision["budget_decision"] = {
        "outcome": SELECT,
        "recommended_max_steps": recommended,
        "not_converged": any(b["not_converged"] for b in per_k_budget.values() if
                             b["recommended_max_steps"] == recommended),
        "per_k": per_k_budget,
        "k_budgets_agree": agree,
        "aggregation": "maximum over the learnable K at the selected width, so the budget is not short for any "
                       "learnable configuration (D5 names a single run; this is the only place the evaluator goes "
                       "beyond its literal text)",
    }
    return decision


def evaluate_d5_file(path) -> Dict:
    """Read a P1b ``diagnostic_report.json`` and evaluate D5 over it.

    A thin wrapper: the decision itself is made by the pure :func:`evaluate_d5`.
    A missing file raises rather than producing a decision.
    """
    path = Path(path)
    if not path.is_file():
        raise D5DataError(f"No P1b diagnostic report at {path}; D5 cannot be evaluated and no width or budget "
                          "may be selected.")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise D5DataError(f"{path} is not valid JSON: {exc}") from exc
    return evaluate_d5(report)


# ============================================================================
# Human-readable output
# ============================================================================


def _fmt(value, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def render_d5_decision(decision: Mapping) -> str:
    """Plain-text D5 decision, suitable for attaching to the P1b report."""
    width, budget = decision["width_decision"], decision["budget_decision"]
    p1b, thresholds = decision["p1b"], decision["thresholds"]
    lines = [
        "AMENDMENT 02 D5 — WIDTH AND BUDGET SELECTION",
        "=" * 72,
        f"Source: P1b diagnostic at T={p1b['t']}, seed {p1b['seed']}, K={p1b['k_values']}, "
        f"{p1b['max_steps']} steps, evaluated every {p1b['eval_every']}.",
        "",
        "Learnability criteria (all three, for at least one K):",
        f"  1. final validation chance-normalised accuracy >= {thresholds['min_chance_normalised']}",
        f"  2. mean over the final {thresholds['final_evaluations']} evaluations "
        f"(last {thresholds['final_evaluation_steps']} steps) >= {thresholds['min_chance_normalised']}",
        f"  3. final validation loss < ln(60) = {thresholds['final_val_loss_below']:.6f}",
        "",
    ]

    for w in decision["widths"]:
        lines.append(f"{w['width']} ({w['model_base'] or 'unknown base'}, {w['parameter_count']:,} params) — "
                     f"{'LEARNABLE' if w['learnable'] else 'NOT LEARNABLE'}")
        for k in sorted(w["per_k"]):
            r = w["per_k"][k]
            marks = "".join("✓" if ok else "✗" for ok in r["criteria"].values())
            note = f"  [{r['criterion_2_note']}]" if r["criterion_2_note"] else ""
            lines.append(f"    K={k:<2} {marks}  final {_fmt(r['final_val_chance_normalised'])}, "
                         f"last-{thresholds['final_evaluations']} mean {_fmt(r['final_evaluations_mean'])}, "
                         f"final loss {_fmt(r['final_val_loss'])} "
                         f"-> {'learnable' if r['learnable'] else 'not learnable'}{note}")
        lines.append("")

    lines.append(f"WIDTH: {width['outcome']}")
    if width["outcome"] == DO_NOT_PROCEED:
        lines.append(f"  {width['reason']}")
    else:
        lines.append(f"  selected: {width['selected_width']} ({width['model_base']}, "
                     f"{width['parameter_count']:,} params)")
        lines.append(f"  reason:   {width['reason']}")
        floor = width["k_low_at_floor"]
        state = "unknown (no final score)" if floor is None else ("AT FLOOR" if floor else "above the floor")
        lines.append(f"  K={width['k_low']} at the selected width: {state} "
                     f"(final {_fmt(width['k_low_final_chance_normalised'])} vs floor {p1b['floor_threshold']})")
        for other in width["rejected_widths"]:
            lines.append(f"  not selected: {other['width']} ({other['parameter_count']:,} params, "
                         f"{'learnable' if other['learnable'] else 'not learnable'})")

    lines += ["", f"BUDGET: {budget['outcome']}"]
    if budget["outcome"] == DO_NOT_PROCEED:
        lines.append(f"  {budget['reason']}")
    else:
        flag = "  [NOT CONVERGED — still rising at the last observed step]" if budget["not_converged"] else ""
        lines.append(f"  recommended max_steps: {budget['recommended_max_steps']}{flag}")
        for k in sorted(budget["per_k"]):
            b = budget["per_k"][k]
            lines.append(f"    K={k:<2} -> {b['recommended_max_steps']}"
                         f"{' (not converged)' if b['not_converged'] else ''}: {b['reason']}")
        if not budget["k_budgets_agree"]:
            lines.append(f"    note: per-K budgets differ; {budget['aggregation']}")

    lines += ["", "Caveats:"] + [f"  - {c}" for c in decision["caveats"]]
    lines += ["", "This is an advisory application of a pre-registered rule. Amendment 02 is not frozen by it, and "
              "no experiment is started by it."]
    return "\n".join(lines)
