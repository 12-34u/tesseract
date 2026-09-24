"""Mechanical application of the pre-registered P1c probe decision rule.

The rule is fixed in PHASE2_PROBE_PREREGISTRATION.md and
configs/phase2/probe_preregistration.yaml, both committed before either probe
ran. Nothing here reads a threshold from a result: the thresholds come from the
pre-registration, and the classification is a pure function of the recorded
numbers, so it cannot be steered after the fact.

Probe records carry the validation numbers the rule needs:

    {"seed": int,
     "best_val":  {"token_accuracy": %, "loss": nats},
     "final_val": {"token_accuracy": %, "loss": nats},
     "num_evaluations": int,          # M, for the multiplicity correction
     "per_example_token_accuracy_se": float}   # SE in accuracy points
"""

import math
from typing import Dict, List, Optional, Sequence

LEARNABLE = "LEARNABLE"
PARTIAL = "PARTIAL"
NOT_LEARNABLE = "NOT_LEARNABLE"

ABOVE_CHANCE_ALPHA = 0.05
ABOVE_CHANCE_LOSS_MARGIN_NATS = 0.01


def _inverse_normal_cdf(p: float) -> float:
    """Phi^-1 via the Acklam rational approximation (|error| < 1.15e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        return -_inverse_normal_cdf(1 - p)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def above_chance_threshold(chance_percent: float, se_points: float, num_evaluations: int) -> float:
    """Chance + z*SE, with z corrected for choosing the best of M evaluations.

    The primary checkpoint is the maximum over M validation evaluations, so an
    uncorrected bar would be crossed by selection noise alone.
    """
    m = max(1, int(num_evaluations))
    z = _inverse_normal_cdf(1 - ABOVE_CHANCE_ALPHA / m)
    return chance_percent + z * se_points


def evaluate_seed(record: Dict, chance_percent: float, chance_loss_nats: float,
                  accuracy_bar: float, loss_bar: float) -> Dict:
    """Apply both bars to one seed. Returns the evidence, not just the verdict."""
    best = record.get("best_val") or {}
    final = record.get("final_val") or {}
    accuracy = best.get("token_accuracy")
    best_loss = best.get("loss")
    final_loss = final.get("loss")

    meets_accuracy = accuracy is not None and accuracy >= accuracy_bar
    candidate_losses = [v for v in (best_loss, final_loss) if v is not None]
    meets_loss = bool(candidate_losses) and min(candidate_losses) <= loss_bar

    threshold = above_chance_threshold(chance_percent, record.get("per_example_token_accuracy_se") or 0.0,
                                       record.get("num_evaluations") or 1)
    above_by_accuracy = accuracy is not None and accuracy > threshold
    above_by_loss = final_loss is not None and final_loss <= chance_loss_nats - ABOVE_CHANCE_LOSS_MARGIN_NATS

    return {
        "seed": record.get("seed"),
        "best_val_token_accuracy": accuracy,
        "best_val_loss": best_loss,
        "final_val_loss": final_loss,
        "meets_learnable_accuracy_bar": meets_accuracy,
        "meets_learnable_loss_bar": meets_loss,
        "learnable": meets_accuracy or meets_loss,
        "learnable_by_loss_arm_only": meets_loss and not meets_accuracy,
        "above_chance_threshold_percent": threshold,
        "above_chance_by_accuracy": above_by_accuracy,
        "above_chance_by_loss": above_by_loss,
        "above_chance": above_by_accuracy or above_by_loss,
    }


def classify(records: Sequence[Dict], preregistration: Dict) -> Dict:
    """LEARNABLE / PARTIAL / NOT LEARNABLE, exactly as pre-registered."""
    rule = preregistration["rule"]
    chance = preregistration["chance"]
    accuracy_bar = rule["learnable"]["accuracy_percent_at_least"]
    loss_bar = rule["learnable"]["loss_nats_at_most"]
    seeds_required = rule["learnable"]["seeds_required"]

    per_seed = [evaluate_seed(r, chance["token_accuracy_percent"], chance["loss_nats"], accuracy_bar, loss_bar)
                for r in records]
    learnable_seeds = [s["seed"] for s in per_seed if s["learnable"]]
    above_chance_seeds = [s["seed"] for s in per_seed if s["above_chance"]]

    if len(learnable_seeds) >= seeds_required:
        classification = LEARNABLE
    elif above_chance_seeds:
        classification = PARTIAL
    else:
        classification = NOT_LEARNABLE

    return {
        "preregistration_id": preregistration["id"],
        "preregistration_date": preregistration["date"],
        "classification": classification,
        "bars": {"accuracy_percent_at_least": accuracy_bar, "loss_nats_at_most": loss_bar,
                 "seeds_required": seeds_required},
        "seeds_meeting_learnable_bar": learnable_seeds,
        "seeds_above_chance": above_chance_seeds,
        "learnable_by_loss_arm_only": [s["seed"] for s in per_seed if s["learnable_by_loss_arm_only"]],
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
    }
