#!/usr/bin/env python3
"""Takeoff steps and final slope for every Batch 02 cell, from the recorded curves.

Reads `detail[].curve` out of the Batch 02 report and computes nothing that is
not already in it. No model is loaded, no training step is run, and no new
number is generated: this is a reading of recorded evaluations, so it cannot
constitute a second attempt at anything.

Three quantities per cell:

**Takeoff steps.** The first evaluation at which T = 3 validation token accuracy
crosses the pre-registered LEARNABLE bar (6.6667 %) and the T = 3 shortcut
ceiling (26.6257 %, `PHASE2_BATCH_02_PREREG.md` §4.1). Crossing the second one
is what a shortcut-only model could not do, so the step at which it happens is
the step from which the cell's accuracy is no longer attributable to the shallow
candidate.

**Final slope.** Least-squares slope of accuracy against step over the last
`--window` steps, in accuracy points per 1,000 steps. This is the quantity that
distinguishes a plateau from a budget limit. Reported with the window's first
and last accuracy so the fit can be checked by eye against the endpoints.

**Whether it was still rising.** The slope alone is not enough — a slope can be
positive and negligible. A cell counts as still rising only if the slope exceeds
`--rising-threshold` points per 1,000 steps, which is stated as a flag rather
than left to the reader.

This is descriptive. It is not part of any classification and does not touch the
pre-registered outcome table.

    python scripts/phase2_batch02_curves.py [--report PATH] [--window 2500]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from utils.run_artifacts import write_json  # noqa: E402

LEARNABLE_BAR_PERCENT = 100 / 60 + 5.0            # 6.6667 %, the pre-registered bar
DEFAULT_REPORT_GLOB = "runs/phase2/batch02/t3_curriculum*/t3_curriculum_report.json"


def first_crossing(curve, threshold: float):
    """First evaluation whose accuracy is at or above ``threshold``."""
    for row in curve:
        accuracy = row.get("val_token_accuracy")
        if accuracy is not None and accuracy >= threshold:
            return {"step": row["step"], "token_accuracy": accuracy}
    return None


def final_slope(curve, window: int):
    """Least-squares slope over the last ``window`` steps, in points per 1,000 steps."""
    if not curve:
        return None
    last_step = curve[-1]["step"]
    points = [(r["step"], r["val_token_accuracy"]) for r in curve
              if r.get("val_token_accuracy") is not None and r["step"] >= last_step - window]
    if len(points) < 3:
        return None
    steps = np.array([p[0] for p in points], dtype=float)
    accuracy = np.array([p[1] for p in points], dtype=float)
    slope = float(np.polyfit(steps, accuracy, 1)[0]) * 1000.0
    return {
        "window_steps": window,
        "num_points": len(points),
        "first_step": int(steps[0]), "first_accuracy": float(accuracy[0]),
        "last_step": int(steps[-1]), "last_accuracy": float(accuracy[-1]),
        "slope_points_per_1000_steps": slope,
        "endpoint_delta_points": float(accuracy[-1] - accuracy[0]),
    }


#: Token accuracy of the G2-failing shallow candidate on the T=3 validation
#: split, from PHASE2_BATCH_02_PREREG.md §4.1. A property of the split, fixed
#: before the probe ran, so it is a constant here and not read from a result.
SHORTCUT_CEILING_PERCENT = 26.6257


def analyse(report: dict, window: int, rising_threshold: float) -> dict:
    ceiling = SHORTCUT_CEILING_PERCENT
    rows = []
    for cell in report.get("detail", []):
        curve = cell.get("curve", [])
        slope = final_slope(curve, window)
        rows.append({
            "k": cell["k"], "seed": cell["seed"],
            "best_val_token_accuracy": max((r["val_token_accuracy"] for r in curve
                                            if r.get("val_token_accuracy") is not None), default=None),
            "final_val_token_accuracy": curve[-1]["val_token_accuracy"] if curve else None,
            "shortcut_agreement_percent": cell.get("best_val_shortcut_agreement_percent"),
            "distinct_predicted_tokens": cell.get("best_val_distinct_predicted_tokens"),
            "crossed_learnable_bar": first_crossing(curve, LEARNABLE_BAR_PERCENT),
            "crossed_shortcut_ceiling": first_crossing(curve, ceiling),
            "final_slope": slope,
            "still_rising": bool(slope and slope["slope_points_per_1000_steps"] > rising_threshold),
        })

    by_k = {}
    for k in sorted({r["k"] for r in rows}):
        cells = [r for r in rows if r["k"] == k]
        crossed = [c for c in cells if c["crossed_shortcut_ceiling"]]
        by_k[str(k)] = {
            "cells": len(cells),
            "cells_crossing_learnable_bar": sum(1 for c in cells if c["crossed_learnable_bar"]),
            "cells_crossing_shortcut_ceiling": len(crossed),
            "mean_best_accuracy": float(np.mean([c["best_val_token_accuracy"] for c in cells])),
            "mean_takeoff_step_shortcut_ceiling": (float(np.mean([c["crossed_shortcut_ceiling"]["step"]
                                                                  for c in crossed])) if crossed else None),
            "mean_final_slope_points_per_1000_steps": float(np.mean(
                [c["final_slope"]["slope_points_per_1000_steps"] for c in cells if c["final_slope"]])),
            "cells_still_rising": sum(1 for c in cells if c["still_rising"]),
        }

    return {
        "source_report": report.get("_path"),
        "target_t": report.get("target_t"),
        "learnable_bar_percent": LEARNABLE_BAR_PERCENT,
        "shortcut_ceiling_percent": ceiling,
        "rising_threshold_points_per_1000_steps": rising_threshold,
        "descriptive_only": True,
        "note": ("Read from recorded curves. Not part of any classification; the pre-registered "
                 "outcome table governs. The K comparison here is descriptive, and a curriculum "
                 "probe is not a grid cell."),
        "per_cell": rows,
        "by_k": by_k,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=None)
    parser.add_argument("--window", type=int, default=2500)
    parser.add_argument("--rising-threshold", type=float, default=0.1,
                        help="accuracy points per 1,000 steps above which a cell counts as still rising")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = args.report or (sorted(glob.glob(DEFAULT_REPORT_GLOB))[-1]
                           if glob.glob(DEFAULT_REPORT_GLOB) else None)
    if path is None:
        print(f"no Batch 02 report found at {DEFAULT_REPORT_GLOB}", file=sys.stderr)
        return 1
    report = json.loads(Path(path).read_text())
    report["_path"] = str(path)
    result = analyse(report, args.window, args.rising_threshold)

    print(f"source: {path}")
    print(f"bars: LEARNABLE {LEARNABLE_BAR_PERCENT:.4f} %   shortcut ceiling "
          f"{result['shortcut_ceiling_percent']:.4f} %   window {args.window} steps\n")
    header = f"{'K':>3} {'seed':>4} {'best %':>8} {'final %':>8} {'sc %':>6} " \
             f"{'>6.67% at':>10} {'>26.63% at':>11} {'slope/1k':>9} {'rising':>7}"
    print(header)
    print("-" * len(header))
    for r in result["per_cell"]:
        bar = r["crossed_learnable_bar"]
        ceil = r["crossed_shortcut_ceiling"]
        slope = r["final_slope"]
        slope_text = f"{slope['slope_points_per_1000_steps']:+.3f}" if slope else "n/a"
        print(f"{r['k']:>3} {r['seed']:>4} {r['best_val_token_accuracy']:>8.4f} "
              f"{r['final_val_token_accuracy']:>8.4f} "
              f"{(r['shortcut_agreement_percent'] or 0):>6.2f} "
              f"{(str(bar['step']) if bar else 'never'):>10} "
              f"{(str(ceil['step']) if ceil else 'never'):>11} "
              f"{slope_text:>9} "
              f"{('yes' if r['still_rising'] else 'no'):>7}")
    print()
    for k, agg in result["by_k"].items():
        takeoff = agg["mean_takeoff_step_shortcut_ceiling"]
        print(f"  K={k}: mean best {agg['mean_best_accuracy']:.4f} %  "
              f"crossed bar {agg['cells_crossing_learnable_bar']}/{agg['cells']}  "
              f"crossed ceiling {agg['cells_crossing_shortcut_ceiling']}/{agg['cells']}  "
              f"mean takeoff {('%.0f' % takeoff) if takeoff else '—'}  "
              f"mean slope {agg['mean_final_slope_points_per_1000_steps']:+.3f}/1k  "
              f"still rising {agg['cells_still_rising']}/{agg['cells']}")

    out = Path(args.output) if args.output else Path(path).parent / "curve_analysis.json"
    write_json(out, result)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
