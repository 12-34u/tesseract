#!/usr/bin/env python3
"""Mac CPU diagnostic: untrained per-step hidden-state RMS, per-step norm OFF vs ON.

The question is narrow and structural: with z_L and z_H normalised after every
block call, does the hidden state stay bounded as K grows? Because it involves
no training, it can be answered on a CPU while the GPU box is unreachable.

**This is not an Arm 2c result.** Arm 2c is a training arm, pre-registered to run
on the GPU under Amendment 02a P1. This script only measures the shape of an
untrained model. Its output is labelled `mac_cpu_diagnostic` for that reason and
must not be entered in the Batch 01 report as an arm outcome.

Conditions are chosen to match `scripts/phase2_residual_norms.py` exactly —
medium width, vocab 60, the T=4 validation split, the first 256 examples,
seed 0 before each build — so that the flag-OFF column is directly comparable to
the RMS figures already recorded in PHASE2_BATCH_01_PREREG.md Addendum 01.

Usage:
    TESSERACT_RUNS_DIR=<scratch> python diagnostics/run_per_step_norm_rms.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from diagnostics.per_step_norm import (  # noqa: E402
    PerStepNormConfig,
    apply_per_step_norm,
    growth_ratio,
    hidden_state_rms,
    per_step_norm_parameter_cost,
)
from phase2.config import ModelVariant, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.runner import splits_dir  # noqa: E402
from utils.param_count import count_parameters  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.seed import set_seed  # noqa: E402

K_VALUES = (1, 2, 4, 8)
EXAMPLES = 256
TARGET_T = 4
VOCAB = 60
# Within a run: the stream must not grow step to step. Across K: the state the
# output head sees must not depend on how many steps produced it.
GROWTH_TOLERANCE = 1.05
ACROSS_K_TOLERANCE = 0.05


def _table(rows, title):
    """Markdown table of per-step z_L RMS, one row per condition."""
    width = max(len(r["per_step"]) for r in rows)
    header = " | ".join(str(i) for i in range(1, width + 1))
    lines = [f"**{title}**", "", f"| condition | {header} | growth |",
             "|---" * (width + 2) + "|"]
    for row in rows:
        cells = [f"{s['z_L_rms']:.2f}" for s in row["per_step"]]
        cells += ["—"] * (width - len(cells))
        lines.append(f"| K={row['k']}, flag {row['flag']} | " + " | ".join(cells)
                     + f" | {row['growth_ratio']:.2f}x |")
    return "\n".join(lines)


def main() -> int:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("note: CUDA is available; this diagnostic still runs on CPU by design.",
              file=sys.stderr)

    benchmark = load_benchmark()
    cfg = resolve_model_variant(ModelVariant(name="medium", base="phase2/prototype_medium"), VOCAB)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    val = store.get(benchmark.group, "main", "val", TARGET_T, benchmark.n, benchmark.val_size)
    inputs = val.inputs[:EXAMPLES]
    if len(inputs) != EXAMPLES:
        raise SystemExit(f"expected {EXAMPLES} validation inputs, got {len(inputs)}")

    off = PerStepNormConfig(enabled=False)
    on = PerStepNormConfig(enabled=True)
    # The proposed Addendum 02 variant: no learnable gain or bias, so the bound
    # cannot drift during training and the parameter cost is zero.
    on_non_affine = PerStepNormConfig(enabled=True, elementwise_affine=False)

    rows = []
    for config, flag in ((off, "OFF"), (on, "ON"), (on_non_affine, "ON_NON_AFFINE")):
        for k in K_VALUES:
            set_seed(0)
            model = build_model("tesseract", k, cfg, benchmark.n).model
            before = count_parameters(model)["trainable"]
            patched = apply_per_step_norm(model, config)
            after = count_parameters(model)["trainable"]
            per_step = hidden_state_rms(model, inputs, device)
            rows.append({
                "k": k, "flag": flag, "patched_module": patched,
                "parameters_before": before, "parameters_after": after,
                "parameters_added": after - before,
                "expected_added": per_step_norm_parameter_cost(cfg.d_model, config),
                "per_step": per_step,
                "z_L_rms_first": per_step[0]["z_L_rms"],
                "z_L_rms_last": per_step[-1]["z_L_rms"],
                "z_H_rms_last": per_step[-1]["z_H_rms"],
                "z_L_max_abs_last": per_step[-1]["z_L_max_abs"],
                "growth_ratio": growth_ratio(per_step),
            })

    def last(flag):
        return [r["z_L_rms_last"] for r in rows if r["flag"] == flag]

    def spread(values):
        return (max(values) - min(values)) / min(values)

    on_rows = [r for r in rows if r["flag"] == "ON"]
    off_rows = [r for r in rows if r["flag"] == "OFF"]
    non_affine_rows = [r for r in rows if r["flag"] == "ON_NON_AFFINE"]

    # Cross-check against the implementation on the GPU box. Imported, never
    # edited. If the two agree to floating-point noise, this diagnostic also
    # confirms phase2/batch01.py's reasoner.
    crosscheck = {"available": False}
    try:
        from phase2.batch01 import apply_per_step_norm as phase2_apply
        from phase2.batch01 import hidden_state_norms as phase2_norms
        deltas = []
        for k in K_VALUES:
            set_seed(0)
            model = build_model("tesseract", k, cfg, benchmark.n).model
            phase2_apply(model)
            theirs = phase2_norms(model, inputs, device)
            mine = next(r for r in on_rows if r["k"] == k)["per_step"]
            deltas.append(max(abs(a["z_L_rms"] - b["z_L_rms"]) for a, b in zip(theirs, mine)))
        crosscheck = {"available": True, "max_abs_z_L_rms_difference": max(deltas),
                      "agrees": max(deltas) < 1e-4,
                      "note": "independent reimplementation vs phase2/batch01.py, flag ON"}
    except Exception as exc:  # noqa: BLE001 - a failed cross-check must not fail the diagnostic
        crosscheck = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    report = {
        "label": "mac_cpu_diagnostic",
        "not_an_arm_result": ("Arm 2c is a GPU training arm under Amendment 02a P1. This is an "
                             "untrained structural measurement and must not be reported as its outcome."),
        "question": "with per-step normalisation ON, does hidden-state RMS stay bounded as K grows?",
        "device": "cpu",
        "torch_version": torch.__version__,
        "platform": sys.platform,
        "examples": EXAMPLES,
        "target_t": TARGET_T,
        "width": "medium (phase2/prototype_medium)",
        "d_model": cfg.d_model,
        "k_values": list(K_VALUES),
        "conditions": rows,
        "summary": {
            "off_growth_ratios": {r["k"]: r["growth_ratio"] for r in off_rows},
            "on_growth_ratios": {r["k"]: r["growth_ratio"] for r in on_rows},
            "off_z_L_rms_last": {r["k"]: r["z_L_rms_last"] for r in off_rows},
            "on_z_L_rms_last": {r["k"]: r["z_L_rms_last"] for r in on_rows},
            "non_affine_z_L_rms_last": {r["k"]: r["z_L_rms_last"] for r in non_affine_rows},
            "off_bounded_within_run": all(r["growth_ratio"] < GROWTH_TOLERANCE for r in off_rows),
            "on_bounded_within_run": all(r["growth_ratio"] < GROWTH_TOLERANCE for r in on_rows),
            "non_affine_bounded_within_run": all(
                r["growth_ratio"] < GROWTH_TOLERANCE for r in non_affine_rows),
            "off_spread_across_k": spread(last("OFF")),
            "on_spread_across_k": spread(last("ON")),
            "non_affine_spread_across_k": spread(last("ON_NON_AFFINE")),
            "on_bounded_across_k": spread(last("ON")) < ACROSS_K_TOLERANCE,
            "non_affine_bounded_across_k": spread(last("ON_NON_AFFINE")) < ACROSS_K_TOLERANCE,
            "parameters_added_on": sorted({r["parameters_added"] for r in on_rows}),
            "parameters_added_non_affine": sorted({r["parameters_added"] for r in non_affine_rows}),
            "non_affine_is_free": all(r["parameters_added"] == 0 for r in non_affine_rows),
            "parameter_cost_independent_of_k": len({r["parameters_added"] for r in on_rows}) == 1,
            "parameter_cost_as_expected": all(
                r["parameters_added"] == r["expected_added"] for r in rows),
        },
        "crosscheck_against_phase2_batch01": crosscheck,
    }

    run_dir = resolve_run_dir("mac_diagnostics/per_step_norm")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "per_step_norm_rms.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    s = report["summary"]
    print("MAC CPU DIAGNOSTIC — not an Arm 2c result\n")
    print(f"{'K':>3} {'flag':>5} {'step1':>8} {'last':>9} {'growth':>8} {'+params':>8}")
    for row in rows:
        print(f"{row['k']:>3} {row['flag']:>5} {row['z_L_rms_first']:>8.2f} "
              f"{row['z_L_rms_last']:>9.2f} {row['growth_ratio']:>7.2f}x {row['parameters_added']:>8}")
    print()
    print(_table(off_rows, "Flag OFF — per-step z_L RMS"))
    print()
    print(_table(on_rows, "Flag ON, affine — per-step z_L RMS"))
    print()
    print(_table(non_affine_rows, "Flag ON, affine=False — per-step z_L RMS"))
    print()
    print(f"bounded within run (ON):         {s['on_bounded_within_run']}")
    print(f"bounded across K (ON):           {s['on_bounded_across_k']} "
          f"(spread {s['on_spread_across_k']*100:.2f}%)")
    print(f"bounded across K (non-affine):   {s['non_affine_bounded_across_k']} "
          f"(spread {s['non_affine_spread_across_k']*100:.2f}%)")
    print(f"spread across K (OFF):           {s['off_spread_across_k']*100:.2f}%")
    print(f"non-affine adds zero parameters: {s['non_affine_is_free']} "
          f"{s['parameters_added_non_affine']}")
    print(f"parameter cost K-independent: {s['parameter_cost_independent_of_k']} "
          f"{s['parameters_added_on']}")
    print(f"crosscheck vs phase2/batch01: {crosscheck}")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
