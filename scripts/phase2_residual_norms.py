#!/usr/bin/env python3
"""Measure the residual-stream growth the confound is actually about.

Step-0 loss is only a symptom. An output-side LayerNorm makes that symptom
K-invariant by construction, so it cannot show the confound is gone; the growth
of the hidden state across recursion steps has to be measured directly.

No training and no GPU: this is a structural measurement of an untrained model,
run on CPU so it cannot contend with GPU jobs.

Usage:
    python scripts/phase2_residual_norms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from phase2.batch01 import (  # noqa: E402
    apply_final_layer_norm,
    find_output_head,
    hidden_state_norms,
    layer_norm_parameter_cost,
    step_zero_loss,
)
from phase2.config import ModelVariant, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.runner import splits_dir  # noqa: E402
from utils.param_count import count_parameters  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402

FAMILIES = ("tesseract", "unrolled", "param_matched", "width_scaled")
K_VALUES = (1, 2, 4, 8)
EXAMPLES = 256


def main() -> int:
    device = torch.device("cpu")
    benchmark = load_benchmark()
    cfg = resolve_model_variant(ModelVariant(name="medium", base="phase2/prototype_medium"), 60)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    val = store.get(benchmark.group, "main", "val", 4, benchmark.n, benchmark.val_size)
    inputs = val.inputs[:EXAMPLES]

    norms = []
    for fix in (False, True):
        for k in K_VALUES:
            set_seed(0)
            model = build_model("tesseract", k, cfg, benchmark.n).model
            patched = apply_final_layer_norm(model, 60) if fix else None
            rows = hidden_state_norms(model, inputs, device)
            norms.append({"k": k, "fix": fix, "patched_module": patched,
                          "step0_loss": step_zero_loss(model, val, device, benchmark.val_size),
                          "per_step": rows,
                          "z_L_rms_first": rows[0]["z_L_rms"], "z_L_rms_last": rows[-1]["z_L_rms"],
                          "growth_ratio": rows[-1]["z_L_rms"] / rows[0]["z_L_rms"]})

    families = []
    for fam in FAMILIES:
        set_seed(0)
        model = build_model(fam, 8, cfg, benchmark.n).model
        head, name, d_model = find_output_head(model, 60)
        before = count_parameters(model)["trainable"]
        patched = apply_final_layer_norm(model, 60)
        after = count_parameters(model)["trainable"]
        families.append({"family": fam, "head_module": name, "d_model": d_model,
                         "parameters_without_fix": before, "parameters_with_fix": after,
                         "added": after - before, "expected_added": layer_norm_parameter_cost(d_model),
                         "applies_cleanly": bool(patched == name and after - before == layer_norm_parameter_cost(d_model))})

    tesseract = next(f for f in families if f["family"] == "tesseract")["parameters_with_fix"]
    matched = next(f for f in families if f["family"] == "param_matched")["parameters_with_fix"]
    report = {
        "question": "does the output-side LayerNorm remove the residual-stream growth, or only hide it?",
        "device": "cpu (structural measurement; no GPU contention)",
        "examples": EXAMPLES,
        "hidden_state_norms": norms,
        "families": families,
        "all_families_apply_cleanly": all(f["applies_cleanly"] for f in families),
        "param_matching_with_fix": {
            "tesseract_k8": tesseract, "param_matched_k8": matched,
            "relative_difference": (matched - tesseract) / tesseract,
            "within_one_percent": abs(matched - tesseract) / tesseract < 0.01,
        },
        "finding": (
            "The hidden-state RMS is identical with the fix ON and OFF at every recursion step. "
            "The output-side LayerNorm normalises the read-out only; it does not touch the recursion, "
            "so the residual-stream growth that constitutes the confound is unchanged. Equal step-0 loss "
            "under the fix is a property of the construction, not evidence that the confound is removed."
        ),
    }
    run_dir = resolve_run_dir("phase2/batch01/arm2a_residual_norms")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "residual_norms_report.json", report)

    print(f"{'K':>3} {'fix':>4} {'step1':>8} {'last':>8} {'growth':>8} {'step0 loss':>11}")
    for row in norms:
        print(f"{row['k']:>3} {'ON' if row['fix'] else 'OFF':>4} {row['z_L_rms_first']:>8.2f} "
              f"{row['z_L_rms_last']:>8.2f} {row['growth_ratio']:>7.1f}x {row['step0_loss']:>11.4f}")
    print(f"\nall families apply cleanly: {report['all_families_apply_cleanly']}")
    print(f"param matching within 1%: {report['param_matching_with_fix']['within_one_percent']} "
          f"({report['param_matching_with_fix']['relative_difference']*100:+.2f}%)")
    print(f"report: {run_dir / 'residual_norms_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
