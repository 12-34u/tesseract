#!/usr/bin/env python3
"""Record the G1/G2 gate results at T = 3 as an artifact.

The gates had only ever been run at T in {1, 2, 4, 8}, the benchmark's
``t_values``. T = 3 is an ``interpolation_t_values`` depth — evaluation only —
and PHASE2_BATCH_02_PREREG.md §2 reports gate results there, so those results
need to exist as a re-readable artifact rather than as console output. That is
the same defect the Arm 3 deviation was about, and it is not worth repeating in
the file that fixes it.

This trains nothing and loads no model: G1 and G2 are properties of the
automaton, computed from the committed gate configuration. It does not change
any gate verdict. The benchmark's amended verdict (PASS_WITH_AMENDMENT_01) is
over the primary depths T in {4, 8} and is untouched; T = 3, like T = 2, is a
diagnostic depth whose G2 failure is documented rather than fatal.

CPU only.

    python scripts/phase2_gates_t3.py [--t 3] [--output runs/phase2/gates_t3/gates_t3_report.json]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from phase2.automaton import iterate  # noqa: E402
from phase2.batch02 import SHORTCUT_CANDIDATE, shortcut_prediction  # noqa: E402
from phase2.config import load_benchmark  # noqa: E402
from phase2.gates import evaluate_group  # noqa: E402
from phase2.groups import get_group  # noqa: E402
from utils.paths import resolve_run_dir  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402

MECHANISM_STATES = 20000


def involution_mechanism(group, t: int, states: np.ndarray) -> dict:
    """Show that the shortcut is right exactly when a local product is an involution.

    A matching *rate* would leave open that the candidate is a good
    approximation of F^T. Matching token-for-token with an algebraic condition
    shows it is not an approximation at all: it is exact on a fixed subset of
    windows and unrelated elsewhere. That distinction is what lets Batch 02 use
    agreement with the candidate as a discriminator.
    """
    table = group.table
    identity = next(e for e in range(group.order) if all(table[e, x] == x for x in range(group.order)))
    if t == 2:
        # F^2[i] = s_i s_{i+1}^2 s_{i+2} equals s_i s_{i+2} iff s_{i+1}^2 = e.
        local = np.roll(states, -1, axis=-1)
        expression = "s[i+1]"
    elif t == 3:
        # F^3[i] equals s_i s_{i+1} s_{i+2} s_{i+3} iff (s_{i+1} s_{i+2})^2 = e.
        local = table[np.roll(states, -1, axis=-1), np.roll(states, -2, axis=-1)]
        expression = "s[i+1]*s[i+2]"
    else:
        return {"applicable": False, "reason": f"no G2-failing candidate recorded at T={t}"}

    condition = table[local, local] == identity
    agrees = shortcut_prediction(group, states, t) == iterate(group, states, t)
    squaring_to_identity = [e for e in range(group.order) if table[e, e] == identity]
    return {
        "applicable": True,
        "candidate": SHORTCUT_CANDIDATE[t],
        "condition": f"({expression})^2 = e",
        "elements_squaring_to_identity": len(squaring_to_identity),
        "element_fraction": len(squaring_to_identity) / group.order,
        "states": int(states.shape[0]),
        "token_agreement": float(agrees.mean()),
        "condition_frequency": float(condition.mean()),
        "sequence_agreement": float(agrees.all(axis=-1).mean()),
        # The claim: not "these rates are close" but "these masks are identical".
        "agreement_iff_condition": bool(np.array_equal(agrees, condition)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    benchmark = load_benchmark()
    if args.t in benchmark.t_values:
        print(f"T={args.t} is a benchmark t_value and is already covered by scripts/phase2_gates.py.",
              file=sys.stderr)
        return 2
    group = get_group(benchmark.group)

    gates = evaluate_group(benchmark.group, [args.t], benchmark.n, benchmark.gates)
    per_t = gates["per_T"].get(args.t, gates["per_T"].get(str(args.t)))
    rng = np.random.default_rng(benchmark.gates.seed)
    states = rng.integers(0, group.order, size=(MECHANISM_STATES, benchmark.n), dtype=np.int64)

    report = {
        "purpose": "G1/G2 at an interpolation-only depth, recorded as an artifact",
        "preregistration": "PHASE2_BATCH_02_PREREG.md",
        "group": benchmark.group,
        "t": args.t,
        "t_role": "interpolation (evaluation-only)" if args.t in benchmark.interpolation_t_values else "unknown",
        "n": benchmark.n,
        "gate_config": {"seed": benchmark.gates.seed,
                        "uniformity_states": benchmark.gates.uniformity_states,
                        "uniformity_tv_max": benchmark.gates.uniformity_tv_max,
                        "sensitivity_states": benchmark.gates.sensitivity_states,
                        "sensitivity_min": benchmark.gates.sensitivity_min,
                        "shortcut_states": benchmark.gates.shortcut_states,
                        "shortcut_agreement_max": benchmark.gates.shortcut_agreement_max},
        "gates": gates,
        "G1_passed": per_t["G1_passed"],
        "G2_passed": per_t["G2_passed"],
        "shortcut_mechanism": involution_mechanism(group, args.t, states),
        "does_not_change_any_gate_verdict": True,
        "note": ("The amended benchmark verdict PASS_WITH_AMENDMENT_01 is over the primary depths "
                 "T in {4, 8}, both of which pass G2. This depth is evaluation-only and its G2 "
                 "failure is documented, as the T=2 failure is."),
    }

    out = Path(args.output) if args.output else resolve_run_dir("phase2/gates_t3") / f"gates_t{args.t}_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)

    print(f"=== G1/G2 at T={args.t} ({benchmark.group}, n={benchmark.n})")
    print(f"  G1 uniformity  : TV {per_t['G1_uniformity']['tv_distance']:.5f} "
          f"(limit {benchmark.gates.uniformity_tv_max})")
    print(f"  G1 sensitivity : min in-window {per_t['G1_sensitivity']['min_inside_fraction']:.5f}, "
          f"max out-of-window {per_t['G1_sensitivity']['max_outside_fraction']:.5f}")
    print(f"  G1 cycles      : {'pass' if gates['G1_cycles_passed'] else 'FAIL'}")
    print(f"  G1             : {'pass' if per_t['G1_passed'] else 'FAIL'}")
    print(f"  G2             : {'pass' if per_t['G2_passed'] else 'FAIL'} — worst candidate "
          f"{per_t['G2']['max_token_agreement_candidate']} at "
          f"{per_t['G2']['max_token_agreement'] * 100:.2f} % token agreement "
          f"(limit {benchmark.gates.shortcut_agreement_max * 100:.0f} %), "
          f"max sequence agreement {per_t['G2']['max_sequence_agreement'] * 100:.2f} %")
    mech = report["shortcut_mechanism"]
    if mech.get("applicable"):
        print(f"  mechanism      : agrees iff {mech['condition']}; "
              f"{mech['elements_squaring_to_identity']}/{group.order} elements square to e "
              f"({mech['element_fraction'] * 100:.3f} %); "
              f"masks identical: {mech['agreement_iff_condition']}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
