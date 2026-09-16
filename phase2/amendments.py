"""Pre-training design amendments (PHASE2_BENCHMARK_DESIGN.md §8).

An amendment never changes the raw gates. It adds a separate, strictly checked
verdict layer: every raw gate failure must be at a diagnostic T and must match
a documented, exactly verified shortcut, and everything else must still pass.
"""

import json
from dataclasses import asdict
from typing import Dict, Iterable, Optional

import numpy as np

from phase2.automaton import iterate
from phase2.config import AmendmentConfig, BenchmarkConfig
from phase2.groups import get_group
from utils.run_artifacts import to_json_safe

BASE_GATE_VERDICT = "PASS"


def accepted_gate_verdicts(amendments: Iterable[AmendmentConfig] = ()) -> set:
    return {BASE_GATE_VERDICT, *(a.gate_verdict for a in amendments)}


def verify_involution_shortcut(group_name: str, n: int, num_states: int, seed: int) -> Dict:
    """Check exactly that F²(s)[i] == s[i]·s[i+2]  ⇔  s[i+1]² == e."""
    group = get_group(group_name)
    ids = np.arange(group.order)
    squares_to_identity = group.table[ids, ids] == group.identity
    states = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2, n))).integers(0, group.order, (num_states, n))
    agree = iterate(group, states, 2) == group.table[states, np.roll(states, -2, axis=1)]
    condition = squares_to_identity[np.roll(states, -1, axis=1)]
    return {
        "group": group_name,
        "states": num_states,
        "elements_squaring_to_identity": int(squares_to_identity.sum()),
        "element_fraction": float(squares_to_identity.mean()),
        "token_agreement": float(agree.mean()),
        "sequence_agreement": float(agree.all(axis=1).mean()),
        "agreement_iff_condition": bool(np.array_equal(agree, condition)),
    }


def _roundtrip(value):
    return json.loads(json.dumps(to_json_safe(value)))


def evaluate_amendment(
    amendment: AmendmentConfig,
    benchmark: BenchmarkConfig,
    verification: Dict,
    gates: Dict,
    sanity: Dict,
    original_report: Optional[Dict],
    shortcut_check: Dict,
    training_runs_present: Iterable[str],
) -> Dict:
    primary = gates["primary"]
    per_t = primary["per_T"]
    training_runs_present = list(training_runs_present)

    roles = set(amendment.anchor_t_values) | set(amendment.diagnostic_t_values) | set(amendment.primary_depth_t_values)
    g2_failures = sorted(t for t, r in per_t.items() if r["G2_passed"] is False)
    documented = {(s.group, s.task, s.t): s for s in amendment.documented_shortcuts}

    failure_matches = {}
    for t in g2_failures:
        record = documented.get((benchmark.group, "main", t))
        observed = per_t[t]["G2"]
        candidate = observed.get("max_token_agreement_candidate")  # a malformed report fails the match below
        agreement = observed.get("max_token_agreement")
        failure_matches[t] = {
            "diagnostic_t": t in amendment.diagnostic_t_values,
            "documented": record is not None,
            "observed_candidate": candidate,
            "observed_token_agreement": agreement,
            "matches_documented_candidate": record is not None and candidate == record.candidate,
            "within_tolerance": record is not None and agreement is not None
            and abs(agreement - record.expected_token_agreement) <= record.tolerance,
        }

    shortcut_keys = [f"{benchmark.group}/main/test/T{s.t}/n{benchmark.n}/N{benchmark.test_size}"
                     for s in amendment.documented_shortcuts]
    shortcut_baselines = {key: sanity.get(key, {}).get("t2_shortcut_s0_s2") for key in shortcut_keys}

    rerun_raw = _roundtrip({"primary": primary, "self_test": gates["self_test"]})
    checks = {
        "no_training_runs_exist": not training_runs_present,
        "t_roles_partition_benchmark_t_values": roles == set(benchmark.t_values),
        "original_gate_run_found_and_failed": original_report is not None and original_report.get("passed") is False,
        "benchmark_identical_to_original": original_report is not None
        and _roundtrip(asdict(benchmark)) == original_report.get("benchmark"),
        "raw_gate_results_identical_to_original": original_report is not None
        and rerun_raw == {"primary": original_report.get("primary"), "self_test": original_report.get("self_test")},
        "verification_passed": bool(verification["passed"]),
        "G1_passed_all_T": all(r["G1_passed"] for r in per_t.values()),
        "G1_cycles_passed": bool(primary["G1_cycles_passed"]),
        "gate_self_test_passed": bool(gates["self_test_passed"]),
        "G2_passed_all_primary_depth_T": all(per_t[t]["G2_passed"] is True for t in amendment.primary_depth_t_values),
        "every_G2_failure_documented_and_matching": all(
            m["diagnostic_t"] and m["documented"] and m["matches_documented_candidate"] and m["within_tolerance"]
            for m in failure_matches.values()),
        "shortcut_mechanism_verified_exact": bool(shortcut_check["agreement_iff_condition"]),
        "oracle_100_percent_on_all_test_splits": all(
            s["oracle_independent_simulator"]["exact_match"] == 100.0 for s in sanity.values()),
        "shortcut_baseline_reported": all(
            b is not None and b["agreement_iff_condition"] for b in shortcut_baselines.values()),
    }
    return {
        "amendment": asdict(amendment),
        "raw_gate_verdict": "PASS" if gates["passed"] else "FAIL",
        "raw_G2_passed_by_T": {t: r["G2_passed"] for t, r in per_t.items()},
        "raw_G1_passed_by_T": {t: r["G1_passed"] for t, r in per_t.items()},
        "G2_failures": failure_matches,
        "shortcut_mechanism": shortcut_check,
        "shortcut_baselines": shortcut_baselines,
        "training_runs_present": training_runs_present,
        "checks": checks,
        "amended_verdict": amendment.gate_verdict if all(checks.values()) else "FAIL",
    }
