"""Amendment 01: T=2 shortcut documented, baseline added, raw gates preserved, verdict strictly checked."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from phase2.amendments import accepted_gate_verdicts, evaluate_amendment, verify_involution_shortcut
from phase2.analysis import aggregate
from phase2.config import AmendedGatesRunConfig, GatesRunConfig, load_amendment, load_benchmark
from phase2.data import generate_split
from phase2.runner import run_amended_gates, run_verification_and_gates
from phase2.sanity import sanity_baselines, t2_shortcut_baseline
from utils.config import ConfigError, RunConfig

CPU = torch.device("cpu")


def test_amendment_config_records_option_a():
    a = load_amendment()
    assert a.recorded_before_training and a.original_gates_output_dir == "phase2/gates"
    assert (a.anchor_t_values, a.diagnostic_t_values, a.primary_depth_t_values) == ((1,), (2,), (4, 8))
    (shortcut,) = a.documented_shortcuts
    assert (shortcut.group, shortcut.task, shortcut.t, shortcut.candidate) == ("A5", "main", 2, "s[i+0]*s[i+2]")
    assert shortcut.expected_token_agreement == pytest.approx(16 / 60)
    assert a.gate_verdict == "PASS_WITH_AMENDMENT_01"
    assert accepted_gate_verdicts([a]) == {"PASS", "PASS_WITH_AMENDMENT_01"}


def test_benchmark_is_not_changed_by_the_amendment():
    b = load_benchmark()
    assert (b.group, b.n, b.t_values, b.gates.shortcut_agreement_max) == ("A5", 17, (1, 2, 4, 8), 0.05)


def test_amendment_rejects_overlapping_roles():
    a = load_amendment()
    with pytest.raises(ConfigError, match="disjoint"):
        replace(a, primary_depth_t_values=(2, 4, 8))


def test_shortcut_mechanism_is_exact():
    report = verify_involution_shortcut("A5", 17, 5000, seed=3)
    assert report["elements_squaring_to_identity"] == 16
    assert report["agreement_iff_condition"] and report["sequence_agreement"] == 0.0
    assert report["token_agreement"] == pytest.approx(16 / 60, abs=0.01)


def test_t2_shortcut_baseline():
    split = generate_split("A5", "main", "test", 2, 17, 2000, 1729)
    baseline = t2_shortcut_baseline(split)
    assert baseline["agreement_iff_condition"]
    assert baseline["token_accuracy"] == pytest.approx(baseline["condition_token_rate_percent"])
    assert baseline["token_accuracy"] == pytest.approx(100 * 16 / 60, abs=1.0)
    assert "t2_shortcut_s0_s2" in sanity_baselines(split, seed=0)
    assert "t2_shortcut_s0_s2" not in sanity_baselines(generate_split("A5", "main", "test", 4, 17, 50, 1729), seed=0)
    with pytest.raises(ValueError):
        t2_shortcut_baseline(generate_split("A5", "c1_span", "test", 2, 17, 10, 1729))


def test_z60_shortcut_baseline_is_also_exact_on_its_condition():
    baseline = t2_shortcut_baseline(generate_split("Z60", "main", "test", 2, 17, 500, 1729))
    assert baseline["agreement_iff_condition"]  # 2·s[i+1] ≡ 0 mod 60 ⇔ s[i+1] ∈ {0, 30}


def test_analysis_excludes_diagnostic_t_from_primary_inference():
    results = []
    for seed in (0, 1):
        for t in (1, 2, 4, 8):
            for k in (1, 8):
                score = 0.5 if t == 2 else 0.1 * k * t / (k + t)
                results.append({"cell": {"family": "tesseract", "group": "A5", "task": "main", "t": t, "depth": k,
                                         "seed": seed},
                                "test": {"chance_normalised_token_accuracy": score + 0.01 * seed, "exact_match": 0.0}})
    summary = aggregate(results, 0.9, (1, 8), primary_t_values=(4, 8), diagnostic_t_values=(2,), anchor_t_values=(1,))
    assert {s["t"] for s in summary["primary_depth_statistics"]} == {4, 8}
    assert {s["role"] for s in summary["depth_statistics"] if s["t"] == 2} == {"diagnostic"}
    primary = summary["interaction_regressions_primary"]["tesseract/A5/main"]
    everything = summary["interaction_regressions_all_t_reported_only"]["tesseract/A5/main"]
    assert primary["n"] == 8 and everything["n"] == 16


# ----------------------------------------------------------------------------
# End-to-end: original FAIL run preserved, amended run evaluated separately
# ----------------------------------------------------------------------------


@pytest.fixture
def small_benchmark():
    b = load_benchmark()
    gates = replace(b.gates, uniformity_states=500, uniformity_tv_max=0.2, cycle_states=4, cycle_steps=16,
                    sensitivity_states=400, shortcut_states=2000)
    return replace(b, t_values=(1, 2, 4), interpolation_t_values=(3,), ablation_groups=(), tasks=("main",),
                   val_size=16, test_size=400, length_test_size=8, gates=gates)


@pytest.fixture
def small_amendment():
    return replace(load_amendment(), primary_depth_t_values=(4,))


def _original_run(runs, benchmark):
    original = runs / "phase2" / "gates"
    report = run_verification_and_gates(GatesRunConfig(RunConfig("g", str(original), 7, "cpu"), "phase2/benchmark"),
                                        runs / "g.yaml", original, benchmark, CPU, log=lambda s: None)
    assert report["passed"] is False  # T=2 G2 failure
    return original


def test_amended_gate_run_preserves_original_and_passes(tmp_path, monkeypatch, small_benchmark, small_amendment):
    monkeypatch.setenv("TESSERACT_RUNS_DIR", str(tmp_path))
    original = _original_run(tmp_path, small_benchmark)
    before = {p.name: p.read_bytes() for p in original.iterdir()}

    amended_dir = tmp_path / "phase2" / "gates_amendment01"
    config = AmendedGatesRunConfig(RunConfig("a", str(amended_dir), 7, "cpu"), "phase2/benchmark", "phase2/amendment_01")
    evaluation = run_amended_gates(config, tmp_path / "a.yaml", amended_dir, small_benchmark, small_amendment, CPU,
                                   training_dirs=[tmp_path / "phase2" / "pilot_p1"], log=lambda s: None)

    assert evaluation["raw_gate_verdict"] == "FAIL"
    assert evaluation["raw_G2_passed_by_T"] == {1: None, 2: False, 4: True}
    assert evaluation["amended_verdict"] == "PASS_WITH_AMENDMENT_01", {k: v for k, v in evaluation["checks"].items() if not v}
    assert {p.name: p.read_bytes() for p in original.iterdir()} == before  # original run untouched
    meta = json.loads((amended_dir / "run_metadata.json").read_text())
    assert meta["verdict"] == "PASS_WITH_AMENDMENT_01" and meta["results"]["raw_gate_verdict"] == "FAIL"


def test_amended_gates_refuse_to_overwrite_original(tmp_path, monkeypatch, small_benchmark, small_amendment):
    monkeypatch.setenv("TESSERACT_RUNS_DIR", str(tmp_path))
    original = _original_run(tmp_path, small_benchmark)
    config = AmendedGatesRunConfig(RunConfig("a", str(original), 7, "cpu"), "phase2/benchmark", "phase2/amendment_01")
    with pytest.raises(RuntimeError, match="must not write into the original"):
        run_amended_gates(config, tmp_path / "a.yaml", original, small_benchmark, small_amendment, CPU, log=lambda s: None)


def _fake_inputs():
    per_t = {
        1: {"G1_passed": True, "G2_passed": None, "G2": {"applicable": False}},
        2: {"G1_passed": True, "G2_passed": False,
            "G2": {"max_token_agreement_candidate": "s[i+0]*s[i+2]", "max_token_agreement": 0.2665}},
        4: {"G1_passed": True, "G2_passed": True, "G2": {}},
        8: {"G1_passed": True, "G2_passed": True, "G2": {}},
    }
    gates = {"primary": {"per_T": per_t, "G1_cycles_passed": True}, "self_test": {"ok": True}, "self_test_passed": True,
             "passed": False}
    benchmark = load_benchmark()
    key = f"A5/main/test/T2/n17/N{benchmark.test_size}"
    sanity = {key: {"oracle_independent_simulator": {"exact_match": 100.0},
                    "t2_shortcut_s0_s2": {"agreement_iff_condition": True}}}
    from dataclasses import asdict
    original = json.loads(json.dumps({"benchmark": asdict(benchmark), "primary": gates["primary"],
                                      "self_test": gates["self_test"], "passed": False}))
    return benchmark, {"passed": True}, gates, sanity, original, {"agreement_iff_condition": True}


def test_evaluation_passes_only_when_every_check_holds():
    benchmark, verification, gates, sanity, original, mechanism = _fake_inputs()
    amendment = load_amendment()
    assert evaluate_amendment(amendment, benchmark, verification, gates, sanity, original, mechanism, [])["amended_verdict"] \
        == "PASS_WITH_AMENDMENT_01"


@pytest.mark.parametrize("breakage", ["primary_g2_fails", "wrong_candidate", "agreement_off", "training_exists",
                                      "no_original", "undocumented_failure", "g1_fails", "raw_results_differ"])
def test_evaluation_fails_on_any_violation(breakage):
    benchmark, verification, gates, sanity, original, mechanism = _fake_inputs()
    amendment = load_amendment()
    training = []
    per_t = gates["primary"]["per_T"]
    if breakage == "primary_g2_fails":
        per_t[8]["G2_passed"] = False
        original = json.loads(json.dumps({**original, "primary": gates["primary"]}))
    elif breakage == "wrong_candidate":
        per_t[2]["G2"]["max_token_agreement_candidate"] = "s[i+1]*s[i+2]"
        original = json.loads(json.dumps({**original, "primary": gates["primary"]}))
    elif breakage == "agreement_off":
        per_t[2]["G2"]["max_token_agreement"] = 0.40
        original = json.loads(json.dumps({**original, "primary": gates["primary"]}))
    elif breakage == "training_exists":
        training = ["runs/phase2/pilot_p1"]
    elif breakage == "no_original":
        original = None
    elif breakage == "undocumented_failure":
        amendment = replace(amendment, documented_shortcuts=())
    elif breakage == "g1_fails":
        per_t[4]["G1_passed"] = False
        original = json.loads(json.dumps({**original, "primary": gates["primary"]}))
    elif breakage == "raw_results_differ":
        original["primary"]["per_T"]["2"]["G2"]["max_token_agreement"] = 0.25
    result = evaluate_amendment(amendment, benchmark, verification, gates, sanity, original, mechanism, training)
    assert result["amended_verdict"] == "FAIL"
