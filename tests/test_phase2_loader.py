"""Dashboard Phase 2 loader: pending states, artifact mapping, and honesty.

Phase 2 has a pre-registered design but no experimental results, so the loader
has two jobs that these tests pin separately: read the checked-in configuration
faithfully, and report an explicit pending state everywhere an artifact is
absent instead of supplying a value.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app" / "backend"))

from services.phase2_loader import MISSING_P1B_MESSAGE, Phase2Loader  # noqa: E402
from services.results_loader import ArtifactError  # noqa: E402


@pytest.fixture
def empty_runs(tmp_path):
    """A runs root with no Phase 2 artifacts at all — today's real state."""
    return Phase2Loader(tmp_path)


# ---------------------------------------------------------------------------
# Configuration is read from the real checked-in configs
# ---------------------------------------------------------------------------


def test_benchmark_is_read_from_the_config_not_hardcoded(empty_runs):
    benchmark = empty_runs.get_benchmark()
    config = (REPO_ROOT / "configs" / "phase2" / "benchmark.yaml").read_text(encoding="utf-8")

    assert benchmark["status"] == "available"
    for field, value in (("group", benchmark["group"]), ("n", benchmark["n"]),
                         ("data_seed", benchmark["data_seed"]), ("val_size", benchmark["val_size"]),
                         ("test_size", benchmark["test_size"])):
        assert f"{field}: {value}" in config, f"{field} must come from benchmark.yaml"
    assert benchmark["source"].endswith("benchmark.yaml")


def test_t_roles_come_from_amendment_01(empty_runs):
    roles = empty_runs.get_t_roles()
    assert roles["status"] == "available"
    assert roles["anchor"] == [1]
    assert roles["diagnostic"] == [2]
    assert roles["primary_depth"] == [4, 8]
    assert roles["source"].endswith("amendment_01.yaml")


def test_protocol_reports_no_early_stopping_and_the_checkpoint_policy(empty_runs):
    protocol = empty_runs.get_protocol()
    assert protocol["status"] == "available"
    assert protocol["optimizer"] == "adamw"
    assert protocol["early_stopping"] is False
    assert "best-validation" in protocol["primary_checkpoint"]
    assert "final" in protocol["sensitivity_checkpoint"]
    assert "after training" in protocol["test_policy"]


def test_plan_totals_138_runs_derived_from_the_config(empty_runs):
    plan = empty_runs.get_plan()
    assert plan["status"] == "available"
    assert plan["total_planned_runs"] == 138
    assert plan["total_completed_runs"] == 0

    by_label = {row["label"]: row for row in plan["stages"]}
    assert by_label["Stage #1"]["planned_runs"] == 12
    assert by_label["Stage #2"]["planned_runs"] == 12
    assert by_label["Stage #2b"]["planned_runs"] == 24
    assert by_label["Stage #3"]["planned_runs"] == 24 and by_label["Stage #3"]["task"] == "c1_span"
    assert by_label["Stage #4"]["planned_runs"] == 24 and by_label["Stage #4"]["task"] == "c2_word"
    assert by_label["Stage #5"]["planned_runs"] == 18 and by_label["Stage #5"]["depth_symbol"] == "L"
    assert by_label["Stage #6"]["planned_runs"] == 18
    assert by_label["Stage #7"]["planned_runs"] == 6


# ---------------------------------------------------------------------------
# No artifacts → pending, never invented values
# ---------------------------------------------------------------------------


def test_no_artifacts_gives_pending_states_everywhere(empty_runs):
    status = empty_runs.get_status()

    assert status["gates"]["status"] == "missing"
    assert "Awaiting gate artifacts" in status["gates"]["message"]
    assert status["p1b"]["status"] == "missing"
    assert status["p1b"]["message"] == MISSING_P1B_MESSAGE
    assert status["results"]["status"] == "missing"
    assert status["results"]["analysis"] is None

    decisions = status["decisions"]
    assert decisions["amendment_01"]["state"] == "ACTIVE — RECORDED"
    assert decisions["amendment_02"]["state"] == "DRAFT — NOT FROZEN"
    assert decisions["amendment_02"]["frozen"] is False
    assert decisions["model_width"]["state"] == "PENDING P1B"
    assert decisions["model_width"]["value"] is None
    assert decisions["training_budget"]["state"] == "PENDING P1B"
    assert decisions["training_budget"]["value"] is None
    assert decisions["p1b"]["state"] == "PENDING"


def test_every_stage_is_blocked_while_the_pre_registration_is_incomplete(empty_runs):
    plan = empty_runs.get_plan()
    assert all(row["state"] == "BLOCKED" for row in plan["stages"])
    reasons = " ".join(plan["blocked_reasons"])
    assert "model width is PENDING_P1B" in reasons
    assert "step budget is PENDING_P1B" in reasons
    assert "Amendment 02" in reasons
    assert "P1b" in reasons


def test_pending_payload_contains_no_experimental_numbers(empty_runs):
    """A pending status must not carry a score, accuracy or loss anywhere."""
    status = empty_runs.get_status()
    blob = json.dumps({k: v for k, v in status.items() if k in ("gates", "p1b", "results")})
    for forbidden in ("chance_normalised_token_accuracy\":", "exact_match\":", "final_val_loss\":"):
        assert forbidden not in blob, f"pending payload leaked {forbidden}"


# ---------------------------------------------------------------------------
# Gate artifacts: the raw T=2 FAIL is preserved
# ---------------------------------------------------------------------------


def _write_gates(runs_root: Path, t2_passed=False):
    gates = runs_root / "phase2" / "gates"
    gates.mkdir(parents=True)
    per_t = {
        "1": {"G1_passed": True, "G1_uniformity": {"tv_distance": 0.0073},
              "G1_sensitivity": {"min_inside_fraction": 1.0},
              "G2": {"applicable": False, "passed": None}, "G2_passed": None},
        "2": {"G1_passed": True, "G1_uniformity": {"tv_distance": 0.0074},
              "G1_sensitivity": {"min_inside_fraction": 0.9293},
              "G2": {"applicable": True, "passed": t2_passed, "max_token_agreement": 0.2665,
                     "max_token_agreement_candidate": "s[i+0]*s[i+2]"}, "G2_passed": t2_passed},
        "4": {"G1_passed": True, "G1_uniformity": {"tv_distance": 0.0086},
              "G1_sensitivity": {"min_inside_fraction": 0.961},
              "G2": {"applicable": True, "passed": True, "max_token_agreement": 0.0357,
                     "max_token_agreement_candidate": "s[i+0]*s[i+4]"}, "G2_passed": True},
        "8": {"G1_passed": True, "G1_uniformity": {"tv_distance": 0.0081},
              "G1_sensitivity": {"min_inside_fraction": 0.9805},
              "G2": {"applicable": True, "passed": True, "max_token_agreement": 0.0173,
                     "max_token_agreement_candidate": "s[i+7]*s[i+6]"}, "G2_passed": True},
    }
    (gates / "gates_report.json").write_text(json.dumps({
        "primary": {"group": "A5", "n": 17, "per_T": per_t, "G1_cycles_passed": True, "passed": False},
        "self_test_passed": True, "passed": False,
    }), encoding="utf-8")
    return gates


def test_raw_t2_failure_is_preserved_and_annotated(tmp_path):
    _write_gates(tmp_path)
    gates = Phase2Loader(tmp_path).get_gates()

    assert gates["status"] == "available"
    assert gates["raw_verdict"] == "FAIL"
    rows = {row["t"]: row for row in gates["per_t"]}
    assert rows[2]["g2_passed"] is False
    assert rows[2]["g2_state"] == "FAIL"
    assert rows[2]["amendment_01_diagnostic"] is True
    assert rows[4]["g2_state"] == "PASS"
    assert rows[8]["g2_state"] == "PASS"
    assert rows[1]["g2_state"] == "N/A"


def test_a_passing_t2_is_not_labelled_a_diagnostic_failure(tmp_path):
    """The diagnostic annotation tracks the artifact, not the T value."""
    _write_gates(tmp_path, t2_passed=True)
    rows = {row["t"]: row for row in Phase2Loader(tmp_path).get_gates()["per_t"]}
    assert rows[2]["g2_state"] == "PASS"
    assert rows[2]["amendment_01_diagnostic"] is False


def test_malformed_gates_report_raises(tmp_path):
    gates = tmp_path / "phase2" / "gates"
    gates.mkdir(parents=True)
    (gates / "gates_report.json").write_text('{"nonsense": true}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="missing 'primary'"):
        Phase2Loader(tmp_path).get_gates()


def test_invalid_json_gates_report_raises(tmp_path):
    gates = tmp_path / "phase2" / "gates"
    gates.mkdir(parents=True)
    (gates / "gates_report.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="invalid JSON"):
        Phase2Loader(tmp_path).get_gates()


def test_status_reports_a_malformed_block_as_an_error_without_failing_the_page(tmp_path):
    gates = tmp_path / "phase2" / "gates"
    gates.mkdir(parents=True)
    (gates / "gates_report.json").write_text('{"nonsense": true}', encoding="utf-8")
    status = Phase2Loader(tmp_path).get_status()
    assert status["gates"]["status"] == "error"
    assert status["benchmark"]["status"] == "available"  # the rest of the page still renders


# ---------------------------------------------------------------------------
# P1b artifacts
# ---------------------------------------------------------------------------


def _write_p1b(runs_root: Path, learnable=True, status="completed"):
    run_dir = runs_root / "phase2" / "p1b_capacity_diagnostic"
    run_dir.mkdir(parents=True)
    import math

    n_evals = 80
    good_loss, bad_loss = math.log(60) - 0.5, math.log(60) + 0.5
    score = 0.5 if learnable else 0.001

    def cell(params):
        curve = [{"step": float((i + 1) * 250), "val_chance_normalised_token_accuracy": score}
                 for i in range(n_evals)]
        return {"model_base": "prototype_small", "curve": curve,
                "at_floor_best": not learnable, "at_floor_final": not learnable,
                "headline": {"parameter_count": params, "final_val_chance_normalised": score,
                             "final_val_loss": good_loss if learnable else bad_loss,
                             "best_val_chance_normalised": score, "steps_run": 20000,
                             "block_calls_per_forward": 1}}

    (run_dir / "diagnostic_report.json").write_text(json.dumps({
        "settings": {"t": 8, "k_values": [1, 8], "max_steps": 20000, "eval_every": 250, "seed": 0,
                     "floor_threshold": 0.02, "chance_level_val_loss_ln_vocab": math.log(60)},
        "model_configs": {"small": {"d_model": 128}, "medium": {"d_model": 256}},
        "runs": {"small": {"1": cell(222140), "8": cell(222140)},
                 "medium": {"1": cell(837436), "8": cell(837436)}},
        "single_seed_note": "one seed per cell: differences are descriptive, not statistical claims",
    }), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps({"status": status, "verdict": "COMPLETED",
                                                           "run_id": "p1b-test"}), encoding="utf-8")
    return run_dir


def test_p1b_absent_shows_the_pending_message(empty_runs):
    p1b = empty_runs.get_p1b()
    assert p1b["status"] == "missing"
    assert p1b["message"] == MISSING_P1B_MESSAGE
    assert p1b["d5"] is None
    assert "cells" not in p1b or not p1b.get("cells")


def test_p1b_report_is_mapped_to_four_cells_with_the_d5_decision(tmp_path):
    _write_p1b(tmp_path)
    p1b = Phase2Loader(tmp_path).get_p1b()

    assert p1b["status"] == "available"
    assert p1b["state"] == "COMPLETE"
    assert len(p1b["cells"]) == 4
    assert {c["width"] for c in p1b["cells"]} == {"small", "medium"}
    assert {c["k"] for c in p1b["cells"]} == {1, 8}
    assert p1b["report_sha256"]
    assert "one seed per cell" in p1b["single_seed_note"]

    decision = p1b["d5"]["decision"]
    assert p1b["d5"]["status"] == "available"
    assert decision["width_decision"]["outcome"] == "SELECT"
    assert decision["width_decision"]["selected_width"] == "small"  # the smaller learnable width
    assert "AMENDMENT 02 D5" in p1b["d5"]["text"]


def test_p1b_with_no_learnable_width_reports_do_not_proceed(tmp_path):
    _write_p1b(tmp_path, learnable=False)
    decision = Phase2Loader(tmp_path).get_p1b()["d5"]["decision"]
    assert decision["width_decision"]["outcome"] == "DO_NOT_PROCEED"
    assert decision["budget_decision"]["recommended_max_steps"] is None


def test_p1b_running_is_distinguished_from_complete(tmp_path):
    _write_p1b(tmp_path, status="running")
    assert Phase2Loader(tmp_path).get_p1b_status()["state"] == "RUNNING"


def test_malformed_p1b_report_surfaces_a_d5_error_not_a_decision(tmp_path):
    run_dir = tmp_path / "phase2" / "p1b_capacity_diagnostic"
    run_dir.mkdir(parents=True)
    (run_dir / "diagnostic_report.json").write_text(json.dumps({"settings": {}, "runs": {}}), encoding="utf-8")
    p1b = Phase2Loader(tmp_path).get_p1b()
    assert p1b["d5"]["status"] == "error"
    assert p1b["d5"]["decision"] is None


# ---------------------------------------------------------------------------
# Results and discovery
# ---------------------------------------------------------------------------


def test_results_absent_lists_the_future_metrics_without_inventing_them(empty_runs):
    results = empty_runs.get_results()
    assert results["status"] == "missing"
    assert results["analysis"] is None
    assert results["total_completed_runs"] == 0
    for metric in ("delta_s_per_seed", "did_s_per_seed", "ci95_t_interval", "floor_ceiling"):
        assert metric in results["planned_metrics"]


def test_completed_cells_are_discovered_and_counted(tmp_path):
    """A finished cell on disk moves its stage from BLOCKED to a live count."""
    cell = tmp_path / "phase2" / "experiment" / "cells" / "prototype_small" / "tesseract_d4_A5_main_T4_s0"
    cell.mkdir(parents=True)
    (cell / "result.json").write_text("{}", encoding="utf-8")
    (cell / "run_metadata.json").write_text(json.dumps({"status": "completed", "verdict": "COMPLETED"}),
                                            encoding="utf-8")
    plan = Phase2Loader(tmp_path).get_plan()
    stage1 = next(row for row in plan["stages"] if row["label"] == "Stage #1")
    assert stage1["completed_runs"] == 1
    assert plan["total_completed_runs"] == 1


def test_a_failed_cell_is_not_counted_as_complete(tmp_path):
    cell = tmp_path / "phase2" / "experiment" / "cells" / "prototype_small" / "tesseract_d4_A5_main_T4_s0"
    cell.mkdir(parents=True)
    (cell / "result.json").write_text("{}", encoding="utf-8")
    (cell / "run_metadata.json").write_text(json.dumps({"status": "completed", "verdict": "FAIL"}), encoding="utf-8")
    assert Phase2Loader(tmp_path).get_plan()["total_completed_runs"] == 0


def test_status_exposes_provenance_and_config_hashes(empty_runs):
    provenance = empty_runs.get_status()["provenance"]
    assert "git" in provenance and "commit" in provenance["git"]
    configs = provenance["configs"]
    assert configs["benchmark.yaml"]["present"] is True
    assert configs["benchmark.yaml"]["sha256"]
    assert configs["amendment_02_draft.yaml"]["present"] is True


# ---------------------------------------------------------------------------
# Phase 1 must be unaffected
# ---------------------------------------------------------------------------


def test_phase1_endpoints_do_not_depend_on_phase2(tmp_path):
    """A broken Phase 2 artifact must not disturb the Phase 1 loader."""
    from services.results_loader import ResultsLoader

    gates = tmp_path / "phase2" / "gates"
    gates.mkdir(parents=True)
    (gates / "gates_report.json").write_text("{not json", encoding="utf-8")

    phase1 = ResultsLoader(tmp_path)
    summary = phase1.get_summary()
    assert "experiments" in summary
    assert all(entry["id"] in ("crucible", "k_scaling", "cellular_automaton") for entry in summary["experiments"])


def test_phase1_experiment_ids_are_unchanged(tmp_path):
    from services.results_loader import EXPERIMENTS

    assert set(EXPERIMENTS) == {"crucible", "k_scaling", "cellular_automaton"}
    assert all(name.startswith("Phase 1 · ") for name in EXPERIMENTS.values())


# ---------------------------------------------------------------------------
# Model families in the dashboard payload
# ---------------------------------------------------------------------------


def test_model_families_are_listed_with_computed_parameter_counts(empty_runs):
    families = empty_runs.get_model_families()
    assert families["status"] == "available"
    assert [f["name"] for f in families["families"]] == ["small", "medium", "large"]

    by_name = {f["name"]: f for f in families["families"]}
    assert by_name["small"]["label"] == "Small Prototype (222K)"
    assert by_name["medium"]["label"] == "Medium Prototype (837K)"
    assert by_name["large"]["label"] == "Large Research Model (~7M)"
    assert by_name["large"]["d_model"] == 768 and by_name["large"]["head_dim"] == 32
    assert all(f["parameter_count_is_constant_across_k"] for f in families["families"])
    assert not families["errors"]


def test_no_width_is_selected_while_the_choice_is_pending(empty_runs):
    families = empty_runs.get_model_families()
    assert families["selection_pending"] is True
    assert families["selected_base"] is None
    assert not any(f["selected"] for f in families["families"])


def test_backend_parameter_formula_agrees_with_the_research_code(empty_runs):
    """The dashboard must not drift from phase2.models."""
    from phase2.config import MODEL_FAMILIES, ModelVariant, resolve_model_variant
    from phase2.models import model_parameter_count
    from services.phase2_loader import MODEL_FAMILY_CONFIGS, tesseract_parameter_count

    assert [n for n, _, _ in MODEL_FAMILY_CONFIGS] == [f.name for f in MODEL_FAMILIES]
    assert [b for _, b, _ in MODEL_FAMILY_CONFIGS] == [f.base for f in MODEL_FAMILIES]

    by_name = {f["name"]: f for f in empty_runs.get_model_families()["families"]}
    for family in MODEL_FAMILIES:
        cfg = resolve_model_variant(ModelVariant(family.name, family.base), 60)
        expected = model_parameter_count(60, cfg.d_model, cfg.d_ff, cfg.max_seq_len, 1)
        assert tesseract_parameter_count(60, cfg.d_model, cfg.d_ff, cfg.max_seq_len) == expected
        assert by_name[family.name]["parameter_count"] == expected == family.approx_parameters


def test_model_families_appear_in_the_status_payload(empty_runs):
    assert empty_runs.get_status()["model_families"]["status"] == "available"


# ---------------------------------------------------------------------------
# Capacity diagnostics: P1b, P1b-2, and the combined width selection
# ---------------------------------------------------------------------------


def _write_capacity(runs_root: Path, relative: str, widths: dict, run_id: str, status="completed"):
    """widths: {name: (parameter_count, learnable)}"""
    import math

    good, bad = math.log(60) - 0.5, math.log(60) + 0.5
    d = runs_root / relative
    d.mkdir(parents=True, exist_ok=True)

    def cell(p, ok):
        score = 0.5 if ok else 0.001
        return {"model_base": "x", "at_floor_best": not ok, "at_floor_final": not ok,
                "curve": [{"step": float((i + 1) * 250), "val_chance_normalised_token_accuracy": score}
                          for i in range(80)],
                "headline": {"parameter_count": p, "final_val_chance_normalised": score,
                             "final_val_loss": good if ok else bad, "best_val_chance_normalised": score,
                             "steps_run": 20000, "best_step": 20000, "block_calls_per_forward": 1,
                             "final_val_token_accuracy": 50.0, "final_val_exact_match": 1.0}}

    (d / "diagnostic_report.json").write_text(json.dumps({
        "settings": {"t": 8, "n": 17, "k_values": [1, 8], "seed": 0,
                     "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.01},
                     "batch_size": 128, "eval_every": 250, "max_steps": 20000, "early_stopping": False,
                     "test_split_evaluated": False, "floor_threshold": 0.02,
                     "chance_level_val_loss_ln_vocab": math.log(60)},
        "model_configs": {n: {"d_model": 1} for n in widths},
        "single_seed_note": "one seed per cell: differences are descriptive, not statistical claims",
        "gates_run_id": "g", "gates_verdict": "PASS_WITH_AMENDMENT_01",
        "runs": {n: {"1": cell(p, ok), "8": cell(p, ok)} for n, (p, ok) in widths.items()},
    }), encoding="utf-8")
    (d / "run_metadata.json").write_text(json.dumps(
        {"run_id": run_id, "status": status, "verdict": "COMPLETED",
         "git": {"commit": "abc1234", "dirty": False}}), encoding="utf-8")


SMALL_P, MEDIUM_P, LARGE_P = 222_140, 837_436, 7_230_780


def test_p1b2_config_is_loadable_and_mirrors_p1b(empty_runs):
    """P1b-2 must be independently executable and differ only in width and output."""
    from phase2.config import CapacityDiagnosticConfig
    from utils.config import load_config

    p1b = load_config("phase2/capacity_diagnostic_p1b", CapacityDiagnosticConfig)
    p1b2 = load_config("phase2/capacity_diagnostic_p1b2", CapacityDiagnosticConfig)

    assert [m.name for m in p1b2.models] == ["large"]
    assert p1b2.models[0].base == "phase2/prototype_large"
    assert p1b2.experiment.output_dir == "phase2/p1b2_capacity_diagnostic"
    assert p1b2.experiment.output_dir != p1b.experiment.output_dir

    for field in ("t", "k_values", "vocab_size", "batch_size", "eval_every", "eval_batch_size",
                  "max_steps", "floor_threshold", "benchmark", "reference_pilot_output_dir"):
        assert getattr(p1b2, field) == getattr(p1b, field), f"{field} must match P1b"
    assert p1b2.optimizer == p1b.optimizer
    assert p1b2.experiment.seed == p1b.experiment.seed


def test_existing_p1b_config_still_lists_only_small_and_medium():
    from phase2.config import CapacityDiagnosticConfig
    from utils.config import load_config

    p1b = load_config("phase2/capacity_diagnostic_p1b", CapacityDiagnosticConfig)
    assert [m.name for m in p1b.models] == ["small", "medium"]


def test_no_artifacts_gives_the_three_required_pending_states(empty_runs):
    capacity = empty_runs.get_capacity_diagnostics()
    assert capacity["p1b"]["state"] == "PENDING"
    assert capacity["p1b2"]["state"] == "NOT STARTED"
    assert capacity["combined"]["state"] == "WAITING FOR P1b + P1b-2"
    assert capacity["combined"]["d5"] is None
    assert capacity["combined"]["awaiting"] == ["P1b", "P1b-2"]


def test_large_is_shown_as_a_candidate_not_a_selection(empty_runs):
    capacity = empty_runs.get_capacity_diagnostics()
    large = next(w for w in capacity["candidate_widths"] if w["name"] == "large")
    assert large["status"] == "Candidate — P1b-2 pending"
    assert large["diagnostic"] == "P1b-2"
    assert "not a selection" in capacity["note"]

    status = empty_runs.get_status()
    assert status["decisions"]["model_width"]["state"] == "PENDING P1B"
    assert status["decisions"]["model_width"]["value"] is None
    assert status["model_families"]["selection_pending"] is True
    assert not any(f["selected"] for f in status["model_families"]["families"])


def test_p1b2_carries_its_provenance_note(empty_runs):
    p1b2 = empty_runs.get_p1b2()
    assert "introduced after the Large model family was added" in p1b2["provenance_note"]
    assert "does not alter the original P1b data" in p1b2["provenance_note"]


def test_capacity_pending_payload_contains_no_metrics(empty_runs):
    blob = json.dumps(empty_runs.get_capacity_diagnostics())
    for forbidden in ('"final_val_loss":', '"final_val_chance_normalised":', '"best_step":'):
        assert forbidden not in blob, f"pending capacity payload leaked {forbidden}"


def test_stage_plan_is_blocked_until_both_diagnostics_complete(empty_runs):
    reasons = " ".join(empty_runs.get_plan()["blocked_reasons"])
    assert "P1b is PENDING" in reasons
    assert "P1b-2 is NOT STARTED" in reasons


def test_combined_waits_while_only_p1b_exists(tmp_path):
    _write_capacity(tmp_path, "phase2/p1b_capacity_diagnostic",
                    {"small": (SMALL_P, True), "medium": (MEDIUM_P, True)}, "p1b-run")
    capacity = Phase2Loader(tmp_path).get_capacity_diagnostics()
    assert capacity["p1b"]["status"] == "available"
    assert capacity["p1b2"]["state"] == "NOT STARTED"
    assert capacity["combined"]["state"] == "WAITING FOR P1b + P1b-2"
    assert capacity["combined"]["awaiting"] == ["P1b-2"]
    assert capacity["combined"]["d5"] is None


def test_combined_report_and_d5_once_both_exist(tmp_path):
    _write_capacity(tmp_path, "phase2/p1b_capacity_diagnostic",
                    {"small": (SMALL_P, False), "medium": (MEDIUM_P, True)}, "p1b-run")
    _write_capacity(tmp_path, "phase2/p1b2_capacity_diagnostic", {"large": (LARGE_P, True)}, "p1b2-run")
    combined = Phase2Loader(tmp_path).get_combined_capacity()

    assert combined["status"] == "available" and combined["state"] == "COMBINED"
    assert [f"{c['width']}/K={c['k']}" for c in combined["cells"]] == [
        "small/K=1", "small/K=8", "medium/K=1", "medium/K=8", "large/K=1", "large/K=8"]
    assert {c["source"] for c in combined["cells"]} == {"P1b", "P1b-2"}
    assert combined["d5"]["status"] == "available"
    assert combined["d5"]["evaluated_once_on_combined_report"] is True
    # medium is the smallest learnable width here — not large
    assert combined["d5"]["decision"]["width_decision"]["selected_width"] == "medium"


def test_combined_never_prefers_large_over_a_learnable_smaller_width(tmp_path):
    _write_capacity(tmp_path, "phase2/p1b_capacity_diagnostic",
                    {"small": (SMALL_P, True), "medium": (MEDIUM_P, True)}, "p1b-run")
    _write_capacity(tmp_path, "phase2/p1b2_capacity_diagnostic", {"large": (LARGE_P, True)}, "p1b2-run")
    d5 = Phase2Loader(tmp_path).get_combined_capacity()["d5"]
    assert d5["decision"]["width_decision"]["selected_width"] == "small"


def test_combined_surfaces_a_merge_failure_as_an_error(tmp_path):
    """Conflicting sources must not silently produce a decision."""
    _write_capacity(tmp_path, "phase2/p1b_capacity_diagnostic", {"large": (LARGE_P, True)}, "p1b-run")
    _write_capacity(tmp_path, "phase2/p1b2_capacity_diagnostic", {"large": (LARGE_P, True)}, "p1b2-run")
    combined = Phase2Loader(tmp_path).get_combined_capacity()
    assert combined["status"] == "error"
    assert combined["d5"] is None
    assert "duplicate width" in combined["message"]


def test_p1b2_running_state_comes_from_artifacts_only(tmp_path):
    _write_capacity(tmp_path, "phase2/p1b2_capacity_diagnostic", {"large": (LARGE_P, True)},
                    "p1b2-run", status="running")
    assert Phase2Loader(tmp_path).get_p1b2_status()["state"] == "RUNNING"


def test_capacity_diagnostics_appear_in_the_status_payload(empty_runs):
    assert empty_runs.get_status()["capacity_diagnostics"]["status"] == "available"
