"""Combining P1b (small, medium) with P1b-2 (large) into one D5 input.

The merge exists so the width decision is made once, over every candidate, from
a report whose every cell can be traced to the run that produced it. These tests
pin the two halves of that: the combined report is correct and deterministic,
and the merge refuses rather than reconciling when the sources disagree.

The D5 criteria are not exercised here beyond confirming the evaluator consumes
the combined report unchanged; the selection rule itself is pinned by
tests/phase2/test_p1b_decision.py and the eight-combination test below.
"""

import itertools
import json
import math

import pytest

from phase2.capacity_merge import (
    PROVENANCE_NOTE,
    CapacityMergeError,
    expected_cells,
    load_source,
    merge_capacity_report_files,
    merge_capacity_reports,
    missing_widths,
)
from phase2.p1b_decision import evaluate_d5

SMALL, MEDIUM, LARGE = 222_140, 837_436, 7_230_780
GOOD_LOSS, BAD_LOSS = math.log(60) - 0.5, math.log(60) + 0.5
N_EVALS = 80

SETTINGS = {
    "t": 8, "n": 17, "k_values": [1, 8], "seed": 0,
    "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.01},
    "batch_size": 128, "eval_every": 250, "max_steps": 20000, "early_stopping": False,
    "test_split_evaluated": False, "floor_threshold": 0.02,
    "chance_level_val_loss_ln_vocab": math.log(60),
}


def cell(params, learnable=True):
    score = 0.5 if learnable else 0.001
    return {
        "model_base": "x",
        "at_floor_best": not learnable,
        "at_floor_final": not learnable,
        "curve": [{"step": float((i + 1) * 250), "val_chance_normalised_token_accuracy": score}
                  for i in range(N_EVALS)],
        "headline": {
            "parameter_count": params, "final_val_chance_normalised": score,
            "final_val_loss": GOOD_LOSS if learnable else BAD_LOSS,
            "best_val_chance_normalised": score, "steps_run": 20000, "best_step": 20000,
            "block_calls_per_forward": 1, "final_val_token_accuracy": 50.0, "final_val_exact_match": 1.0,
        },
    }


def report(widths, settings=None):
    """widths: {name: (parameter_count, learnable)}

    Cells are generated for exactly the K values the settings declare, so a
    source is internally consistent and a settings difference surfaces as a
    protocol mismatch rather than a missing cell.
    """
    settings = dict(settings or SETTINGS)
    ks = [str(k) for k in settings["k_values"]]
    return {
        "settings": settings,
        "model_configs": {name: {"d_model": 1} for name in widths},
        "single_seed_note": "one seed per cell: differences are descriptive, not statistical claims",
        "gates_run_id": "gates-1", "gates_verdict": "PASS_WITH_AMENDMENT_01",
        "runs": {name: {k: cell(p, ok) for k in ks} for name, (p, ok) in widths.items()},
    }


def write_source(tmp_path, name, widths, run_id, settings=None):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "diagnostic_report.json").write_text(json.dumps(report(widths, settings)), encoding="utf-8")
    (d / "run_metadata.json").write_text(json.dumps({
        "run_id": run_id, "status": "completed", "verdict": "COMPLETED",
        "git": {"commit": "abc1234", "dirty": False},
    }), encoding="utf-8")
    return d / "diagnostic_report.json"


@pytest.fixture
def pair(tmp_path):
    """The real shape: P1b owns small+medium, P1b-2 owns large."""
    return {
        "P1b": write_source(tmp_path, "p1b", {"small": (SMALL, True), "medium": (MEDIUM, True)}, "p1b-run"),
        "P1b-2": write_source(tmp_path, "p1b2", {"large": (LARGE, True)}, "p1b2-run"),
    }


# ---------------------------------------------------------------------------
# The combined six-cell report
# ---------------------------------------------------------------------------


def test_merge_produces_exactly_the_six_expected_cells(pair):
    merged = merge_capacity_report_files(pair)
    assert expected_cells(merged) == [
        "small/K=1", "small/K=8", "medium/K=1", "medium/K=8", "large/K=1", "large/K=8",
    ]
    assert list(merged["runs"]) == ["small", "medium", "large"]
    assert missing_widths(merged, ["small", "medium", "large"]) == []


def test_widths_are_ordered_by_parameter_count(pair):
    merged = merge_capacity_report_files(pair)
    counts = [c["parameter_count"] for c in merged["cells"]]
    assert counts == sorted(counts)
    assert counts[0] == SMALL and counts[-1] == LARGE


def test_every_cell_keeps_its_source_provenance(pair):
    merged = merge_capacity_report_files(pair)
    by_width = {c["width"]: c["source"] for c in merged["cells"]}
    assert by_width == {"small": "P1b", "medium": "P1b", "large": "P1b-2"}

    provenance = merged["provenance_by_width"]
    assert provenance["small"]["run_id"] == "p1b-run"
    assert provenance["large"]["run_id"] == "p1b2-run"
    for entry in provenance.values():
        assert entry["git_commit"] == "abc1234"
        assert entry["report_sha256"]
        assert entry["gates_verdict"] == "PASS_WITH_AMENDMENT_01"
        assert entry["seed"] == 0


def test_sources_are_listed_with_the_widths_they_contributed(pair):
    sources = {s["label"]: s for s in merge_capacity_report_files(pair)["sources"]}
    assert sources["P1b"]["widths"] == ["small", "medium"]
    assert sources["P1b-2"]["widths"] == ["large"]
    assert sources["P1b"]["run_status"] == "completed"


def test_merge_records_that_p1b2_came_after_large_was_added(pair):
    merged = merge_capacity_report_files(pair)
    assert merged["provenance_note"] == PROVENANCE_NOTE
    assert "introduced after the Large model family was added" in merged["provenance_note"]
    assert "does not alter the original P1b data" in merged["provenance_note"]
    assert "smallest width" in merged["decision_criteria_unchanged"]


def test_merge_is_deterministic(pair):
    a = json.dumps(merge_capacity_report_files(pair), sort_keys=True)
    b = json.dumps(merge_capacity_report_files(pair), sort_keys=True)
    assert a == b


def test_merge_does_not_modify_the_source_reports(pair):
    before = {label: path.read_bytes() for label, path in pair.items()}
    merge_capacity_report_files(pair)
    assert {label: path.read_bytes() for label, path in pair.items()} == before


def test_merge_is_pure_over_already_loaded_sources(pair):
    sources = [load_source(path, label) for label, path in sorted(pair.items())]
    snapshot = json.dumps([s["report"] for s in sources], sort_keys=True)
    merge_capacity_reports(sources)
    assert json.dumps([s["report"] for s in sources], sort_keys=True) == snapshot


def test_protocol_settings_are_carried_through_unchanged(pair):
    merged = merge_capacity_report_files(pair)
    for key, value in SETTINGS.items():
        assert merged["settings"][key] == value


# ---------------------------------------------------------------------------
# Rejections: the merge refuses rather than reconciling
# ---------------------------------------------------------------------------


def test_missing_k_in_a_width_is_rejected(tmp_path):
    bad = tmp_path / "p1b2"
    bad.mkdir()
    payload = report({"large": (LARGE, True)})
    del payload["runs"]["large"]["8"]          # declared in k_values, absent from runs
    (bad / "diagnostic_report.json").write_text(json.dumps(payload), encoding="utf-8")
    good = write_source(tmp_path, "p1b", {"small": (SMALL, True)}, "p1b-run")
    with pytest.raises(CapacityMergeError, match=r"missing K \[8\]"):
        merge_capacity_report_files({"P1b": good, "P1b-2": bad / "diagnostic_report.json"})


def test_duplicate_width_across_sources_is_rejected(tmp_path):
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    b = write_source(tmp_path, "b", {"small": (SMALL, True)}, "run-b")
    with pytest.raises(CapacityMergeError, match="duplicate width"):
        merge_capacity_report_files({"P1b": a, "P1b-2": b})


def test_conflicting_cell_for_the_same_width_is_rejected(tmp_path):
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    b = write_source(tmp_path, "b", {"small": (SMALL, False)}, "run-b")   # same width, different numbers
    with pytest.raises(CapacityMergeError, match="duplicate width|conflicting cell"):
        merge_capacity_report_files({"P1b": a, "P1b-2": b})


@pytest.mark.parametrize("key, value", [
    ("max_steps", 10000),
    ("eval_every", 500),
    ("seed", 1),
    ("t", 4),
    ("k_values", [1, 2, 8]),
    ("batch_size", 64),
    ("floor_threshold", 0.05),
    ("optimizer", {"name": "adamw", "learning_rate": 0.0003, "weight_decay": 0.01}),
])
def test_protocol_mismatch_is_rejected(tmp_path, key, value):
    """A different budget, cadence, seed, T, K set or optimizer is a different experiment."""
    other = dict(SETTINGS, **{key: value})
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    b = write_source(tmp_path, "b", {"large": (LARGE, True)}, "run-b", settings=other)
    with pytest.raises(CapacityMergeError, match=f"protocol mismatch on settings.{key}"):
        merge_capacity_report_files({"P1b": a, "P1b-2": b})


def test_missing_report_file_is_rejected(tmp_path):
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    with pytest.raises(CapacityMergeError, match="no diagnostic report"):
        merge_capacity_report_files({"P1b": a, "P1b-2": tmp_path / "nope" / "diagnostic_report.json"})


def test_invalid_json_is_rejected(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "diagnostic_report.json").write_text("{not json", encoding="utf-8")
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    with pytest.raises(CapacityMergeError, match="not valid JSON"):
        merge_capacity_report_files({"P1b": a, "P1b-2": bad / "diagnostic_report.json"})


@pytest.mark.parametrize("drop", ["settings", "runs", "model_configs", "single_seed_note"])
def test_malformed_report_missing_a_top_level_key_is_rejected(tmp_path, drop):
    bad = tmp_path / "bad"
    bad.mkdir()
    payload = report({"large": (LARGE, True)})
    del payload[drop]
    (bad / "diagnostic_report.json").write_text(json.dumps(payload), encoding="utf-8")
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    with pytest.raises(CapacityMergeError, match=f"missing '{drop}'"):
        merge_capacity_report_files({"P1b": a, "P1b-2": bad / "diagnostic_report.json"})


def test_a_single_source_is_rejected(tmp_path):
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    with pytest.raises(CapacityMergeError, match="at least two source diagnostics"):
        merge_capacity_report_files({"P1b": a})


def test_parameter_count_must_be_constant_across_k(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    payload = report({"large": (LARGE, True)})
    payload["runs"]["large"]["8"]["headline"]["parameter_count"] = LARGE + 1
    (bad / "diagnostic_report.json").write_text(json.dumps(payload), encoding="utf-8")
    a = write_source(tmp_path, "a", {"small": (SMALL, True)}, "run-a")
    with pytest.raises(CapacityMergeError, match="parameter count differs across K"):
        merge_capacity_report_files({"P1b": a, "P1b-2": bad / "diagnostic_report.json"})


# ---------------------------------------------------------------------------
# D5 consumes the combined report, criteria unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("small_ok, medium_ok, large_ok", list(itertools.product([True, False], repeat=3)))
def test_all_eight_learnability_combinations(tmp_path, small_ok, medium_ok, large_ok):
    """D5 recommends the SMALLEST learnable width — never Large just because it is the research model."""
    p1b = write_source(tmp_path, "p1b", {"small": (SMALL, small_ok), "medium": (MEDIUM, medium_ok)}, "p1b-run")
    p1b2 = write_source(tmp_path, "p1b2", {"large": (LARGE, large_ok)}, "p1b2-run")
    merged = merge_capacity_report_files({"P1b": p1b, "P1b-2": p1b2})
    assert len(merged["cells"]) == 6

    decision = evaluate_d5(merged)["width_decision"]
    if small_ok:
        expected = "small"
    elif medium_ok:
        expected = "medium"
    elif large_ok:
        expected = "large"
    else:
        expected = None

    if expected is None:
        assert decision["outcome"] == "DO_NOT_PROCEED"
        assert decision["selected_width"] is None
    else:
        assert decision["outcome"] == "SELECT"
        assert decision["selected_width"] == expected


def test_large_is_never_preferred_over_a_learnable_smaller_width(tmp_path):
    """The case the whole design guards against."""
    p1b = write_source(tmp_path, "p1b", {"small": (SMALL, True), "medium": (MEDIUM, True)}, "p1b-run")
    p1b2 = write_source(tmp_path, "p1b2", {"large": (LARGE, True)}, "p1b2-run")
    merged = merge_capacity_report_files({"P1b": p1b, "P1b-2": p1b2})
    assert evaluate_d5(merged)["width_decision"]["selected_width"] == "small"


def test_large_is_selected_only_when_it_is_the_smallest_learnable(tmp_path):
    p1b = write_source(tmp_path, "p1b", {"small": (SMALL, False), "medium": (MEDIUM, False)}, "p1b-run")
    p1b2 = write_source(tmp_path, "p1b2", {"large": (LARGE, True)}, "p1b2-run")
    merged = merge_capacity_report_files({"P1b": p1b, "P1b-2": p1b2})
    assert evaluate_d5(merged)["width_decision"]["selected_width"] == "large"


def test_combined_report_reports_all_three_widths_to_d5(tmp_path, pair):
    widths = {w["width"] for w in evaluate_d5(merge_capacity_report_files(pair))["widths"]}
    assert widths == {"small", "medium", "large"}


def test_d5_source_is_unchanged_by_this_feature():
    """The merge must not have required editing the evaluator's criteria."""
    from pathlib import Path

    import phase2.p1b_decision as d5

    source = Path(d5.__file__).read_text(encoding="utf-8")
    assert d5.MIN_CHANCE_NORMALISED == 0.02
    assert d5.FINAL_EVALUATIONS == 20
    assert d5.ROLLING_WINDOW == 8
    assert d5.BUDGET_FRACTION == 0.90
    assert d5.BUDGET_STEP_MULTIPLE == 5000
    assert d5.LN_VOCAB == math.log(60)
    assert "capacity_merge" not in source, "D5 must not depend on the merge"
    assert "smallest learnable width" in source
