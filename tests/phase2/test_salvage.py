"""Recovering mid-training measurements from a console log.

Batch 03 lost three cells to a reboot after nine had finished, and the report is
only written once every cell completes — so nine cells' curves and
stage-boundary diagnostics existed only in the console log. These tests pin the
parser that recovers them, and the two honesty properties that matter more than
the parsing: an interrupted cell must be visible as incomplete rather than
absent, and a salvaged boundary row must not look like a whole one.
"""

import pytest

from phase2.salvage import (
    BOUNDARY_MISSING_FIELDS,
    parse_best,
    parse_bound,
    parse_boundaries,
    parse_curve,
    salvage,
    split_cells,
)

LOG = """=== Batch 03 gate: per-step hidden-state RMS must be bounded at every K
  K=1: rms 1.0000 -> 1.0000  growth 1.0000x  OK
  bounded-growth gate: PASS
=== t3_k_sweep: target T=3 K=1 seed=0 steps=35000 fix=per_step(non-affine) curriculum=[1, 2, 3]->3
    step    250 | train loss 3.9000 | val loss 4.0900 | val tok   1.70% (norm 0.000) | val EM   0.00% | grad max 0.4
    step  35000 | train loss 0.0010 | val loss 1.2000 | val tok  64.00% (norm 0.634) | val EM   0.00% | grad max 0.9
    [stage boundary] step 5000 ends stage T=1: T1 acc 100.0000% loss 0.0001  T2 acc 1.7636% loss 22.5986 sc 1.64%  T3 acc 1.6401% loss 22.7321 sc 1.61%
    post-training z_L RMS 1.0000 -> 1.0000 (growth 1.0000x) bound held
    best val token 64.5393%  loss 1.2148  distinct 60  shortcut agreement 19.91%
=== t3_k_sweep: target T=3 K=8 seed=0 steps=35000 fix=per_step(non-affine) curriculum=[1, 2, 3]->3
    step    250 | train loss 4.1000 | val loss 4.1000 | val tok   1.64% (norm 0.000) | val EM   0.00% | grad max 0.5
"""


def test_cells_are_split_in_run_order_and_the_gate_header_is_not_one():
    cells = split_cells(LOG)
    assert [(c["k"], c["seed"]) for c in cells] == [(1, 0), (8, 0)]
    assert all(c["target_t"] == 3 for c in cells)


def test_an_interrupted_cell_is_incomplete_rather_than_absent():
    """A cell killed mid-run must be visible, or a resume would silently skip it."""
    out = salvage_text(LOG)
    assert out[(1, 0)]["complete"] is True
    assert out[(8, 0)]["complete"] is False
    assert out[(8, 0)]["console_summary"] is None
    assert out[(8, 0)]["post_training_bound"] is None
    assert len(out[(8, 0)]["curve"]) == 1


def test_curve_points_carry_step_loss_and_accuracy():
    curve = parse_curve(split_cells(LOG)[0]["body"])
    assert curve == [
        {"step": 250, "val_loss": 4.09, "val_token_accuracy": 1.70},
        {"step": 35000, "val_loss": 1.20, "val_token_accuracy": 64.00},
    ]


def test_boundary_rows_keep_shortcut_agreement_only_where_the_log_printed_it():
    """T=1 has no G2 candidate, so its row must have no agreement field at all
    rather than a zero, which would read as a model that never agrees."""
    rows = parse_boundaries(split_cells(LOG)[0]["body"])
    assert len(rows) == 1
    assert rows[0]["step"] == 5000 and rows[0]["stage_t"] == 1
    by_t = {r["t"]: r for r in rows[0]["validation_by_t"]}
    assert by_t[1]["token_accuracy"] == 100.0
    assert "shortcut_agreement_percent" not in by_t[1]
    assert by_t[2]["shortcut_agreement_percent"] == 26.74 or by_t[2]["shortcut_agreement_percent"] == 1.64
    assert set(by_t) == {1, 2, 3}


def test_a_salvaged_cell_names_the_fields_the_log_did_not_print():
    """Otherwise a recovered cell reads as though it were artifact-complete."""
    out = salvage_text(LOG)
    assert out[(1, 0)]["stage_boundary_missing_fields"] == list(BOUNDARY_MISSING_FIELDS)
    assert "distinct_predicted_tokens" in BOUNDARY_MISSING_FIELDS
    assert out[(1, 0)]["source"] == "console_log"


def test_bound_and_best_lines_are_parsed():
    body = split_cells(LOG)[0]["body"]
    assert parse_bound(body) == {"z_L_rms_first": 1.0, "z_L_rms_last": 1.0,
                                 "growth_ratio": 1.0, "bound_held": True}
    assert parse_best(body)["best_val_token_accuracy"] == 64.5393
    assert parse_best(body)["distinct_predicted_tokens"] == 60


def test_a_broken_bound_is_parsed_as_broken():
    body = "    post-training z_L RMS 1.0000 -> 9.5000 (growth 9.5000x) BOUND BROKE\n"
    assert parse_bound(body)["bound_held"] is False
    assert parse_bound(body)["growth_ratio"] == 9.5


def test_nul_padding_from_an_unclean_shutdown_is_tolerated(tmp_path):
    """A log whose writer was killed by a reboot can end in unflushed NULs."""
    path = tmp_path / "log"
    path.write_bytes(LOG.encode() + b"\x00" * 512)
    out = salvage(path)
    assert (1, 0) in out and out[(1, 0)]["complete"] is True


def salvage_text(text):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log"
        path.write_text(text, encoding="utf-8")
        return salvage(path)
