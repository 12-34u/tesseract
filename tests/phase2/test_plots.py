"""Phase 2 figures and report render from analysis output and saved curves."""

import csv
import json

from phase2.config import load_amendment, load_amendment_02
from phase2.inference import analyze
from phase2.plots import load_curves, report_markdown, write_report
from tests.phase2.test_inference import SUPPORT, result, tesseract_cells


def _write_cells(root, results):
    for r in results:
        c = r["cell"]
        d = root / f"{c['family']}_d{c['depth']}_{c['task']}_T{c['t']}_s{c['seed']}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(r))
        with open(d / "val_curve.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "val_chance_normalised_token_accuracy", "val_loss"])
            writer.writeheader()
            for step in (250, 500, 750):
                writer.writerow({"step": step, "val_chance_normalised_token_accuracy": r["test"]["chance_normalised_token_accuracy"] * step / 750,
                                 "val_loss": 4.0})


def test_report_and_figures(tmp_path):
    results = tesseract_cells(SUPPORT) + [result("width_scaled", "main", 8, 8, s, 0.5) for s in range(3)]
    analysis = analyze(results, load_amendment(), load_amendment_02().decision_rule)
    _write_cells(tmp_path / "cells", results)
    curves = load_curves(tmp_path / "cells")
    assert len(curves["tesseract/main/8/8"]) == 3
    paths = write_report(analysis, tmp_path / "out", curves)
    assert [p.name for p in paths] == ["fig1_delta_vs_t.png", "fig2_score_vs_k.png", "fig3_score_vs_flops.png",
                                       "fig4_learning_curves.png", "report.md"]
    assert all(p.stat().st_size > 0 for p in paths)
    text = report_markdown(analysis)
    assert "supports_sequential_depth_effect" in text and "## final checkpoint" in text


def test_figures_render_with_sparse_data(tmp_path):
    analysis = analyze([result("tesseract", "main", 4, 1, 0, 0.1)], load_amendment(), load_amendment_02().decision_rule)
    paths = write_report(analysis, tmp_path, {})
    assert all(p.is_file() for p in paths)
