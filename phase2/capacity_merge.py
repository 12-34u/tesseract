"""Combine the P1b and P1b-2 capacity diagnostics into one report for D5.

P1b covers the small and medium widths; P1b-2 covers the large width, which was
added to the implementation after P1b had started. Both are the *same* diagnostic
— identical optimizer, batch, cadence, seed, T, K and step budget — run over
different widths, so their cells can be evaluated under one selection procedure
instead of two sequential decisions.

This module only reads. It never writes to, regenerates, or mutates a source
report or a run directory, and it never touches the running P1b. The merge is
deterministic: the same inputs always produce byte-identical output, widths are
ordered by measured parameter count, and nothing is inferred.

What it refuses to do, rather than papering over:

* merge reports whose protocol settings disagree (a different budget, cadence,
  seed, T, K set or vocabulary is a different experiment);
* merge when a declared K is missing from a width;
* merge when two sources claim the same width;
* merge when the same width/K cell appears twice with different numbers;
* merge a malformed report.

The combined report keeps every source's provenance, so any cell in it can be
traced back to the run that produced it. It carries no decision of its own: the
authoritative width and budget come from ``phase2.p1b_decision.evaluate_d5``,
whose criteria this module does not touch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Protocol settings that must agree for two diagnostics to be comparable. A
# difference in any of these means the cells were not produced under one
# protocol, and combining them would compare unlike with unlike.
PROTOCOL_KEYS = (
    "t",
    "n",
    "k_values",
    "seed",
    "optimizer",
    "batch_size",
    "eval_every",
    "max_steps",
    "early_stopping",
    "test_split_evaluated",
    "floor_threshold",
    "chance_level_val_loss_ln_vocab",
)

PROVENANCE_NOTE = (
    "P1b-2 was introduced after the Large model family was added to the implementation. It does not alter "
    "the original P1b data, protocol, or decision criteria. It supplies the same capacity diagnostic for the "
    "newly introduced research-scale width so that D5 can evaluate all candidate widths under one selection "
    "procedure."
)


class CapacityMergeError(ValueError):
    """The sources cannot be combined into one comparable diagnostic.

    Raised instead of returning a partial or reconciled report. A merge that
    silently dropped a cell or picked between conflicting numbers would put an
    unexplained value in front of the width decision.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityMergeError(message)


def _canonical(value: Any) -> str:
    """Stable text for equality checks and hashing, independent of key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================================
# Loading
# ============================================================================


def load_source(path, label: str) -> Dict[str, Any]:
    """Read one ``diagnostic_report.json`` plus the run metadata beside it."""
    path = Path(path)
    _require(path.is_file(), f"{label}: no diagnostic report at {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CapacityMergeError(f"{label}: {path} is not valid JSON: {exc}") from exc
    _require(isinstance(report, Mapping), f"{label}: report must be a JSON object")

    metadata = None
    meta_path = path.parent / "run_metadata.json"
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CapacityMergeError(f"{label}: {meta_path} is not valid JSON: {exc}") from exc

    return {
        "label": label,
        "report": report,
        "run_metadata": metadata,
        "report_path": str(path),
        "report_sha256": _sha256_text(path.read_text(encoding="utf-8")),
    }


# ============================================================================
# Validation
# ============================================================================


def _validate_source(source: Mapping) -> Dict[str, Any]:
    label, report = source["label"], source["report"]
    for key in ("settings", "runs", "model_configs", "single_seed_note"):
        _require(key in report, f"{label}: report is missing {key!r}")

    settings = report["settings"]
    _require(isinstance(settings, Mapping), f"{label}: settings must be an object")
    for key in PROTOCOL_KEYS:
        _require(key in settings, f"{label}: settings is missing {key!r}")

    runs = report["runs"]
    _require(isinstance(runs, Mapping) and runs, f"{label}: runs must be a non-empty object")

    k_values = settings["k_values"]
    _require(isinstance(k_values, Sequence) and not isinstance(k_values, str) and len(k_values) > 0,
             f"{label}: settings.k_values must be a non-empty list")
    declared = [int(k) for k in k_values]

    for width, per_k in runs.items():
        _require(isinstance(per_k, Mapping), f"{label}: runs.{width} must be an object")
        present = {int(k) for k in per_k}
        missing = sorted(set(declared) - present)
        _require(not missing, f"{label}: runs.{width} is missing K {missing} declared in settings.k_values")
        extra = sorted(present - set(declared))
        _require(not extra, f"{label}: runs.{width} has K {extra} not declared in settings.k_values")
        for k, run in per_k.items():
            _require(isinstance(run, Mapping), f"{label}: runs.{width}.{k} must be an object")
            _require("headline" in run, f"{label}: runs.{width}.{k} is missing 'headline'")
            _require("curve" in run, f"{label}: runs.{width}.{k} is missing 'curve'")
            headline = run["headline"]
            _require(isinstance(headline, Mapping), f"{label}: runs.{width}.{k}.headline must be an object")
            _require("parameter_count" in headline,
                     f"{label}: runs.{width}.{k}.headline is missing 'parameter_count'")
    return {"declared_k": declared, "settings": settings, "runs": runs}


def _check_protocols_match(sources: List[Mapping], validated: List[Mapping]) -> Dict[str, Any]:
    reference, ref_label = validated[0]["settings"], sources[0]["label"]
    for source, other in zip(sources[1:], validated[1:]):
        for key in PROTOCOL_KEYS:
            a, b = _canonical(reference[key]), _canonical(other["settings"][key])
            _require(a == b,
                     f"protocol mismatch on settings.{key}: {ref_label} has {a}, {source['label']} has {b}. "
                     "These diagnostics were not run under one protocol and must not be combined.")
    return dict(reference)


# ============================================================================
# Merge
# ============================================================================


def merge_capacity_reports(sources: Sequence[Mapping]) -> Dict[str, Any]:
    """Combine validated source reports into one capacity diagnostic.

    ``sources`` are the dicts returned by :func:`load_source`. Pure: it reads
    nothing from disk and mutates neither its inputs nor any artifact.
    """
    sources = list(sources)
    _require(len(sources) >= 2, "a combined report needs at least two source diagnostics")
    labels = [s["label"] for s in sources]
    _require(len(set(labels)) == len(labels), f"source labels must be unique, got {labels}")

    validated = [_validate_source(s) for s in sources]
    settings = _check_protocols_match(sources, validated)

    # One width may come from exactly one source, and a width/K cell from exactly
    # one run. Both duplicates and conflicts are refused rather than reconciled.
    owner: Dict[str, str] = {}
    cells: Dict[tuple, str] = {}
    for source, entry in zip(sources, validated):
        for width, per_k in entry["runs"].items():
            _require(width not in owner,
                     f"duplicate width {width!r}: present in both {owner.get(width)!r} and {source['label']!r}; "
                     "a width must be diagnosed by exactly one source")
            owner[width] = source["label"]
            for k, run in per_k.items():
                key = (width, int(k))
                fingerprint = _canonical(run)
                if key in cells:
                    _require(cells[key] == fingerprint,
                             f"conflicting cell {width}/K={k}: two sources report different results for it")
                    _require(False, f"duplicate cell {width}/K={k}")
                cells[key] = fingerprint

    merged_runs: Dict[str, Any] = {}
    merged_model_configs: Dict[str, Any] = {}
    provenance_by_width: Dict[str, Any] = {}
    for source, entry in zip(sources, validated):
        report = source["report"]
        run_metadata = source["run_metadata"] or {}
        for width in entry["runs"]:
            merged_runs[width] = entry["runs"][width]
            config = (report.get("model_configs") or {}).get(width)
            if config is not None:
                merged_model_configs[width] = config
            provenance_by_width[width] = {
                "source": source["label"],
                "report_path": source["report_path"],
                "report_sha256": source["report_sha256"],
                "run_id": run_metadata.get("run_id"),
                "run_status": run_metadata.get("status"),
                "run_verdict": run_metadata.get("verdict"),
                "seed": (report.get("settings") or {}).get("seed"),
                "git_commit": (run_metadata.get("git") or {}).get("commit"),
                "git_dirty": (run_metadata.get("git") or {}).get("dirty"),
                "gates_run_id": report.get("gates_run_id"),
                "gates_verdict": report.get("gates_verdict"),
            }

    # Deterministic ordering: widths by measured parameter count, K ascending.
    def parameter_count(width: str) -> int:
        per_k = merged_runs[width]
        counts = {int(((per_k[k] or {}).get("headline") or {}).get("parameter_count") or 0) for k in per_k}
        _require(len(counts) == 1,
                 f"{width}: parameter count differs across K ({sorted(counts)}); a Tesseract width has the "
                 "same parameter count at every K")
        return counts.pop()

    ordered_widths = sorted(merged_runs, key=lambda w: (parameter_count(w), w))
    ordered_runs = {
        width: {str(k): merged_runs[width][k] for k in sorted(merged_runs[width], key=lambda x: int(x))}
        for width in ordered_widths
    }

    cell_index = [
        {"width": width, "k": int(k), "source": provenance_by_width[width]["source"],
         "parameter_count": parameter_count(width)}
        for width in ordered_widths
        for k in sorted(ordered_runs[width], key=int)
    ]

    return {
        "kind": "combined_capacity_diagnostic",
        "section": "PHASE2_BENCHMARK_DESIGN.md §9 (P1b) + P1b-2 large-width extension",
        "scope": "capacity/learnability diagnostic; NOT evidence for the Tesseract depth hypothesis",
        "provenance_note": PROVENANCE_NOTE,
        "decision_criteria_unchanged": (
            "D5's criteria and selection rule are unchanged by this merge. D5 recommends the smallest width "
            "satisfying the pre-registered learnability criteria; adding a width to the candidate pool makes "
            "it eligible, never preferred."
        ),
        "sources": [
            {"label": s["label"], "report_path": s["report_path"], "report_sha256": s["report_sha256"],
             "widths": sorted((w for w in ordered_widths if provenance_by_width[w]["source"] == s["label"]),
                              key=lambda w: parameter_count(w)),
             "run_id": (s["run_metadata"] or {}).get("run_id"),
             "run_status": (s["run_metadata"] or {}).get("status")}
            for s in sources
        ],
        "settings": settings,
        "model_configs": {w: merged_model_configs[w] for w in ordered_widths if w in merged_model_configs},
        "runs": ordered_runs,
        "cells": cell_index,
        "provenance_by_width": {w: provenance_by_width[w] for w in ordered_widths},
        "single_seed_note": sources[0]["report"]["single_seed_note"],
    }


def merge_capacity_report_files(paths_by_label: Mapping[str, Any]) -> Dict[str, Any]:
    """Load each labelled ``diagnostic_report.json`` and merge them.

    Convenience wrapper. A missing file raises rather than producing a partial
    report, so an incomplete merge can never reach the width decision.
    """
    sources = [load_source(path, label) for label, path in sorted(paths_by_label.items())]
    return merge_capacity_reports(sources)


def expected_cells(report: Mapping) -> List[str]:
    """Human-readable ``width/K=k`` labels for every cell in a combined report."""
    return [f"{c['width']}/K={c['k']}" for c in report.get("cells", [])]


def missing_widths(report: Mapping, required: Sequence[str]) -> List[str]:
    """Required widths absent from a combined report, in the order given."""
    present = set((report.get("runs") or {}))
    return [w for w in required if w not in present]
