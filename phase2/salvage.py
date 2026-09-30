"""Recover mid-training measurements from a batch's console log.

Batch 03 lost its last three cells to a machine reboot after nine had finished.
Their checkpoints survived, so everything measured *on the final model* can be
recomputed. What cannot is anything measured *during* training — the validation
curve and the stage-boundary diagnostics — because those were taken at steps
5,000 / 10,000 / 15,000 / 35,000 and the report that would have held them is only
written once every cell completes.

Those measurements do exist in the console log, and this module parses them out
so a resumed run can carry them rather than discard nine cells of work.

Two things about this are deliberately uncomfortable and are surfaced rather than
hidden. A console log is not an artifact: it has no checksum, no run id, and
nothing ties it to the checkpoints beyond the directory it sat next to. And the
salvaged stage-boundary rows are a strict *subset* of the recorded field — the
log prints accuracy, loss and shortcut agreement per depth, but not exact match,
distinct tokens or the majority-class frequency. Every salvaged value is
therefore tagged with its source so a reader can see which cells are
artifact-backed and which are log-backed, and `missing_fields` names what is
gone for good.

This is the same defect Batch 01's recorded deviation was about — a measurement
that lived in console output instead of a report field — arriving a second time
by a different route.
"""

import re
from typing import Dict, List, Optional

#: How a cell announces itself in the console log.
CELL_HEADER = re.compile(
    r"^=== (?P<name>\S+): target T=(?P<t>\d+) K=(?P<k>\d+) seed=(?P<seed>\d+)", re.M)

#: A periodic validation evaluation.
CURVE_LINE = re.compile(
    r"^ +step +(?P<step>\d+) \| train loss +(?P<train>[-\d.eE+]+) \| "
    r"val loss +(?P<val_loss>[-\d.eE+]+) \| val tok +(?P<val_acc>[-\d.]+)%", re.M)

#: A stage-boundary probe. The per-depth groups are parsed separately because
#: their number varies with the diagnostic depth list.
BOUNDARY_LINE = re.compile(
    r"^ +\[stage boundary\] step (?P<step>\d+) ends stage T=(?P<stage_t>\d+|\?): (?P<body>.+)$", re.M)
BOUNDARY_DEPTH = re.compile(
    r"T(?P<t>\d+) acc +(?P<acc>[-\d.]+)% loss +(?P<loss>[-\d.eE+]+)(?: sc +(?P<sc>[-\d.]+)%)?")

#: The post-training bound check.
BOUND_LINE = re.compile(
    r"^ +post-training z_L RMS +(?P<first>[\d.]+) -> +(?P<last>[\d.]+) "
    r"\(growth (?P<growth>[\d.]+)x\) (?P<verdict>bound held|BOUND BROKE)", re.M)

#: The per-cell summary.
BEST_LINE = re.compile(
    r"^ +best val token +(?P<acc>[-\d.]+)% +loss +(?P<loss>[-\d.eE+]+) +"
    r"distinct +(?P<distinct>\d+) +shortcut agreement +(?P<sc>[-\d.]+)%", re.M)

#: Fields the recorded stage-boundary artifact has that the log line does not.
#: Named so that a salvaged cell states what is gone rather than appearing whole.
BOUNDARY_MISSING_FIELDS = ("exact_match", "distinct_predicted_tokens", "majority_class_percent")


def read_log(path) -> str:
    """Read a console log, tolerating an unclean shutdown.

    A log whose writer was killed by a reboot can end in NUL padding, because
    the page cache was never flushed. Those bytes are dropped rather than
    allowed to turn the file into something `grep` and `str` treat as binary.
    """
    raw = open(path, "rb").read()
    return raw.replace(b"\x00", b"").decode("utf-8", "replace")


def split_cells(text: str) -> List[Dict]:
    """The log split into one entry per cell, in the order they ran."""
    matches = list(CELL_HEADER.finditer(text))
    cells = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        cells.append({
            "k": int(match.group("k")),
            "seed": int(match.group("seed")),
            "target_t": int(match.group("t")),
            "body": text[match.end():end],
        })
    return cells


def parse_curve(body: str) -> List[Dict]:
    return [{"step": int(m.group("step")),
             "val_loss": float(m.group("val_loss")),
             "val_token_accuracy": float(m.group("val_acc"))}
            for m in CURVE_LINE.finditer(body)]


def parse_boundaries(body: str) -> List[Dict]:
    """Stage-boundary rows, in the shape the report field uses.

    Only the fields the log actually printed are present. The rest are absent
    rather than defaulted, because a zero for `distinct_predicted_tokens` would
    read as a collapsed cell.
    """
    rows = []
    for match in BOUNDARY_LINE.finditer(body):
        stage = match.group("stage_t")
        by_t = []
        for depth in BOUNDARY_DEPTH.finditer(match.group("body")):
            row = {"t": int(depth.group("t")),
                   "token_accuracy": float(depth.group("acc")),
                   "loss": float(depth.group("loss"))}
            if depth.group("sc") is not None:
                row["shortcut_agreement_percent"] = float(depth.group("sc"))
            by_t.append(row)
        rows.append({"step": int(match.group("step")),
                     "stage_t": None if stage == "?" else int(stage),
                     "validation_by_t": by_t})
    return rows


def parse_bound(body: str) -> Optional[Dict]:
    match = BOUND_LINE.search(body)
    if match is None:
        return None
    return {"z_L_rms_first": float(match.group("first")),
            "z_L_rms_last": float(match.group("last")),
            "growth_ratio": float(match.group("growth")),
            "bound_held": match.group("verdict") == "bound held"}


def parse_best(body: str) -> Optional[Dict]:
    match = BEST_LINE.search(body)
    if match is None:
        return None
    return {"best_val_token_accuracy": float(match.group("acc")),
            "best_val_loss": float(match.group("loss")),
            "distinct_predicted_tokens": int(match.group("distinct")),
            "shortcut_agreement_percent": float(match.group("sc"))}


def salvage(path) -> Dict[tuple, Dict]:
    """Everything recoverable from a console log, keyed by (k, seed).

    A cell that did not finish is returned with ``complete: False`` and whatever
    partial curve it produced, so an interrupted cell is visible rather than
    silently absent.
    """
    out = {}
    for cell in split_cells(read_log(path)):
        best = parse_best(cell["body"])
        boundaries = parse_boundaries(cell["body"])
        out[(cell["k"], cell["seed"])] = {
            "k": cell["k"], "seed": cell["seed"], "target_t": cell["target_t"],
            "complete": best is not None,
            "curve": parse_curve(cell["body"]),
            "stage_boundary_validation": boundaries,
            "post_training_bound": parse_bound(cell["body"]),
            "console_summary": best,
            "source": "console_log",
            "stage_boundary_missing_fields": list(BOUNDARY_MISSING_FIELDS),
        }
    return out
