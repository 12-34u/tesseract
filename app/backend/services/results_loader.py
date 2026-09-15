"""Read-only access to Tesseract experiment artifacts for the dashboard.

The dashboard contributes no numbers of its own. Every value comes from files
written by the experiments under the runs root (``$TESSERACT_RUNS_DIR`` or
``<repo>/runs``). Each experiment's directory name is read from its config in
``configs/``. A value that was not recorded is returned as ``null``, never as
a default, and a file that cannot be parsed is reported as an error.

This module deliberately does not import torch or the research packages, so
the backend stays lightweight.
"""

import csv
import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
RUNS_DIR_ENV = "TESSERACT_RUNS_DIR"

EXPERIMENTS = {
    "crucible": "Crucible (Copy Overfitting + BPTT)",
    "k_scaling": "K-Scaling (Parameter Invariance + Latency)",
    "cellular_automaton": "Cellular Automaton (T × K)",
}

# Files the raw-data view may return. Fixed names only: no user-supplied paths.
RAW_FILES = {
    "crucible": ["summary.json", "bptt_verification.json", "metrics.csv", "config.yaml", "run_metadata.json"],
    "k_scaling": ["results.csv", "summary.txt", "config.yaml", "run_metadata.json"],
    "cellular_automaton": ["results.csv", "summary.txt", "config.yaml", "run_metadata.json"],
}

LEGACY_NOTE = (
    "No run_metadata.json: these artifacts predate run metadata, so the code "
    "version, seed and device that produced them cannot be verified."
)


class ArtifactError(Exception):
    """An artifact exists but is malformed."""


# ============================================================================
# File reading (cached by mtime/size so repeated requests do not re-read)
# ============================================================================


@lru_cache(maxsize=128)
def _read_cached(path: str, mtime_ns: int, size: int) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_text(path: Path) -> Optional[str]:
    """File contents, or None if the file does not exist."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return _read_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _json_safe(value: Any) -> Any:
    """NaN/Inf cannot be sent as JSON; they are missing values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> Optional[Any]:
    text = read_text(path)
    if text is None:
        return None
    try:
        return _json_safe(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path.name}: invalid JSON ({exc})") from exc


def read_yaml(path: Path) -> Optional[Any]:
    text = read_text(path)
    if text is None:
        return None
    try:
        return _json_safe(yaml.safe_load(text))
    except yaml.YAMLError as exc:
        raise ArtifactError(f"{path.name}: invalid YAML ({exc})") from exc


def _parse_number(value: str, where: str) -> Optional[float]:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        raise ArtifactError(f"{where}: {value!r} is not a number") from None
    if not math.isfinite(number):
        return None
    if number.is_integer() and value.lstrip("-").isdigit():
        return int(number)
    return number


def parse_csv(text: str, name: str, required: Set[str], text_columns: Set[str] = frozenset()) -> List[Dict[str, Any]]:
    """Parse a results CSV strictly.

    Every column must be numeric (or empty → None) unless listed in
    ``text_columns``. Missing required columns, ragged rows, non-numeric values
    and files with no data rows raise :class:`ArtifactError`.
    """
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    missing = sorted(required - set(header))
    if missing:
        raise ArtifactError(f"{name}: missing required columns {missing}")

    rows: List[Dict[str, Any]] = []
    for line, raw in enumerate(reader, start=2):
        if None in raw or any(v is None for v in raw.values()):
            raise ArtifactError(f"{name}: line {line} has the wrong number of fields")
        rows.append({
            key: (value or None) if key in text_columns else _parse_number(value, f"{name}: line {line}, column {key!r}")
            for key, value in raw.items()
        })
    if not rows:
        raise ArtifactError(f"{name}: no data rows")
    return rows


# ============================================================================
# Loader
# ============================================================================


class ResultsLoader:
    def __init__(self, runs_root: Optional[Path] = None, config_dir: Path = CONFIG_DIR) -> None:
        if runs_root is None:
            override = os.environ.get(RUNS_DIR_ENV)
            runs_root = Path(override).expanduser().resolve() if override else REPO_ROOT / "runs"
        self.runs_root = Path(runs_root)
        self.config_dir = Path(config_dir)

    # -- helpers ---------------------------------------------------------------

    def output_dir(self, experiment: str) -> Path:
        config = read_yaml(self.config_dir / f"{experiment}.yaml")
        try:
            output_dir = Path(config["experiment"]["output_dir"])
        except (TypeError, KeyError) as exc:
            raise ArtifactError(f"configs/{experiment}.yaml: missing experiment.output_dir") from exc
        return output_dir if output_dir.is_absolute() else self.runs_root / output_dir

    def _base(self, experiment: str, run_dir: Path) -> Dict[str, Any]:
        metadata = read_json(run_dir / "run_metadata.json")
        return {
            "experiment": experiment,
            "name": EXPERIMENTS[experiment],
            "output_dir": run_dir.name,
            "run": metadata,
            "provenance_note": None if metadata else LEGACY_NOTE,
            "config": read_yaml(run_dir / "config.yaml"),
        }

    def _load(self, experiment: str, primary: str, build) -> Dict[str, Any]:
        try:
            run_dir = self.output_dir(experiment)
            text = read_text(run_dir / primary)
            if text is None:
                # Metadata is still reported, so a running or crashed run is visible as such.
                return {**self._base(experiment, run_dir), "status": "missing",
                        "message": f"{run_dir.name}/{primary} not found — run the experiment first."}
            return {**self._base(experiment, run_dir), "status": "available", **build(run_dir, text)}
        except ArtifactError as exc:
            return {"experiment": experiment, "name": EXPERIMENTS[experiment], "status": "error", "message": str(exc)}

    # -- experiments -------------------------------------------------------------

    def get_crucible(self) -> Dict[str, Any]:
        def build(run_dir: Path, text: str) -> Dict[str, Any]:
            return {
                "metrics": parse_csv(text, "metrics.csv", {"step", "loss", "token_accuracy", "exact_match_accuracy"}),
                "summary": read_json(run_dir / "summary.json"),
                "bptt": read_json(run_dir / "bptt_verification.json"),
            }

        return self._load("crucible", "metrics.csv", build)

    def get_k_scaling(self) -> Dict[str, Any]:
        required = {"K", "parameter_count", "recursive_calls", "latency_mean_ms", "latency_median_ms", "latency_std_ms"}

        def build(run_dir: Path, text: str) -> Dict[str, Any]:
            return {"results": parse_csv(text, "results.csv", required, {"stopped_reason"})}

        return self._load("k_scaling", "results.csv", build)

    def get_cellular_automaton(self) -> Dict[str, Any]:
        required = {"T", "K", "parameter_count", "train_token_accuracy", "train_exact_match",
                    "val_token_accuracy", "val_exact_match", "training_steps"}

        def build(run_dir: Path, text: str) -> Dict[str, Any]:
            return {"results": parse_csv(text, "results.csv", required, {"stopped_reason"})}

        return self._load("cellular_automaton", "results.csv", build)

    # -- aggregate views -----------------------------------------------------------

    def get_experiments(self) -> List[Dict[str, Any]]:
        return [self._status_entry(payload) for payload in self._all().values()]

    def get_summary(self) -> Dict[str, Any]:
        data = self._all()
        k_scaling, crucible = data["k_scaling"], data["crucible"]
        k_rows = k_scaling.get("results") or []

        model = None
        for payload in (k_scaling, crucible, data["cellular_automaton"]):
            candidate = (payload.get("config") or {}).get("model")
            if isinstance(candidate, dict) and "d_model" in candidate:
                model = {key: candidate.get(key) for key in ("d_model", "num_heads", "d_ff", "max_seq_len", "alpha", "dropout", "vocab_size")}
                model["source"] = f"{payload['output_dir']}/config.yaml"
                break

        counts = sorted({r["parameter_count"] for r in k_rows if r["parameter_count"] is not None})
        run = k_scaling.get("run") or {}
        return {
            "model": model,
            "parameter_counts": counts,
            "parameter_count": counts[0] if len(counts) == 1 else None,
            "k_tested": sorted(r["K"] for r in k_rows),
            "transformer_block_instances": (run.get("results") or {}).get("transformer_block_instances"),
            "device": (run.get("device") or {}).get("type"),
            "experiments": [self._status_entry(p) for p in data.values()],
            "findings": derive_findings(data),
        }

    def get_raw_results(self) -> Dict[str, Dict[str, Optional[str]]]:
        raw: Dict[str, Dict[str, Optional[str]]] = {}
        for experiment, names in RAW_FILES.items():
            try:
                run_dir = self.output_dir(experiment)
            except ArtifactError as exc:
                raw[experiment] = {"error": str(exc)}
                continue
            raw[experiment] = {name: read_text(run_dir / name) for name in names}
        return raw

    def _all(self) -> Dict[str, Dict[str, Any]]:
        return {
            "crucible": self.get_crucible(),
            "k_scaling": self.get_k_scaling(),
            "cellular_automaton": self.get_cellular_automaton(),
        }

    @staticmethod
    def _status_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
        run = payload.get("run") or {}
        return {
            "id": payload["experiment"],
            "name": payload["name"],
            "artifact_status": payload["status"],  # available | missing | error
            "message": payload.get("message"),
            "run_status": run.get("status"),  # running | completed | failed | None (legacy)
            "verdict": run.get("verdict"),
            "run_id": run.get("run_id"),
            "finished_at": run.get("finished_at"),
            "git_commit": (run.get("git") or {}).get("commit"),
            "git_dirty": (run.get("git") or {}).get("dirty"),
            "provenance_note": payload.get("provenance_note"),
        }


# ============================================================================
# Findings derived from artifacts
# ============================================================================


def _finding(level: str, text: str, source: str) -> Dict[str, str]:
    return {"level": level, "text": text, "source": source}


def derive_findings(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Plain-language statements computed from the artifacts (level: ok | warn | info)."""
    findings: List[Dict[str, str]] = []

    k = data["k_scaling"]
    if k["status"] == "available":
        rows = k["results"]
        src = f"{k['output_dir']}/results.csv"
        ks = [r["K"] for r in rows]
        counts = {r["parameter_count"] for r in rows}
        if len(counts) == 1:
            findings.append(_finding("ok", f"Trainable parameter count is identical ({next(iter(counts)):,}) for K ∈ {ks}.", src))
        else:
            findings.append(_finding("warn", f"Trainable parameter count differs across K: {sorted(counts)}.", src))
        if k.get("run"):
            matches = all(r["recursive_calls"] == r["K"] for r in rows)
            findings.append(_finding("ok" if matches else "warn",
                                     "Measured shared-block executions equal K for every model." if matches
                                     else "Measured shared-block executions do NOT equal K.", src))
        else:
            findings.append(_finding("warn", "recursive_calls in this legacy results.csv were not recorded as measured values.", src))
        by_median = sorted(rows, key=lambda r: r["K"])
        device = ((k.get("run") or {}).get("device") or {}).get("type", "unrecorded device")
        findings.append(_finding(
            "info",
            f"Median forward latency: {by_median[0]['latency_median_ms']:.2f} ms at K={by_median[0]['K']} → "
            f"{by_median[-1]['latency_median_ms']:.2f} ms at K={by_median[-1]['K']} ({device}; wall-clock, not FLOPs).",
            src,
        ))

    c = data["crucible"]
    if c["status"] == "available":
        s = c.get("summary")
        if s:
            findings.append(_finding(
                "ok" if s["verdict"] == "PASS" else "warn",
                f"Crucible {s['verdict']}: final loss {s['final_loss']:.6f}, token accuracy {s['final_token_accuracy']:.1f}%, "
                f"exact match {s['final_exact_match_accuracy']:.1f}% after {s['training_steps']} steps ({s['stopped_reason']}).",
                f"{c['output_dir']}/summary.json",
            ))
        else:
            findings.append(_finding("warn", "Crucible has metrics.csv but no summary.json; no verdict was recorded.", c["output_dir"]))
        b = c.get("bptt")
        if b:
            findings.append(_finding(
                "ok" if b["passed"] else "warn",
                f"BPTT verification {'passed' if b['passed'] else 'FAILED'}: gradient reached z_L at all {b['k']} recursive steps."
                if b["passed"] else f"BPTT verification FAILED: {b['failures']}",
                f"{c['output_dir']}/bptt_verification.json",
            ))

    ca = data["cellular_automaton"]
    if ca["status"] == "available":
        rows = ca["results"]
        src = f"{ca['output_dir']}/results.csv"
        train_em = [r["train_exact_match"] for r in rows if r["train_exact_match"] is not None]
        val_em = [r["val_exact_match"] for r in rows if r["val_exact_match"] is not None]
        val_tok = [r["val_token_accuracy"] for r in rows if r["val_token_accuracy"] is not None]
        if train_em and val_em and val_tok:
            findings.append(_finding(
                "warn" if max(val_em) == 0.0 else "info",
                f"Cellular automaton: train exact match {min(train_em):.1f}–{max(train_em):.1f}%, validation exact match "
                f"{min(val_em):.1f}–{max(val_em):.1f}%, validation token accuracy {min(val_tok):.1f}–{max(val_tok):.1f}% "
                f"(chance ≈ 50%) across {len(rows)} (T, K) cells.",
                src,
            ))
    return findings
