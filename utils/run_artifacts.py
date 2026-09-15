"""Run-directory management and reproducibility metadata for experiments.

Every experiment run owns one directory and declares the artifacts it writes.
At start, those artifacts left over from a previous run are removed, so a
partial or crashed run can never leave old files that look like new results.
``run_metadata.json`` records what produced the directory and whether the run
completed.
"""

import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch
import yaml

from utils.config import config_to_dict
from utils.device import describe_device
from utils.paths import REPO_ROOT

METADATA_FILENAME = "run_metadata.json"
CONFIG_FILENAME = "config.yaml"
SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_json_safe(value: Any) -> Any:
    """Recursively convert to strict JSON: tuples → lists, NaN/±Inf → None.

    Strict JSON (and Starlette's JSONResponse) cannot represent non-finite
    floats; ``None`` marks the value as missing rather than inventing a number.
    """
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(data), f, indent=2, allow_nan=False)
        f.write("\n")
    return path


def git_state() -> dict[str, Any]:
    """Return the current commit and whether the working tree has local changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return {"commit": None, "dirty": None, "note": "git executable not found"}
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "note": head.stderr.strip()}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


class RunRecorder:
    """Owns one experiment run directory.

    Usage::

        with RunRecorder(run_dir, "crucible", config, config_path, device, ARTIFACTS) as run:
            ...                                  # write files via run.path(name)
            run.finish(verdict="PASS", results={...})

    If the block raises, or exits without ``finish()``, the metadata is marked
    ``failed`` and any exception propagates.
    """

    def __init__(
        self,
        run_dir: Path,
        experiment: str,
        config: Any,
        config_path: Path,
        device: torch.device,
        artifacts: list[str],
    ) -> None:
        reserved = {METADATA_FILENAME, CONFIG_FILENAME}
        if reserved & set(artifacts):
            raise ValueError(f"artifact names {sorted(reserved)} are managed by RunRecorder")
        self.run_dir = Path(run_dir)
        self.experiment = experiment
        self.config = config
        self.config_path = Path(config_path)
        self.device = device
        self.artifacts = list(artifacts)
        self._started = _utc_now()
        self._metadata: dict[str, Any] = {}
        self._finished = False

    # -- paths -----------------------------------------------------------------

    def path(self, name: str) -> Path:
        """Path of a declared artifact. Undeclared names are rejected so that
        every file written is also cleaned up by the next run."""
        if name not in self.artifacts:
            raise ValueError(f"{name!r} is not a declared artifact of {self.experiment}: {self.artifacts}")
        return self.run_dir / name

    # -- lifecycle -------------------------------------------------------------

    def __enter__(self) -> "RunRecorder":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in [*self.artifacts, METADATA_FILENAME, CONFIG_FILENAME]:
            target = self.run_dir / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

        with open(self.run_dir / CONFIG_FILENAME, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_to_dict(self.config), f, sort_keys=False)

        try:
            config_file = str(self.config_path.relative_to(REPO_ROOT))
        except ValueError:
            config_file = str(self.config_path)

        self._metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": f"{self.experiment}-{self._started.strftime('%Y%m%dT%H%M%SZ')}",
            "experiment": self.experiment,
            "status": "running",
            "verdict": None,
            "started_at": self._started.isoformat(timespec="seconds"),
            "finished_at": None,
            "config_file": config_file,
            "seed": self.config.experiment.seed,
            "device": describe_device(self.device),
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "git": git_state(),
            "command": sys.argv,
            "results": None,
            "error": None,
        }
        self._write_metadata()
        return self

    def finish(self, verdict: str, results: dict[str, Any]) -> None:
        self._metadata.update(
            status="completed",
            verdict=verdict,
            results=results,
            finished_at=_utc_now().isoformat(timespec="seconds"),
        )
        self._write_metadata()
        self._finished = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is not None or not self._finished:
            self._metadata.update(
                status="failed",
                finished_at=_utc_now().isoformat(timespec="seconds"),
                error=f"{exc_type.__name__}: {exc}" if exc is not None else "run exited without finish()",
            )
            self._write_metadata()
        return False  # never suppress exceptions

    def _write_metadata(self) -> None:
        write_json(self.run_dir / METADATA_FILENAME, self._metadata)
