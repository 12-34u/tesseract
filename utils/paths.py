"""Repository-relative path resolution for Tesseract.

All paths are derived from this file's location, so experiments behave the
same regardless of the current working directory. The runs root can be moved
with the ``TESSERACT_RUNS_DIR`` environment variable.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
RUNS_DIR_ENV = "TESSERACT_RUNS_DIR"


def runs_root() -> Path:
    """Return the directory that holds experiment run artifacts."""
    override = os.environ.get(RUNS_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "runs"


def resolve_run_dir(output_dir: str | Path) -> Path:
    """Resolve an experiment ``output_dir``: absolute paths are kept, relative
    paths are placed under :func:`runs_root`."""
    path = Path(output_dir).expanduser()
    return path if path.is_absolute() else runs_root() / path
