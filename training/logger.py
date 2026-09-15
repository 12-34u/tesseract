"""Experiment logger for Tesseract training runs: in-memory history → CSV + plots."""

import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt


def _natural_key(name: str) -> tuple:
    """Sort 'grad_zL_step_10' after 'grad_zL_step_2'."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


class ExperimentLogger:
    """Collects per-step metric dicts and writes them as CSV and plots.

    Args:
        run_dir: Directory for CSV and plot files.
        experiment_name: Used in plot titles.
    """

    def __init__(self, run_dir: str | Path, experiment_name: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self.history.append({"step": step, **metrics})

    def _keys(self) -> List[str]:
        keys = set()
        for record in self.history:
            keys.update(record.keys())
        return sorted(keys, key=_natural_key)

    def save_csv(self, path: Path) -> Path:
        """Write all logged records. Absent/None values become empty cells."""
        if not self.history:
            raise ValueError("No metrics have been logged; refusing to write an empty CSV.")
        fieldnames = ["step"] + [k for k in self._keys() if k != "step"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        return path

    def _series(self, key: str) -> tuple[list, list]:
        pairs = [(r["step"], r[key]) for r in self.history if _is_number(r.get(key))]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def plot_loss_curve(self, path: Path) -> Path:
        steps, losses = self._series("loss")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, losses, linewidth=1.5, color="#2196F3")
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax.set_title(f"{self.experiment_name} — Loss Curve", fontsize=14)
        ax.grid(True, alpha=0.3)
        if losses and max(losses) / max(min(losses), 1e-10) > 100:
            ax.set_yscale("log")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_gradient_norms(self, path: Path) -> Path:
        """Plot every recorded ``grad_z*`` series (keys gathered from ALL records)."""
        keys = [k for k in self._keys() if k.startswith("grad_z")]
        if not keys:
            raise ValueError("No recursive-state gradient norms were logged (instrumentation disabled?).")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0", "#00BCD4", "#FF5722", "#795548"]
        for i, key in enumerate(keys):
            steps, values = self._series(key)
            if values:
                ax.plot(steps, values, linewidth=1.2, marker=".", label=key, color=colors[i % len(colors)])
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient Norm", fontsize=12)
        ax.set_title(f"{self.experiment_name} — Recursive State Gradient Norms", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_accuracy(self, path: Path) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label, color in (
            ("token_accuracy", "Token Accuracy", "#4CAF50"),
            ("exact_match_accuracy", "Exact-Match Accuracy", "#E91E63"),
        ):
            steps, values = self._series(key)
            if values:
                ax.plot(steps, values, linewidth=1.5, label=label, color=color)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"{self.experiment_name} — Accuracy Curves", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
