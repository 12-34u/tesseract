"""Experiment Logger for Tesseract training runs.

Provides a unified logging interface supporting:
    - Console output with formatted metrics
    - CSV file logging for post-experiment analysis
    - Matplotlib plot generation (loss curves, gradient norms)

W&B integration is attempted but falls back to local-only if unavailable.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt


class ExperimentLogger:
    """Simple experiment logger with CSV + matplotlib support.

    Args:
        run_dir (str | Path): Directory for saving logs and plots.
        experiment_name (str): Name of the experiment run.
        use_wandb (bool): Whether to attempt W&B logging. Default: False.
    """

    def __init__(
        self,
        run_dir: str | Path,
        experiment_name: str = "tesseract_experiment",
        use_wandb: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history: List[Dict[str, Any]] = []
        self.wandb_run = None

        # Attempt W&B initialization
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project="tesseract",
                    name=experiment_name,
                    reinit=True,
                )
            except Exception:
                print("[Logger] W&B unavailable — falling back to local CSV logging.")
                self.wandb_run = None

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a dictionary of metrics for a given training step.

        Args:
            metrics: Dict of metric_name → value.
            step: Current training step number.
        """
        record = {"step": step, **metrics}
        self.history.append(record)

        # W&B logging
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(record, step=step)
            except Exception:
                pass

    def save_csv(self, filename: str = "metrics.csv") -> Path:
        """Save all logged metrics to a CSV file.

        Args:
            filename: Name of the CSV file.

        Returns:
            Path to the saved CSV file.
        """
        if not self.history:
            return self.run_dir / filename

        filepath = self.run_dir / filename
        fieldnames = list(self.history[0].keys())

        # Collect all unique keys across all records
        all_keys = set()
        for record in self.history:
            all_keys.update(record.keys())
        fieldnames = sorted(all_keys)

        # Ensure 'step' is first
        if "step" in fieldnames:
            fieldnames.remove("step")
            fieldnames = ["step"] + fieldnames

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in self.history:
                writer.writerow(record)

        return filepath

    def plot_loss_curve(self, filename: str = "loss_curve.png") -> Path:
        """Generate and save a loss vs. training step plot.

        Args:
            filename: Name of the output PNG file.

        Returns:
            Path to the saved plot.
        """
        steps = [r["step"] for r in self.history if "loss" in r]
        losses = [r["loss"] for r in self.history if "loss" in r]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, losses, linewidth=1.5, color="#2196F3")
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax.set_title("Tesseract Crucible — Loss Curve", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Log scale if loss spans large range
        if losses and max(losses) / max(min(losses), 1e-10) > 100:
            ax.set_yscale("log")

        filepath = self.run_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def plot_gradient_norms(
        self,
        filename: str = "gradient_norms.png",
        keys: Optional[List[str]] = None,
    ) -> Path:
        """Generate and save gradient norm plots across recursive steps.

        Args:
            filename: Name of the output PNG file.
            keys: List of metric keys to plot. Auto-detects grad_zL_step_* if None.

        Returns:
            Path to the saved plot.
        """
        if keys is None:
            # Auto-detect gradient keys
            if self.history:
                keys = sorted([
                    k for k in self.history[-1].keys()
                    if k.startswith("grad_z")
                ])

        if not keys:
            # Nothing to plot
            filepath = self.run_dir / filename
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No gradient data", ha="center", va="center")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return filepath

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0", "#00BCD4", "#FF5722", "#795548"]

        for i, key in enumerate(keys):
            steps = [r["step"] for r in self.history if key in r]
            values = [r[key] for r in self.history if key in r]
            color = colors[i % len(colors)]
            ax.plot(steps, values, linewidth=1.2, label=key, color=color)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient Norm", fontsize=12)
        ax.set_title("Tesseract Crucible — Recursive State Gradient Norms", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        filepath = self.run_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def plot_accuracy(self, filename: str = "accuracy_curve.png") -> Path:
        """Generate and save accuracy curves (token + exact-match).

        Returns:
            Path to the saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        steps_tok = [r["step"] for r in self.history if "token_accuracy" in r]
        tok_acc = [r["token_accuracy"] for r in self.history if "token_accuracy" in r]

        steps_em = [r["step"] for r in self.history if "exact_match_accuracy" in r]
        em_acc = [r["exact_match_accuracy"] for r in self.history if "exact_match_accuracy" in r]

        if tok_acc:
            ax.plot(steps_tok, tok_acc, linewidth=1.5, label="Token Accuracy", color="#4CAF50")
        if em_acc:
            ax.plot(steps_em, em_acc, linewidth=1.5, label="Exact-Match Accuracy", color="#E91E63")

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Tesseract Crucible — Accuracy Curves", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

        filepath = self.run_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def finish(self) -> None:
        """Finalize the logger and close W&B run if active."""
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
