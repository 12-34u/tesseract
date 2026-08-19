#!/usr/bin/env python3
"""Generate high-resolution presentation graphs for Tesseract August 21 Milestone.

Outputs:
  - runs/presentation/param_count_vs_k.png
  - runs/presentation/latency_vs_k.png
  - runs/presentation/crucible_loss.png
  - runs/presentation/gradient_norms_bptt.png
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

output_dir = os.path.join(repo_root, "runs", "presentation")
os.makedirs(output_dir, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18,
})


def plot_param_count_vs_k():
    k_vals = [1, 2, 4, 8]
    params = [210832, 210832, 210832, 210832]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(k_vals, params, marker="o", color="#1f77b4", linewidth=3, markersize=10, label="Trainable Parameters")

    for x, y in zip(k_vals, params):
        plt.annotate(
            f"{y:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontweight="bold",
            fontsize=11,
            color="#1f77b4",
        )

    plt.title("Tesseract: Parameter Count Invariance vs Recursive Depth K", pad=15)
    plt.xlabel("Recursive Depth (K)")
    plt.ylabel("Trainable Parameters")
    plt.xticks(k_vals)
    plt.ylim(180000, 240000)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "param_count_vs_k.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_latency_vs_k():
    k_vals = [1, 2, 4, 8]
    latency_ms = [1.32, 2.49, 5.03, 9.54]

    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar([str(k) for k in k_vals], latency_ms, color="#2ca02c", alpha=0.85, width=0.5, edgecolor="black")

    for bar, lat in zip(bars, latency_ms):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.2,
            f"{lat:.2f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.title("Tesseract: Forward Pass Latency vs Recursive Depth K (CPU)", pad=15)
    plt.xlabel("Recursive Depth (K)")
    plt.ylabel("Forward Latency (ms)")
    plt.ylim(0, 11.5)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "latency_vs_k.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_crucible_loss():
    metrics_path = os.path.join(repo_root, "runs", "crucible_k04", "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found at {metrics_path}")
        return

    steps = []
    loss = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(steps, loss, color="#d62728", linewidth=2.5, label="Training Loss")
    plt.axhline(0.01, color="black", linestyle=":", linewidth=1.5, label="Early Stop Target (0.01)")

    plt.title("Tesseract Crucible: Loss Convergence (K=4)", pad=15)
    plt.xlabel("Training Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.yscale("log")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "crucible_loss.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_gradient_norms_bptt():
    steps = [1, 2, 3, 4]
    grad_norms = [0.001346, 0.001066, 0.000900, 0.000820]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(steps, grad_norms, marker="s", color="#9467bd", linewidth=2.5, markersize=9, label=r"$\| \partial L / \partial z_L^{(k)} \|$")

    for x, y in zip(steps, grad_norms):
        plt.annotate(
            f"{y:.6f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=10,
            color="#9467bd",
        )

    plt.title("Tesseract BPTT: Recursive Latency State Gradient Norms (K=4)", pad=15)
    plt.xlabel("Recursive Step (k)")
    plt.ylabel(r"State Gradient Norm $\| \partial L / \partial z_L^{(k)} \|$")
    plt.xticks(steps, [r"$z_L^{(1)}$", r"$z_L^{(2)}$", r"$z_L^{(3)}$", r"$z_L^{(4)}$"])
    plt.ylim(0.0005, 0.0016)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "gradient_norms_bptt.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Generating presentation assets in runs/presentation/...")
    plot_param_count_vs_k()
    plot_latency_vs_k()
    plot_crucible_loss()
    plot_gradient_norms_bptt()
    print("✓ All presentation assets generated successfully.")


if __name__ == "__main__":
    main()
