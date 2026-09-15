#!/usr/bin/env python3
"""Generate high-resolution presentation figures from experiment artifacts.

Every plotted number is read from run artifacts; nothing is typed in:
  runs/k_scaling/results.csv               → param_count_vs_k.png, latency_vs_k.png
  runs/crucible_k04/metrics.csv            → crucible_loss.png
  runs/crucible_k04/bptt_verification.json → gradient_norms_bptt.png

Outputs go to <runs root>/presentation/. Exits 1 if a required artifact is missing.

Usage:
    python scripts/generate_presentation_assets.py
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.paths import runs_root

DPI = 300


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def device_label(run_dir: Path) -> str:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.is_file():
        return "device unrecorded"
    return json.loads(meta_path.read_text(encoding="utf-8"))["device"]["type"].upper()


def plot_param_count_vs_k(rows: List[Dict[str, str]], out: Path) -> None:
    ks = [int(r["K"]) for r in rows]
    params = [int(r["parameter_count"]) for r in rows]
    plt.figure(figsize=(8, 5), dpi=DPI)
    plt.plot(ks, params, marker="o", color="#1f77b4", linewidth=3, markersize=10)
    for x, y in zip(ks, params):
        plt.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontweight="bold")
    margin = max(params) * 0.15
    plt.ylim(min(params) - margin, max(params) + margin)
    plt.title("Tesseract: Parameter Count vs Recursive Depth K", pad=15)
    plt.xlabel("Recursive Depth (K)")
    plt.ylabel("Trainable Parameters")
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved: {out}")


def plot_latency_vs_k(rows: List[Dict[str, str]], device: str, out: Path) -> None:
    ks = [r["K"] for r in rows]
    medians = [float(r["latency_median_ms"]) for r in rows]
    plt.figure(figsize=(8, 5), dpi=DPI)
    bars = plt.bar(ks, medians, color="#2ca02c", alpha=0.85, width=0.5, edgecolor="black")
    for bar, value in zip(bars, medians):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f} ms",
                 ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, max(medians) * 1.2)
    plt.title(f"Tesseract: Median Forward Latency vs K ({device})", pad=15)
    plt.xlabel("Recursive Depth (K)")
    plt.ylabel("Forward Latency (ms)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved: {out}")


def plot_crucible_loss(rows: List[Dict[str, str]], early_stop_loss: float, out: Path) -> None:
    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    plt.figure(figsize=(8, 5), dpi=DPI)
    plt.plot(steps, losses, color="#d62728", linewidth=2.5, label="Training Loss")
    plt.axhline(early_stop_loss, color="black", linestyle=":", linewidth=1.5, label=f"Early stop ({early_stop_loss})")
    plt.title("Tesseract Crucible: Loss Convergence", pad=15)
    plt.xlabel("Training Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved: {out}")


def plot_bptt_gradients(bptt: dict, out: Path) -> None:
    steps = [s["step"] for s in bptt["steps"]]
    norms = [s["grad_z_L_norm"] for s in bptt["steps"]]
    plt.figure(figsize=(8, 5), dpi=DPI)
    plt.plot(steps, norms, marker="s", color="#9467bd", linewidth=2.5, markersize=9)
    for x, y in zip(steps, norms):
        plt.annotate(f"{y:.6f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")
    plt.title(f"Tesseract BPTT: Recursive State Gradient Norms (K={bptt['k']})", pad=15)
    plt.xlabel("Recursive Step (k)")
    plt.ylabel(r"$\| \partial L / \partial z_L^{(k)} \|$")
    plt.xticks(steps, [rf"$z_L^{{({s})}}$" for s in steps])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"Saved: {out}")


def main() -> int:
    runs = runs_root()
    k_dir, crucible_dir = runs / "k_scaling", runs / "crucible_k04"
    required = [k_dir / "results.csv", crucible_dir / "metrics.csv", crucible_dir / "bptt_verification.json",
                crucible_dir / "summary.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("Missing experiment artifacts (run scripts/run_all_experiments.py first):", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    out_dir = runs / "presentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    k_rows = read_csv(k_dir / "results.csv")
    crucible_summary = json.loads((crucible_dir / "summary.json").read_text(encoding="utf-8"))

    plot_param_count_vs_k(k_rows, out_dir / "param_count_vs_k.png")
    plot_latency_vs_k(k_rows, device_label(k_dir), out_dir / "latency_vs_k.png")
    plot_crucible_loss(read_csv(crucible_dir / "metrics.csv"), crucible_summary["early_stop_loss"],
                       out_dir / "crucible_loss.png")
    plot_bptt_gradients(json.loads((crucible_dir / "bptt_verification.json").read_text(encoding="utf-8")),
                        out_dir / "gradient_norms_bptt.png")
    print(f"✓ Presentation assets generated in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
