#!/usr/bin/env python3
"""Single Reproducibility Entry Point for Tesseract Experiments — Day 7.

Runs the Crucible and K-scaling experiments, and optionally Cellular Automaton.
Usage:
    python scripts/run_all_experiments.py [--skip-ca] [--device cpu|cuda]
"""

import argparse
import os
import subprocess
import sys

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def main():
    parser = argparse.ArgumentParser(description="Run all Tesseract prototype experiments.")
    parser.add_argument(
        "--skip-ca",
        action="store_true",
        default=False,
        help="Skip Cellular Automaton experiment (optional / post-Aug-21 work).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for execution (cpu or cuda).",
    )
    args = parser.parse_args()

    python_bin = sys.executable

    print("========================================================")
    print("  TESSERACT REPRODUCIBILITY SUITE — DAY 7 FREEZE")
    print("========================================================")
    print(f"  Python binary:  {python_bin}")
    print(f"  Device:         {args.device}")
    print(f"  Skip CA:        {args.skip_ca}")
    print("========================================================\n")

    # 1. Crucible Experiment
    print(">>> 1/3 Running Crucible Experiment (Copy Overfitting)...")
    crucible_script = os.path.join(repo_root, "experiments", "crucible.py")
    cmd = [python_bin, crucible_script]
    res_crucible = subprocess.run(cmd, cwd=repo_root)

    if res_crucible.returncode != 0:
        print("ERROR: Crucible experiment failed.")
        sys.exit(res_crucible.returncode)

    print("\n✓ Crucible experiment completed successfully.\n")

    # 2. K-Scaling Experiment
    print(">>> 2/3 Running K-Scaling Experiment (Structural & Latency)...")
    k_scaling_script = os.path.join(repo_root, "experiments", "k_scaling.py")
    cmd = [python_bin, k_scaling_script]
    res_k_scaling = subprocess.run(cmd, cwd=repo_root)

    if res_k_scaling.returncode != 0:
        print("ERROR: K-scaling experiment failed.")
        sys.exit(res_k_scaling.returncode)

    print("\n✓ K-scaling experiment completed successfully.\n")

    # 3. Cellular Automaton Experiment (Optional / Skippable)
    if args.skip_ca:
        print(">>> 3/3 Skipping Cellular Automaton experiment (--skip-ca flag provided).")
        print("    CA experiment status: POST-AUG-21 — Iterative reasoning experiment requires further validation.\n")
    else:
        print(">>> 3/3 Running Cellular Automaton Experiment...")
        ca_script = os.path.join(repo_root, "experiments", "cellular_automaton.py")
        cmd = [python_bin, ca_script]
        res_ca = subprocess.run(cmd, cwd=repo_root)
        if res_ca.returncode != 0:
            print("WARNING: Cellular Automaton experiment exited with non-zero code.")

    print("========================================================")
    print("  ALL EXPERIMENTS EXECUTED")
    print("========================================================")
    print("  Crucible:        PASS")
    print("  K-Scaling:       PASS")
    print(f"  CA Experiment:   {'SKIPPED / POST-AUG-21' if args.skip_ca else 'EXECUTED'}")
    print("========================================================")


if __name__ == "__main__":
    main()
