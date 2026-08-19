# Tesseract

**Tesseract — Transformer Engineered for Sequential State Evaluation and Recursive Action**

Tesseract is a research-oriented deep learning project exploring parameter-efficient recursive reasoning in transformer architectures. By reusing a single set of shared transformer block weights over configurable recursion steps ($K$) using dual latent states ($z_H$ and $z_L$) and Backpropagation Through Time (BPTT), Tesseract demonstrates that dynamic computational depth can be varied without introducing additional trainable parameters.

---

## Research Question

> **Can shared-weight recursive transformer computation increase computation depth $K$ while keeping parameter count approximately fixed?**

---

## Current Prototype Architecture

```text
Input Tokens [B, N]
↓
Token + Positional Embedding [B, N, D]
↓
Dual Latent Initialization (z_H^(0) = x_emb, z_L^(0) = x_emb)
↓
Shared TransformerBlock (K recursive iterations)
  ├─ z_H^(k) = alpha * z_H^(k-1) + (1 - alpha) * z_L^(k-1)
  └─ z_L^(k) = Block(z_L^(k-1) + z_H^(k))
↓
Final Latent Output z_L^(K) [B, N, D]
↓
Output Projection Head [B, N, V]
```

### Prototype Configuration (`configs/prototype_small.yaml`)

* **Trainable Parameters:** **210,832**
* **Embedding Dimension ($d_{model}$):** 128
* **Attention Heads:** 4
* **Feed-Forward Hidden Dim ($d_{ff}$):** 512
* **Default Recursive Depth ($K$):** 4 (configurable to 1, 2, 4, 8)
* **EMA Blend Alpha ($\alpha$):** 0.9

---

## Research Roadmap & Phased Execution

The Tesseract major-project research follows a structured 5-phase progression:

* **Phase 1: Architecture Implementation** — Multi-head attention, Pre-LN Transformer block, shared recursive core, dual latent states.
* **Phase 2: Small-Scale Validation** — ~211K prototype build and unit testing suite.
* **Phase 3: Recursive Computation Validation** — Crucible overfitting benchmark and $K$-scaling structural parameter invariance verification.
* **Phase 4: Scaling** — Expansion toward intermediate (~1M–4M) and final (~7M) parameter models *(Post-August 21)*.
* **Phase 5: Reasoning Evaluation** — ARC-AGI task evaluation, matched baseline comparisons, and ablations *(Post-August 21)*.

> *Note: The ~211K prototype is a deliberate validation stage to ensure mathematical correctness, parameter invariance, and gradient stability before scaling.*

---

## Research Status

### Completed

- Custom multi-head attention (`models/attention.py`)
- Pre-LN TransformerBlock (`models/block.py`)
- Shared-weight recursive core (`models/recursive_core.py`)
- Dual latent states ($z_H, z_L$) with EMA update rule
- Full Backpropagation Through Time (BPTT) without state truncation
- ~211K prototype implementation (210,832 trainable parameters)
- Crucible overfitting experiment (Loss < 0.01, 100% exact match)
- K-scaling parameter invariance experiment ($K \in \{1, 2, 4, 8\}$)
- Automated prototype validation suite (`scripts/validate_prototype.py`)
- Single reproducibility entry point (`scripts/run_all_experiments.py`)

### In Progress

- Iterative reasoning benchmark refinement
- Hierarchical latent-state update exploration

### Future (Post-August 21)

- ~7M parameter model scaling
- ARC-style reasoning evaluation
- Matched unrolled baseline comparison
- Architectural ablations
- Adaptive computation depth (dynamic halting)
- FastAPI interactive demonstration service

---

## Quick Start & Verification

### Running Tests

```bash
pytest -v
```

### Running Automated Prototype Validation

```bash
python scripts/validate_prototype.py
```

### Reproducing Main Experiments

```bash
python scripts/run_all_experiments.py --skip-ca
```

### Running Viva Demonstration

```bash
python scripts/viva_demo.py
```

---

## Repository Structure

```text
tesseract/
├── configs/          # YAML configurations for prototype, crucible, and 7M scaling
├── models/           # PyTorch model definitions (attention, block, recursive core, tesseract)
├── data/             # Synthetic data generators (toy copy task, cellular automaton)
├── training/         # Trainer loop and experiment logger
├── experiments/      # Experiment scripts (crucible overfitting, K-scaling, cellular automaton)
├── evaluation/       # Evaluation metrics and benchmark utilities
├── utils/            # Seed utility, parameter counter, and common helpers
├── tests/            # Pytest test suite for model modules
├── notebooks/        # Jupyter notebooks for mathematical sanity checks
├── scripts/          # Automated validation, reproducibility runner, and presentation tools
├── app/              # Future FastAPI web demonstration interface
├── runs/             # Saved experiment metrics, CSVs, markdown results, and presentation figures
├── requirements.txt  # Project dependencies
├── ROADMAP.md        # Authoritative implementation roadmap
├── POST_AUG21.md     # Post-presentation research agenda
├── VIVA.md           # Presentation and oral examination cheat sheet
└── README.md         # Main documentation
```

---

## Research Contribution & Novelty Disclaimer

Tesseract is an implementation and experimental investigation of shared-weight recursive transformer computation with a dual-state prototype and systematic evaluation of how dynamic computation depth $K$ affects execution latency and gradient dynamics under fixed trainable parameter counts.
