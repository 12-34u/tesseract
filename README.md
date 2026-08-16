# Tesseract

**Tesseract — Transformer Engineered for Sequential State Evaluation using Recursive ACTion**

Tesseract is a research-oriented deep learning project exploring parameter-efficient recursive reasoning in transformer architectures. By reusing a single set of shared transformer block weights over configurable recursion steps ($K$) using dual latent states ($z_H$ and $z_L$) and Backpropagation Through Time (BPTT), Tesseract aims to evaluate whether dynamic computational depth can boost reasoning capability without increasing the trainable parameter count.

---

## Research Question

> **Can shared-weight recursive transformer computation increase reasoning capability by increasing computation K while keeping parameter count approximately fixed?**

---

## Current Status

* **Phase**: **Phase: Prototype Initialization / Day 1**
* **Milestone**: The current milestone is the ~211K parameter prototype (`prototype_small.yaml`). The eventual scaling target is approximately 7M parameters (`full_7m.yaml`).

---

## Scope & Components

### Implemented (Day 1)
- Standard encoder-style Multi-Head Attention module (`models/attention.py`)
- Reproducibility utilities (`utils/seed.py`)
- Parameter counting utility (`utils/param_count.py`)
- Comprehensive test suite for attention module (`tests/test_attention.py`)
- Configuration schemas (`configs/`)

### Future Scope (Planned)
- Shared Transformer Block (`models/block.py`)
- Dual Latent States ($z_H$ and $z_L$) & Embeddings (`models/embeddings.py`)
- Recursive Core engine with BPTT (`models/recursive_core.py`)
- Full Tesseract model wrapper (`models/tesseract.py`)
- Synthetic Reasoning Datasets (`data/`)
- Crucible Overfitting & $K$-scaling Experiments (`experiments/`)
- Evaluation metrics (`evaluation/`)
- FastAPI interactive demo (`app/`)

---

## Repository Structure

```text
tesseract/
├── configs/          # YAML configurations for prototype, crucible, and 7M scaling
├── models/           # PyTorch model definitions (attention, block, recursive core, tesseract)
├── data/             # Synthetic data generators (toy copy task, cellular automaton)
├── training/         # Trainer loop and experiment logger
├── experiments/      # Experiment scripts (crucible overfitting, K-scaling)
├── evaluation/       # Evaluation metrics and benchmark utilities
├── utils/            # Seed utility, parameter counter, and common helpers
├── tests/            # Pytest test suite for model modules
├── notebooks/        # Jupyter notebooks for mathematical sanity checks
├── scripts/          # Auxiliary runner scripts
├── app/              # Future FastAPI web demonstration interface
├── requirements.txt  # Project dependencies
├── ROADMAP.md        # Authoritative implementation roadmap
└── README.md         # Project documentation
```

---

## Setup

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Testing

Run the test suite with verbose output:

```bash
pytest -v
```

---

## Research Disclaimer

Tesseract is a student research and implementation project inspired by modern recursive reasoning architectures (such as Recurrent Transformers, Universal Transformers, and latent state reasoning models). It is conducted as a final-year major project in Computer Engineering and does not claim the underlying recursive architecture concepts as novel academic discoveries.
