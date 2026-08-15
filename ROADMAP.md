# Tesseract Implementation Roadmap

**Tesseract — Transformer Engineered for Sequential State Evaluation and Recursive Action**

## Core Research Question
> Can shared-weight recursive transformer computation increase reasoning capability by increasing computation $K$ while keeping trainable parameter count approximately fixed?

---

## Technical Overview
Tesseract investigates parameter-efficient recursive reasoning in transformer models. Instead of adding depth via distinct layers with separate parameters, Tesseract applies a single, shared-weight transformer block recursively over $K$ iterations, operating on dual latent states ($z_H$ high-level context and $z_L$ low-level dynamic state) and trained via Backpropagation Through Time (BPTT).

---

## Milestones & Parameter Budget

- **Prototype Phase**: ~211K parameters ($d_{\text{model}}=128$, $H=4$, $d_{\text{ff}}=512$, $K=4$)
- **Scaling Phase**: ~6.3M–7M parameters ($d_{\text{model}}=512$, $H=8$, $d_{\text{ff}}=2048$, $K=4$)

---

## Phase Breakdown & Development Schedule

### Phase 1: Foundation & Attention Engine (Day 1 - CURRENT)
- [x] Project repository initialization & directory layout
- [x] Reproducibility utilities (`utils/seed.py`)
- [x] Parameter counting tools (`utils/param_count.py`)
- [x] Baseline Multi-Head Attention module (`models/attention.py`)
- [x] Comprehensive attention unit tests (`tests/test_attention.py`)

### Phase 2: Shared Transformer Block (Day 2)
- [ ] LayerNorm & Feed-Forward Network (FFN) implementation
- [ ] Residual connections & dropout options
- [ ] Integrated `TransformerBlock` module (`models/block.py`)
- [ ] Unit tests for `TransformerBlock` shape & gradient flow (`tests/test_block.py`)

### Phase 3: Embeddings & Dual Latent State Management (Day 3)
- [ ] Token embeddings & positional encoding (`models/embeddings.py`)
- [ ] Latent state initializers for $z_H$ (High-level context) and $z_L$ (Low-level state)
- [ ] State update mechanics and combination functions

### Phase 4: Recursive Core & Tesseract Model (Day 4)
- [ ] `RecursiveCore` module executing $K$ unrolled steps (`models/recursive_core.py`)
- [ ] Full `Tesseract` sequence model wrapper (`models/tesseract.py`)
- [ ] BPTT gradient verification across varying $K \in \{1, 2, 4, 8\}$
- [ ] Verification that parameter count remains constant across $K$

### Phase 5: Synthetic Data & Tasks (Day 5)
- [ ] Sequence Copy task generator (`data/toy_copy.py`)
- [ ] 1D Cellular Automaton execution data (`data/toy_cellular_automaton.py`)
- [ ] Data loaders & batching pipeline

### Phase 6: The Crucible Experiment (Day 6)
- [ ] Training engine & loss computation (`training/trainer.py`)
- [ ] Experiment logger & WandB tracking (`training/logger.py`)
- [ ] Crucible overfitting trial on 16 examples ($K=4$, 500 steps) (`experiments/crucible.py`)

### Phase 7: Depth ($K$) Scaling & Evaluation (Day 7)
- [ ] $K$-scaling comparison ($K=1$ vs $K=2$ vs $K=4$ vs $K=8$) (`experiments/k_scaling.py`)
- [ ] Evaluation metrics (Exact match accuracy, step efficiency) (`evaluation/metrics.py`)
- [ ] Ablation study reports & visual analysis

### Phase 8: Demonstration & Deployment (Day 8)
- [ ] Model serialization and checkpoint loading
- [ ] Interactive FastAPI demonstration endpoint (`app/main.py`)
- [ ] ARC-style reasoning task visualization interface
