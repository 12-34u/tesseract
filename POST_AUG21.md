# Tesseract — Post-August 21 Research Agenda & Roadmap

This document outlines planned research, scaling, and engineering tasks to be undertaken after the August 21 presentation milestone freeze.

---

## 1. Immediate Research & Architectural Refinements

* **Latent State Update Mechanism:**
  * Explore learnable gating/blend parameters $\alpha(z_H, z_L)$ rather than fixed scalar EMA decay ($\alpha = 0.9$).
  * Investigate hierarchical state refinement where $z_H$ updates at lower temporal frequencies than $z_L$.
* **Iterative Reasoning Benchmark:**
  * Resolve out-of-distribution generalization bottlenecks on 1D Cellular Automata (Rule 90/30).
  * Introduce spatial positional embeddings or relative attention to improve multi-step sequence transformations.
* **Matched Baseline Model:**
  * Implement an unrolled $K$-layer Transformer baseline with independent parameter weights per layer.
  * Perform controlled compute-matched and parameter-matched performance comparisons against Tesseract.

---

## 2. Model Scaling Agenda

* **Intermediate Scale Steps:**
  * **~1M parameters:** $d_{model} = 256, H = 8, d_{ff} = 1024$
  * **~2M parameters:** $d_{model} = 384, H = 8, d_{ff} = 1536$
  * **~4M parameters:** $d_{model} = 512, H = 8, d_{ff} = 2048$
* **Target Scale (~6.3M–7.0M):**
  * Instantiate full scaling configuration (`configs/full_7m.yaml`).
  * $d_{model} = 768, H = 12, d_{ff} = 3072$, shared blocks = 1 or 2, $K \in [4, 16]$.

---

## 3. Evaluation & Benchmarking Suite

* **Structured Reasoning Tasks:**
  * Evaluate grid-based spatial transformations, graph reachability, and symbolic logic puzzles.
* **ARC-AGI Benchmark Pipeline:**
  * Construct ARC task pre-processing, grid tokenization, and evaluation metrics.
* **Extrapolation Evaluation ($K_{test} > K_{train}$):**
  * Test model performance when evaluated with recursive depth $K_{test} = 8$ or $16$ after training at $K_{train} = 4$.
* **Ablation Studies:**
  * Compare Dual State ($z_H, z_L$) vs Single State ($z_L$ only) vs Fixed State ($z_H$ only).

---

## 4. Systems & Software Engineering

* **Weights & Biases Integration:** Clean up experiment logging, hyperparameter sweeps, and artifact versioning.
* **Checkpointing & Model Hub:** Serialization, state dict loading, and pre-trained weights release.
* **FastAPI Interactive Web Service:** Deploy interactive UI/API (`app/`) allowing real-time visualization of recursive state evolution across steps $1 \dots K$.
* **Inference Optimization:** Export models via TorchScript / ONNX and measure latency on GPU/TPU targets.
