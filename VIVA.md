# Tesseract — Viva & Oral Examination Cheat Sheet

This document contains concise, defensible answers for the August 21 major-project presentation and viva examination. All responses are backed by actual experimental evidence and prototype measurements.

---

### 1. What is Tesseract?
**Tesseract** is a research prototype implementing shared-weight recursive transformer computation with dual latent states ($z_H, z_L$) and full Backpropagation Through Time (BPTT).

---

### 2. What problem does it address?
Standard transformers scale computation strictly by adding parameters (adding more layers). Tesseract addresses parameter efficiency by scaling computational depth dynamically through recursive weight reuse rather than growing model size.

---

### 3. Why recursive computation?
Recursive computation allows a neural network to apply the same processing module multiple times to refine intermediate latent representations, enabling variable compute per token without increasing parameter count.

---

### 4. What does K represent?
$K$ represents the **recursive depth**—the number of times the shared transformer block is executed sequentially during a single forward pass.

---

### 5. Why does parameter count remain constant?
Because the model contains only **one** instance of the `TransformerBlock`. Running $K=1, 2, 4,$ or $8$ steps reuses the exact same parameter tensors.

---

### 6. What is weight sharing?
Weight sharing is an architectural constraint where the same module parameters ($\theta$) process inputs at multiple steps of a computational sequence: $z^{(k)} = f_\theta(z^{(k-1)})$.

---

### 7. What are z_H and z_L?
* **$z_L$ (Low-level state):** The dynamic state updated directly by the transformer block at every step.
* **$z_H$ (High-level state):** An EMA-accumulated slower state ($z_H^{(k)} = \alpha z_H^{(k-1)} + (1-\alpha) z_L^{(k)}$) that retains context across recursive steps.

---

### 8. Why are we using a simplified dual-state mechanism?
To establish a stable, mathematically tractable baseline prototype (~211K params) for gradient stability and BPTT verification before introducing complex hierarchical updates.

---

### 9. What is BPTT?
**Backpropagation Through Time (BPTT)** unwinds the recursive computation graph across all $K$ steps, allowing loss gradients to flow backward through the entire sequence of recursive state updates.

---

### 10. Why is BPTT required?
Without BPTT (or if state gradients are detached), the model cannot learn how past recursive state transformations contribute to the final output loss.

---

### 11. Why does memory increase with K?
Because during training, PyTorch retains activation tensors for all $K$ intermediate steps in memory to calculate gradients during the backward pass.

---

### 12. Why is the model only ~211K right now?
The ~211K prototype (210,832 trainable parameters) is a **deliberate initial validation stage** built to verify weight sharing, gradient propagation, and $K$-scaling mechanics reliably.

---

### 13. Why not directly build 7M?
Building a 7M model without first validating BPTT and numerical stability on a lightweight prototype risks unresolvable training instability and wasteful compute.

---

### 14. What did the Crucible prove?
The Crucible experiment proved that Tesseract's recursive architecture can successfully propagate gradients via BPTT and converge on a target task (loss < 0.01, 100% exact match in 32 steps).

---

### 15. What did the K-scaling experiment prove?
It proved parameter invariance: the trainable parameter count remains exactly **210,832** across $K=1, 2, 4, 8$, while execution latency scales predictably from 1.32 ms to 9.54 ms.

---

### 16. Does K=8 mean the model is better?
Not automatically. Higher $K$ increases computational depth, but whether performance improves depends on whether the task requires multi-step iterative processing.

---

### 17. Have we demonstrated improved reasoning?
No. On simple tasks (like sequence copy), all $K$ values converge. On complex tasks (like Cellular Automata), generalization requires further architectural refinement planned for post-Aug-21.

---

### 18. Why use a cellular automaton?
1D Cellular Automata (like Rule 90) require $T$ sequential local steps to compute step $T$, making them an ideal synthetic testbed for iterative computation depth.

---

### 19. Why ARC later?
ARC-AGI is a complex benchmark requiring large capacity and spatial reasoning. Validating the core recursive engine on small synthetic tasks must precede ARC evaluation.

---

### 20. What is the eventual ~7M model?
The scaled version (`configs/full_7m.yaml`) with $d_{model}=768, H=12, d_{ff}=3072$, designed to test whether parameter-efficient recursion holds at higher model scales.

---

### 21. What is the baseline?
The planned baseline is an unrolled $K$-layer Transformer where each layer has independent (non-shared) weight parameters.

---

### 22. What is novel about our work?
We do **not** claim to have invented recursive transformers or dual latent states.

Our contribution is:
> **An implementation and experimental investigation of shared-weight recursive transformer computation, with a dual-state prototype and systematic evaluation of how computation depth K affects behavior under approximately fixed parameter count.**
