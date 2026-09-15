# Tesseract Phase 1 — Pre-Change Audit Report

This report records the state of the repository **before** any audit changes
(HEAD `7b141e9`). Every finding below was checked against the code, and where
noted, by running it. What was fixed, and the results after the fixes, are in
`PHASE1_FINAL_AUDIT.md`.

Baseline environment: Python 3.12.3, torch 2.13.0+cpu (12 intra-op threads),
16 logical CPUs. An NVIDIA GPU is present, but the installed torch build is
CPU-only, so every run was on CPU.

Baseline checks run before changing any code:

* `pytest`: **159 passed** in 4.97 s.
* The unmodified `experiments/*.py` were re-run into a scratch directory
  (the tracked `runs/` was not touched):
  * Crucible reproduced the committed numbers exactly (initial loss 8.385885,
    final loss 0.00997074, early stop at step 32, 100 % EM).
  * K-scaling training columns reproduced exactly. Latency differs, as expected.
  * CA: see §C.

---

## A. Architecture

### Model structure (as implemented)

```text
tokens [B, N]  (int64)
  → TokenPositionalEmbedding: token Embedding(V, D) + learned position Embedding(max_seq_len, D)
  → x_emb [B, N, D]
  → RecursiveReasoner (ONE TransformerBlock instance, K iterations)
        z_H^(0) = learnable_init_H  ([D], broadcast to [B, N, D])
        z_L^(0) = learnable_init_L  ([D], broadcast)
        for k = 1..K:
            m^(k)   = z_L^(k-1) + z_H^(k-1) + x_emb
            z_L^(k) = Block_θ(m^(k))
            z_H^(k) = α·z_H^(k-1) + (1-α)·z_L^(k)
  → z_L^(K) [B, N, D]
  → output_head Linear(D, V) → logits [B, N, V]
  → CrossEntropyLoss (mean over B·N tokens)
```

TransformerBlock (pre-LN): `y = x + Drop(MHA(LN1(x)))`, `z = y + Drop(FFN(LN2(y)))`,
FFN = `Linear(D, d_ff) → GELU → Dropout → Linear(d_ff, D) → Dropout`.
MHA is a custom bidirectional self-attention: a fused QKV projection, einops
head split, `softmax(QKᵀ/√d_head)`, attention-weight dropout, and an output
projection. There is no mask. None is needed: both tasks are
sequence-to-sequence and every position is allowed to see the whole input.

There is **no final LayerNorm** before the output head (see C-2).

### Parameter count (derived analytically; matches `count_parameters`)

| Component | Formula | V = 16 | V = 2 |
|---|---|---:|---:|
| Token embedding | V·D | 2,048 | 256 |
| Position embedding | max_seq_len·D | 8,192 | 8,192 |
| Block: LN1 + LN2 | 4D | 512 | 512 |
| Block: MHA | 3D²+3D + D²+D | 66,048 | 66,048 |
| Block: FFN | D·d_ff+d_ff + d_ff·D+D | 131,712 | 131,712 |
| Learnable z_H/z_L init | 2D | 256 | 256 |
| Output head | D·V+V | 2,064 | 258 |
| **Total** | | **210,832** | **207,234** |

(D = 128, H = 4, d_ff = 512, max_seq_len = 64.) The count does not depend on K.

### Weight sharing and BPTT

* `RecursiveReasoner.__init__` creates exactly one `TransformerBlock`; the loop
  calls `self.shared_block` K times. Verified by forward hooks (existing tests)
  and by the parameter count being identical for K ∈ {1, 2, 4, 8}.
* There is no `.detach()`, `torch.no_grad()` or `.data` inside the recursion.
  `state_history` stores references (no copies).
* `z_H^(K)` is computed but never used downstream, so its gradient is `None`.
  This is mathematically expected.

### Documentation vs. code discrepancies

* **README architecture block does not match the code.** The README says
  `z_H^(0) = z_L^(0) = x_emb`, `z_H^(k) = α z_H^(k-1) + (1-α) z_L^(k-1)` and
  `z_L^(k) = Block(z_L^(k-1) + z_H^(k))`. The code (above) uses learnable init
  vectors, re-injects `x_emb` every step, and updates z_H *after* z_L.
  `VIVA.md` matches the code. The code produced all results, so the README is
  what is wrong.
* The ~7M configuration is described three different ways. `ROADMAP.md` and
  `configs/full_7m.yaml` give d = 512 / H = 8 / d_ff = 2048 (≈3.2M for one
  block). `POST_AUG21.md` and `VIVA.md` give d = 768 / H = 12 / d_ff = 3072.
  Out of scope for Phase 1; noted only.

---

## B. Code quality

### Duplication
* **Accuracy metrics implemented 4×:** `training/trainer.py`, inline in
  `experiments/k_scaling.py` (Part B), and twice inline in
  `experiments/cellular_automaton.py` (train and val). Meanwhile
  `evaluation/metrics.py` is an empty placeholder.
* **Parameter counting 3×:** `utils/param_count.py`,
  `k_scaling.measure_parameter_count`, and inline in `cellular_automaton.py`.
* **Training loop 3×:** `trainer.train_step` (Crucible), an inline loop in
  K-scaling Part B, and an inline loop in CA.
* **Model hyper-parameters duplicated in 7+ places:** the three experiments,
  `validate_prototype.py`, `viva_demo.py`, `demo_day3.py`, and the tests.
* `save_config()` is re-implemented by hand in each experiment from
  module-level constants.

### Dead code
* `configs/prototype_small.yaml` and `configs/crucible.yaml` are **never loaded**
  by any code.
* `experiments/crucible.py`: an unused `DataLoader(shuffle=True)`, and unused
  imports (`os`, `compute_token_accuracy`, `compute_exact_match_accuracy`).
* `training/trainer.py`: unused imports (`Optional`, `nn`, `DataLoader`, `ExperimentLogger`).
* `training/logger.py`: the W&B path is never enabled (`use_wandb=False`
  everywhere) and is wrapped in `except Exception: pass`. `save_csv` computes
  `fieldnames` twice.
* `experiments/cellular_automaton.py`: `time` and `mcolors` are unused.
  `save_predictions` builds five models it never uses.
* Other unused imports: `models/tesseract.py` (`F`),
  `scripts/validate_prototype.py` and `scripts/viva_demo.py` (`nn`),
  `scripts/generate_presentation_assets.py` (`np`), `data/toy_cellular_automaton.py`
  (`Literal`), `experiments/k_scaling.py` (`os`, `count_parameters`).
* `scripts/demo_day3.py` duplicates `viva_demo.py`. It also cannot run as
  documented (no `sys.path` setup).
* Vite template leftovers that nothing references: `src/App.css`,
  `src/assets/{hero.png,react.svg,vite.svg}`, `public/icons.svg`.

### Questionable abstractions / complexity
* `ExperimentLogger` plot titles are hardcoded to "Tesseract Crucible".
* Experiments are 450–700-line scripts built on module-level constants. They
  cannot be called with a different configuration or output directory, so they
  cannot be smoke-tested.

---

## C. Scientific correctness

### C-1 (HIGH) Cellular automaton: T ∈ {1, 2, 4, 8} does not vary the required computation the way the experiment claims
Rule 90 is linear over GF(2): `s' = L(s) ⊕ R(s)`. For T = 2^m, the binomial
coefficients mod 2 vanish except at the ends, so

```text
target_i = x_{i-T} ⊕ x_{i+T}      (periodic)
```

**Verified numerically** on the actual datasets. For T = 1, 2, 4, 8 the
dataset target equals `roll(x, T) ^ roll(x, -T)` exactly. For T = 3, 5, 6, 7 it
does not. So every T in the grid is a two-cell XOR at a different offset, not
a problem whose sequential depth grows with T. This alone can explain the
absence of any T-vs-K trend. The comment
`SEQ_LEN = 32 # length-16 collapses to all-zeros at T=8` is a symptom of the
same identity (i−8 ≡ i+8 mod 16).
*This is a limitation of the experimental design, not a code bug.* The T grid
is part of the recorded experiment, so it will be documented, not changed.

### C-2 (MEDIUM) Residual-stream growth across K confounds K comparisons
Each iteration adds two residual branches onto `z_L + z_H + x_emb`, and there
is no final LayerNorm. Activation norms grow with K. In the baseline Crucible
BPTT printout, `‖z_L‖` goes 258 → 541 → 877 → 1292 over four steps. As a
result, the initial loss grows with K (K-scaling: 3.24, 4.33, 8.39, 28.31 for
K = 1, 2, 4, 8, against ln 16 = 2.77 for uniform predictions). "Steps to
converge" and loss comparisons across K are therefore confounded by the
initial logit scale. *This is architectural; it must not be changed in Phase 1.*

### C-3 (HIGH) Crucible gradient check can never fail for z_H
`experiments/crucible.py:390`: when an intermediate z_H gradient is zero, the
code sets `all_zH_nonzero_except_final = True`, not `False`. The printed
"z_H grads (1..K-1): True" and the PASS verdict therefore never examine z_H.

### C-4 (HIGH) Failures are invisible to the reproducibility runner
* Crucible `return`s (exit 0) when the optimizer does not update any parameter.
  It also exits 0 after printing `CRUCIBLE: ✗ FAIL`.
* `scripts/run_all_experiments.py` prints `Crucible: PASS` and
  `K-Scaling: PASS` **unconditionally**. It treats a failing CA run as a
  warning and still exits 0.
* `run_all_experiments.py --device` is parsed and printed but **never passed**
  to the experiments. The experiments pick CUDA on their own if it is available.
* Scientific invariants are checked with `assert`, which `python -O` strips.

### C-5 (HIGH) "recursive_calls" in K-scaling is not measured
`results.csv` column `recursive_calls` is filled with the configured `k`, not an
observed count. The unit tests do count calls with hooks, but the experiment
artifact only restates its configuration.

### C-6 (HIGH) Hand-typed metrics shown as results
* `scripts/generate_presentation_assets.py` plots **typed-in** values: params
  `[210832]*4`, latency `[1.32, 2.49, 5.03, 9.54]` ms, gradient norms
  `[0.001346, 0.001066, 0.000900, 0.000820]`.
* `app/backend/services/results_loader.py` hardcodes the parameter count
  (210832 / 207234), architecture, device "CPU", every status as "PASS" (based
  only on whether a file exists), `prototype_validation: "PASS"`, "key findings"
  prose with numbers, Crucible dataset/K/params, fallback losses
  8.385885 / 0.009971, `early_stop: True`, and fallback BPTT gradients. It also
  reads only `grad_zL_step_1..4`.
* The BPTT gradient chart takes the gradients from the **last row** of
  `metrics.csv`. That row is only instrumented every 10 steps. Crucible stops
  at step 32, so the dashboard always falls back to the typed-in array.
  (The typed values do match what the baseline Crucible prints, but no file
  holds them.)
* The frontend repeats these fallbacks and adds more: `'210,832'`,
  `"1 / 2 / 4 / 8"`, a Crucible card fixed at `"100.0%"` / `PASS`, CA `N = 32`,
  `256 / 64`, `207,234`, a "Val 0.0%" badge, "16 / 16 full sequences", and a
  KeyFindings list labelled "Programmatically Derived" that is static text.
* **Missing CA cells are rendered as 0 %** (`getCellValue` returns `0`, and
  `|| 0` is used in the bar chart). The backend also defaults a missing
  `best_val_exact_match` to `0.0`.
* Latency is reported three different ways, and none of them matches the
  committed `results.csv` (0.81 → 5.82 ms): VIVA.md says 1.32 → 9.54 ms, and
  the backend key findings say 0.87 → 10.36 ms.
* `viva_demo.py` prints "K (Recursive iterations executed): 8" and "1 block × 8
  executions" for a model built with K = 4.
* `k_scaling.save_summary` writes "Answer: YES" and "1, 2, 4, or 8 times" as
  literals.

### C-7 (MEDIUM) Stale / misleading artifacts
* `runs/` is in `.gitignore`, but `runs/k_scaling` and `runs/cellular_automaton`
  were force-added. `runs/crucible_k04/` holds only an untracked `model.pt`.
  **There is no `metrics.csv`**, so the dashboard's Crucible section is
  unavailable on a fresh clone.
* Result files carry no run ID, timestamp, git commit, library versions or
  thread count. A new run overwrites some files and leaves others from an
  older run in place.
* `runs/cellular_automaton/predictions/examples_T*.txt` (written by
  `save_predictions`) contain **no model predictions**, only simulator
  trajectories. The (4,1), (4,4) and (4,8) entries overwrite the same file.
* Crucible's final BPTT verification is printed but not saved.
* All `RUN_DIR`s are relative to the **current working directory**
  (`Path("runs/…")`), so running from another directory writes somewhere else.

### C-8 (INFO) Metric semantics that must be documented, not silently changed
* Crucible and K-scaling "final loss / token accuracy / exact match" come from
  the forward pass of the last step, **before** that step's optimizer update,
  in `train()` mode. With `dropout = 0.0`, train and eval mode are numerically
  identical.
* CA "final_loss" is the last **mini-batch** loss (64 examples), and early
  stopping uses that single mini-batch. Train/val accuracies are computed in
  `eval()` mode on the full sets at steps 1–5, every 200 steps, at step 2000,
  and at the early-stop step. `best_val_exact_match` is the maximum over those
  sparse points only, not over all steps.
* K values early-stop at different steps. Across K, CA results are compared at
  different amounts of training.

### C-9 Validation exact match = 0 % (CA) — pre-change investigation
Checked so far:
* **Simulator is correct.** An independent implementation
  (`y ← roll(y,1) ^ roll(y,-1)`, applied T times) reproduces every target.
* **No leakage or overlap.** Train seed 42 and val seed 1042 give 256 and 64
  distinct sequences with 0 in common. Targets are balanced (mean ≈ 0.50), so
  chance token accuracy is 50 %.
* **Encoding is consistent.** Both splits use the same generator and dtype, and
  inputs and targets are both {0,1}.
* **Exact-match implementation** `(preds == targets).all(-1)` is correct, and
  so is token accuracy `(preds == targets).mean()`.
* **model.eval() / dropout:** eval is used, and dropout = 0.
* **Consistency with token accuracy:** val token accuracy is 68–84 %. If each of
  the 32 tokens were independently correct with p = 0.84, the expected number of
  fully correct sequences out of 64 is 64·0.84³² ≈ 0.24. With p = 0.75 it is
  0.006. **0/64 is the expected outcome of the observed token accuracy**, not an
  artifact of a broken metric.

Still to do (see the final report): re-train one cell with the refactored
code and evaluate it with an independent metric implementation, check a
train-seed "validation" reproduces train accuracy, and inspect the
per-sequence error distribution.

### C-10 (LOW) Double dropout on the FFN residual branch
`FFN` ends with `nn.Dropout`, and `TransformerBlock.forward` applies
`self.dropout` to its output again. With `dropout = 0.0` (every Phase 1 run)
this has no effect on results.

### C-11 (LOW) Reproducibility metadata
`set_seed` covers Python, NumPy, torch CPU and CUDA, and cuDNN flags. The CA
mini-batch indices come from the global torch RNG. That is reproducible because
`set_seed` is called before every (T, K) cell, but it does mean all 16 cells
see the identical index sequence (a sensible control). Device, thread count,
versions and git commit are not recorded.

---

## D. Performance

| # | Location | Issue |
|---|---|---|
| P-1 | `trainer.train_step` | Total gradient norm calls `.item()` once per parameter tensor (≈24 host syncs/step on GPU), and it reads `p.grad.data`. |
| P-2 | `crucible.py` loop | `dataset.inputs.to(DEVICE)` on every step: a repeated host→device copy on GPU. |
| P-3 | CA / Crucible / K-scaling | `torch.isfinite(torch.tensor(loss_val))` allocates a tensor every step to check a Python float. |
| P-4 | CA `train_model` | Mini-batch indices are created on CPU and used to index device tensors (an implicit transfer per step on GPU). |
| P-5 | CA `save_predictions` | Builds 5 full models that are never used. |
| P-6 | Backend | Each dashboard load re-parses every CSV/YAML ~3× (`summary`, `raw-results` and each section re-read the same files). |
| P-7 | Attention | A manual softmax attention materialises `[B,H,N,N]`. `F.scaled_dot_product_attention` would be faster, but the brief forbids replacing the custom implementation and N ≤ 64 anyway. **Not changed.** |

No retained-graph leaks were found. `state_history` is only built when
instrumentation is requested; latency and eval run under `torch.no_grad()`.
Keeping activations for all K steps during training is inherent to BPTT and
must stay.

---

## E. Reliability

* No validation of `num_recursive_steps` (K = 0 silently returns the init
  vector; negative K runs 0 iterations), of `alpha ∈ [0,1]`, or of the
  `logits`/`targets` shape match in `compute_loss`.
* `ExperimentLogger.save_csv` returns a path to a file that does not exist when
  history is empty. `plot_gradient_norms` finds gradient keys only in the
  **last** record, so for Crucible (last step 32, not instrumented) it draws an
  empty "No gradient data" plot. **Confirmed** in the baseline re-run.
* Broad `except Exception` in `logger.py` (W&B, `pass`), `results_loader.py`
  (turns parse errors into "unavailable") and `validate_prototype.py` (prints
  the message, drops the traceback).
* If CA hits NaN, the loop `break`s without re-evaluating, so the stale
  validation numbers from the last eval point are reported.
* Frontend `api.js` swallows network errors, so the dashboard's "Backend Error"
  banner can never appear.
* Backend CORS uses `allow_origins=["*"]` together with
  `allow_credentials=True`.
* `requirements.txt` is unpinned and includes the unused `wandb`.

---

## F. Hardcoding inventory

### F-1 Acceptable architectural / definitional constants (keep)
* `nn.init.normal_(std=0.02)` for the learnable z_H/z_L init.
* The `3 * d_model` fused QKV projection, the `√d_head` scale, GELU, and the
  LayerNorm default eps.
* `vocab_size = 2` inside `CellularAutomatonDataset` (binary by definition),
  and rule range 0–255.
* Periodic boundary in the CA simulator.
* Plot styling (dpi, colours, figure sizes).

### F-2 Experimental configuration that must be configurable (currently Python constants)

| Value | Where |
|---|---|
| seed = 42 | crucible, k_scaling, cellular_automaton, validate_prototype, viva_demo, demo_day3 |
| d_model = 128, num_heads = 4, d_ff = 512, max_seq_len = 64, alpha = 0.9, dropout = 0.0 | all 3 experiments + 3 scripts + tests (and the unused `prototype_small.yaml`) |
| vocab_size = 16 / 2 | crucible, k_scaling / CA |
| K = 4 | crucible; `K_VALUES = [1,2,4,8]` in k_scaling and CA |
| T_VALUES = [1,2,4,8], rule = 90, seq_len = 32, train = 256, val = 64, val seed = seed + 1000 | CA |
| num_examples = 16, seq_len = 16, task = copy | crucible, k_scaling |
| lr = 3e-4, max_steps = 500, early_stop = 0.01, batch = 16 | crucible |
| lr = 3e-4, max_steps = 200, early_stop = 0.01, batch = 16 (unused) | k_scaling Part B |
| warmup = 10, timing_runs = 30, benchmark batch = 16, global warmup = 5 iters with a K=4 model | k_scaling Part A |
| lr = 1e-3, max_steps = 2000, early_stop = 0.005, batch = 64, eval every 200, eval at steps ≤ 5 | CA |
| PASS thresholds: final loss < 0.1, EM ≥ 90 % | crucible |
| grad-instrument every 10, log every 10/50/200 | experiments |
| RUN_DIR `runs/crucible_k04`, `runs/k_scaling`, `runs/cellular_automaton` | experiments (cwd-relative) |
| `--device` default `cpu` (ignored) | run_all_experiments |

### F-3 Accidental hardcoding to remove
* Hand-typed results in `generate_presentation_assets.py`, `results_loader.py`
  and the frontend components (see C-6).
* `viva_demo.py` "8 executions" for a K = 4 model.
* `k_scaling` summary literals ("Answer: YES", "1, 2, 4, or 8"), and the
  "K=8 / K=1" plot annotation that assumes the last K is 8.
* `range(1, 5)` (K = 4) in the backend gradient parsing.
* `tests/test_block.py` compares against the literal `198272` instead of the
  analytic formula.
* The KScaling chart Y-axis domain is fixed at `[150000, 250000]`, which would
  clip any other model size.

### Paths, devices, secrets
* **No machine-specific absolute paths** found (`/home`, `/Users`, `C:\`).
* The device is chosen as `"cuda" if available else "cpu"` separately in each
  experiment. `--device` is not honoured and not configurable.
* No secrets, tokens or `.env` files are committed. Nothing loads `.env`. The
  only environment variable in use is `VITE_API_BASE_URL` (frontend).

### Notebook
`notebooks/00_math_sanity_checks.ipynb` imports the real project modules (no
copied model code) and has no stored outputs. It is fine as is.
