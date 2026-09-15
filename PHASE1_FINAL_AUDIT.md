# Phase 1 Final Audit

Scope: a full audit of the Phase 1 Tesseract repository (pre-change findings in
`AUDIT_REPORT.md`), the resulting fixes, and a re-run of every Phase 1
experiment. Environment: Python 3.12.3, torch 2.13.0+cpu (12 threads), 16 CPUs.
An NVIDIA GPU is present, but the installed torch build is CPU-only, so every
run below is on **CPU**.

## Overall Status

**PASS WITH WARNINGS**

The code does what it claims to do. The three experiments reproduce the
committed Phase 1 results **bit-exactly**: every training/accuracy column of
every CSV, and the final Crucible weights. The metrics were verified by
independent implementations. The warnings are scientific limitations of the
experimental design (§Scientific Issues) plus a few items not verifiable in
this environment (§Remaining Warnings).

## Architecture

The architecture is unchanged by this audit.

```text
tokens [B,N] → token + learned positional embedding x [B,N,D]
→ z_H^(0), z_L^(0): learnable [D] vectors broadcast to [B,N,D]
→ for k = 1..K with ONE TransformerBlock (pre-LN, custom bidirectional MHA):
      z_L^(k) = Block(z_L^(k-1) + z_H^(k-1) + x)
      z_H^(k) = α z_H^(k-1) + (1-α) z_L^(k)
→ Linear(z_L^(K)) → logits [B,N,V]
```

* Prototype (`configs/prototype_small.yaml`): d_model 128, 4 heads, d_ff 512,
  max_seq_len 64, α 0.9, dropout 0.0.
* Measured trainable parameters: **210,832** (V=16) and **207,234** (V=2),
  identical for K ∈ {1,2,4,8}. They also match an analytic formula (test).
* Verified by tests:
  * The model output **equals a manual K-step unroll of the single block**
    under the rule above, bit-exactly. This proves both weight reuse and the
    update rule.
  * The parameter layout is identical across K, and a checkpoint from one K
    loads strictly into any other.
  * Measured block calls = K.
  * `verify_bptt` passes for K=1,2,4,8, and it **fails** on a deliberately
    detached recursion.
* The README described a different update rule
  (`z_H^(0)=z_L^(0)=x_emb`, z_H updated before z_L). **The README was wrong,
  not the code**, and has been corrected.

## Tests

| | Before | After |
|---|---|---|
| Tests | 159 passed | **255 passed, 0 failed** |
| Runtime (idle machine) | 4.97 s | 5.13 s / 5.48 s (two runs) |

New coverage:
* **Metrics:** hand-checked examples, including `[1,0,1,1]` vs `[1,0,0,1]` and one error in 32 tokens.
* **Config:** strict loading (unknown/missing keys, wrong types, the YAML `3e-4` string trap).
* **Device:** device resolution.
* **Recursion and BPTT:** manual-unroll equivalence, BPTT at K=1/2/4/8, detached-recursion detection.
* **Dropout:** applied exactly once per site, and only in train mode.
* **Training:** the train loop (early stop, max steps, non-finite loss), no gradient accumulation, instrumentation does not change the update, determinism.
* **Cellular automaton:** Rule 90 against an independent roll-XOR reference for T=0..8, the power-of-two identity, and train/val disjointness of the shipped config.
* **Logger:** logger behaviour.
* **Experiments:** end-to-end smoke tests of all three experiments, including exit codes, reproducibility and stale-artifact cleanup.
* **Backend loader:** missing → null, 0.0 preserved, malformed CSV → explicit error.

Static checks:
* `python -m compileall` clean.
* AST unused-import scan clean.
* Frontend `oxlint` 0 warnings, and `vite build` succeeds.

### Final test matrix (executed)

|                      | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| Forward              | ✓ | ✓ | ✓ | ✓ |
| Backward             | ✓ | ✓ | ✓ | ✓ |
| BPTT (all z_L^(k) receive gradient) | ✓ | ✓ | ✓ | ✓ |
| Parameter invariance | ✓ | ✓ | ✓ | ✓ |
| Weight sharing (1 block, calls = K, unroll equivalence) | ✓ | ✓ | ✓ | ✓ |
| No NaN/Inf           | ✓ | ✓ | ✓ | ✓ |

`scripts/validate_prototype.py` prints OVERALL: PASS (18/18 checks).

## Experiments

All experiments were run with `python scripts/run_all_experiments.py`
(exit 0, total 1193 s). Artifacts are in `runs/`, each with
`run_metadata.json`.

### Reproducibility against the committed Phase 1 results

| Check | Result |
|---|---|
| Original (unmodified) code re-run vs committed CSVs | Crucible, K-scaling training columns, and all 16 CA rows **identical** |
| Refactored code vs committed CSVs | Every original column **identical** (latency excepted) |
| Crucible final `model.pt`, original vs refactored code | All tensors **bit-identical**. The old checkpoint loads strictly into the new model. |

### Crucible — PASS

* **Convergence:** loss 8.385885 → 0.009971 in **32 steps** (early stop).
  Token accuracy 100.0 % and exact match 100.0 % at the final pre-update
  step. The post-training eval-mode check is also 100 % / 100 %.
* **BPTT on the trained model:** ‖∂L/∂z_L^(k)‖ = 0.001346, 0.001066, 0.000900,
  0.000820 for k = 1..4. ‖∂L/∂z_H^(k)‖ is non-zero for k = 1..3. z_H^(4) has
  no gradient, which is expected.
* **Checks:** all five pass (parameters update, finite loss, loss < 0.1,
  EM ≥ 90 %, BPTT). Exit code 0.

### K-Scaling — PASS

| K | Params | Block calls (measured) | Median forward latency (ms, CPU) | Init loss | Steps to loss < 0.01 |
|---:|---:|---:|---:|---:|---:|
| 1 | 210,832 | 1 | 1.63 | 3.2446 | 84 |
| 2 | 210,832 | 2 | 2.42 | 4.3291 | 55 |
| 4 | 210,832 | 4 | 4.46 | 8.3859 | 32 |
| 8 | 210,832 | 8 | 9.21 | 28.3127 | 29 |

* Latency is wall-clock time for inference (eval mode, `no_grad`), with 10
  warmup and 30 timed passes. It is machine-dependent: the committed Aug-20
  run measured 0.81 → 5.82 ms, and today's baseline re-run 1.17 → 8.93 ms.
  The monotone increase is stable; the absolute values are not.
* FLOPs were not computed.
* All four training rows early-stopped, each at 100 % token and exact-match
  accuracy.

### Cellular Automaton (Rule 90, N=32, 256 train / 64 val) — COMPLETED, genuine generalization failure

Each cell shows train exact match % / validation exact match % / validation
token accuracy % / fewest wrong tokens in any validation sequence (of 32).

| T \ K | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | 40.6 / 0.0 / 68.8 / 6 | 94.1 / 0.0 / 72.6 / 4 | 96.1 / 0.0 / 79.5 / 3 | 98.0 / 0.0 / 76.5 / 3 |
| 2 | 71.5 / 0.0 / 80.4 / 2 | 92.6 / 0.0 / 76.7 / 3 | 98.8 / 0.0 / 82.1 / 2 | 99.2 / 0.0 / 84.0 / 2 |
| 4 | 94.9 / 0.0 / 80.1 / 2 | 95.7 / 0.0 / 77.1 / 3 | 96.1 / 0.0 / 79.2 / 1 | 98.4 / 0.0 / 83.5 / 2 |
| 8 | 50.8 / 0.0 / 70.7 / 4 | 93.4 / 0.0 / 68.2 / 4 | 95.7 / 0.0 / 74.6 / 2 | 97.7 / 0.0 / 80.6 / 2 |

The train and validation exact-match columns are identical to the committed Phase 1 CSV. In summary:

* **Train exact match:** 40.6–99.2 %.
* **Validation exact match:** 0.0 % in all 16 cells.
* **Validation token accuracy:** 68.2–84.0 % (chance 50 %).
* **Closest miss:** the fewest wrong tokens in any validation sequence is 1–6
  depending on the cell. The best is 1 of 32 (T=4, K=4).
* **Stopping:** three cells (T=1/2/8 at K=1) hit `max_steps`; the others early-stopped.
* **K trend:** there is none in validation exact match. Train exact match is
  lowest at K=1.

## Bugs Found

| # | Problem | Root cause | Fix | Impact |
|---|---|---|---|---|
| 1 | Crucible "z_H grads (1..K-1)" check could never fail | `all_zH_nonzero_except_final = True` where `False` was intended | BPTT verification moved to `training.trainer.verify_bptt` with explicit failure list; tested against a detached recursion | The PASS verdict did not actually check z_H; it does now (and passes) |
| 2 | Experiment failures invisible | Crucible returned exit 0 on FAIL and on a failed update check; `run_all_experiments.py` printed "PASS" unconditionally and treated CA failure as a warning; invariants used `assert` (stripped by `-O`) | Real exit codes, verdicts in `summary.json`/`run_metadata.json`; runner reports each subprocess's status and exits 1 on failure; invariants raise | A failing run could have been reported as passing |
| 3 | `run_all_experiments.py --device` ignored | Argument parsed but never forwarded | Forwarded as `--device`; experiments resolve devices centrally (`utils/device.py`); explicit `cuda` without CUDA is an error | Device requested ≠ device used |
| 4 | K-scaling `recursive_calls` column not measured | Filled with configured `k` | Counted with a forward hook on the shared block; mismatch raises | The column claimed a measurement it did not make |
| 5 | Crucible `gradient_norms.png` was an empty "No gradient data" plot | Logger discovered keys from the last record only; step 32 was not instrumented | Keys gathered from all records; plotting with no data raises | Confirmed in the baseline re-run; plot now shows real norms |
| 6 | Dashboard displayed hand-typed or fallback numbers | Backend/frontend hardcoded parameter counts, architecture, device, "PASS" statuses (based on file existence), findings text, BPTT gradients (always used, because the last metrics row lacks gradients), missing CA cells rendered as 0 % | Backend derives every value from artifacts (strict CSV parsing, null for missing, errors surfaced); findings computed from data; frontend shows "—" for missing, verdicts and provenance from `run_metadata.json` | The GUI could show numbers no artifact contained |
| 7 | `generate_presentation_assets.py` plotted typed-in results | Literal arrays (e.g. latency 1.32 → 9.54 ms, not reproduced by any saved run) | Reads `results.csv`, `metrics.csv`, `bptt_verification.json`, `summary.json`; exits 1 if missing | Presentation figures could disagree with data |
| 8 | CA "predictions" files contained no predictions and overwrote each other | `save_predictions` wrote simulator trajectories only; (4,1)/(4,4)/(4,8) shared a filename | `examples/examples_T*.txt` (clearly labelled ground truth) + `predictions/T{t}_K{k}.txt` with real validation predictions and error markers | Misleading artifact |
| 9 | Outputs depended on the working directory | `RUN_DIR = Path("runs/…")` | Paths derived from the repo (`utils/paths.py`), `TESSERACT_RUNS_DIR` override, `--output-dir` | Running from elsewhere scattered results |
| 10 | Stale results could masquerade as current | No run IDs/metadata; partial overwrites | `RunRecorder` deletes the experiment's own declared artifacts at start and writes `run_metadata.json` (running → completed/failed, verdict, seed, device, threads, versions, git commit + dirty flag, argv) | Old files were indistinguishable from new ones |
| 11 | Configs were never loaded; values duplicated in 7+ files | Module-level constants | Typed strict config system (`utils/config.py`), 4 experiment configs, model shape defined once | See §Hardcoding Removed |
| 12 | Double dropout on the FFN residual branch | Trailing `nn.Dropout` in FFN plus `self.dropout` in `forward` | Trailing module removed (state-dict keys unchanged); test asserts 4 dropout applications per block call | **None for Phase 1** (dropout = 0.0) |
| 13 | Missing argument validation | K ≥ 1, α ∈ [0,1], `targets` shape unchecked (K=0 silently returned the init vector) | `ValueError`s + tests | Silent nonsense on bad input |
| 14 | Silent failures in the logger | `save_csv` returned a nonexistent path when empty; W&B path wrapped in `except Exception: pass` (and never enabled) | Raise on empty; unused W&B code and dependency removed | — |
| 15 | CA reported stale validation metrics after a NaN break | No re-evaluation on break | `stopped_reason`/`eval_step` recorded; non-finite loss makes the run FAIL | Did not occur in Phase 1 |
| 16 | `viva_demo.py` claimed "1 block × 8 executions" for a K=4 model | Literal text | Prints measured call count | Presentation error |
| 17 | Backend CORS `*` with credentials; frontend swallowed fetch errors (error banner unreachable) | — | Origins from `TESSERACT_CORS_ORIGINS` (default Vite dev origins), GET only; errors propagated and displayed per endpoint | Security hygiene / invisible backend failures |
| 18 | ∂L/∂z_H^(K) logged as 0.0 | `None` gradient converted to 0.0 | Logged as empty/null | "Absent" is not "zero" |

## Scientific Issues

### Validation exact match = 0 % (CA): genuine generalization failure (investigated, preserved)

Evidence (`scripts/diagnose_ca_generalization.py`,
`runs/ca_generalization_diagnostic/diagnostic.json`):

1. **Not a data problem.** The simulator matches an independent roll-XOR
   implementation for T = 0..8. Train and validation share 0 of 320 initial
   states. Targets are balanced (mean ≈ 0.50).
2. **Not a metric or evaluation bug.** Re-training T=1,K=4 and T=8,K=4
   reproduces the `results.csv` rows exactly. An independent pure-Python
   metric gives identical train and validation numbers (e.g. T=1,K=4: val
   token accuracy 79.49 %, EM 0.0 %). Evaluating a regenerated copy of the
   training set reproduces the train metrics. Evaluation runs in `eval()` mode
   with dropout = 0.
3. **Consistent with the token accuracy.** Validation sequences have 3–11
   (T=1) or 2–16 (T=8) wrong tokens out of 32. Per-position validation
   accuracy ranges from 39 % to 100 %. The independence estimate
   `token_acc^32` predicts 0.06 % and 0.009 % exact match, i.e. ≈ 0 of 64.
4. **The pipeline can register generalization.** Control: the same T=1, K=4
   cell trained on **4,096** examples reaches **95.3 % validation exact
   match** (99.85 % token accuracy). The 0 % at 256 examples is a small-data
   generalization failure, not an artifact.
5. **But not only data.** With 4,096 examples T=8, K=4 does not even fit the
   training set: 85.1 % token accuracy and 0.6 % exact match after 2,000
   steps.

**Conclusion:** option (1), a genuine generalization failure. With 256
examples the model memorises the training set. The recorded result is
unchanged.

### T ∈ {1, 2, 4, 8} does not scale the task's dependency structure (unresolved: design limitation)

Rule 90 is linear over GF(2). For T = 2^m, `target_i = x[i−T] XOR x[i+T]`
exactly. This was verified on the real datasets, is re-checked in every CA
`summary.txt`, and is covered by a test. Every T in the grid is a two-cell XOR
at a different offset, so the grid cannot show that "more required sequential
steps need more K". Control 5 above shows the larger offset is still
empirically harder, so T is not irrelevant, but its meaning is not the one
stated in the experiment. **Not changed** (it is the recorded experiment).

### Initial loss grows with K (unresolved: architectural)

There is no final LayerNorm, and each iteration adds residual branches onto
`z_L + z_H + x`. Activation norms grow (Crucible ‖z_L‖ 258 → 1292 over four
steps), and the initial loss rises 3.24 → 28.31 for K=1 → 8. Across-K
comparisons of loss or steps-to-converge are confounded by this. **Not
changed** (architecture freeze).

### Other confounders and metric semantics (documented; nothing silently changed)

* **Crucible / K-scaling `final_*`:** computed from the last step's forward
  pass *before* its optimizer update. An explicit `post_training_eval` was
  added to the Crucible summary.
* **CA `final_loss`:** the last 64-example mini-batch loss, which is also what
  early stopping uses. Train/val metrics are full-set measurements at
  `eval_step`, now recorded. `best_val_exact_match` is the maximum over sparse
  evaluation points only.
* **Unequal training lengths:** CA cells stop at different steps (787–2000),
  so K values are compared at different amounts of training.
* **Mini-batch order:** all CA cells use the identical mini-batch index stream
  (seeded before model construction). This is a sensible control, now
  documented.
* **Latency claims:** three different latency claims existed (README
  dashboard text, VIVA.md, backend findings). None matched a saved run. All
  are now derived from `results.csv` or removed.

### Reproducibility

* **Deterministic:** yes on CPU, verified twice (baseline vs refactored code),
  plus a determinism smoke test.
* **Recorded:** seed, device, threads, versions, commit, dirty flag and full
  resolved config for every run.
* **GPU:** bit-reproducibility not established (see Remaining Warnings).

## Performance Improvements

| Change | Before | After | Note |
|---|---|---|---|
| Global gradient norm (`train_step`) | 92.1 µs, one `.item()` per parameter tensor (18 host syncs on GPU) | 74.4 µs, one sync | CPU measurement; the GPU sync reduction was not measured |
| Shared training step (adds per-step accuracy + gradient norm to the CA loop) | 51.67 ms/step (original CA loop body) | 51.91 ms/step | +0.5 %, within run-to-run noise (rounds 49.5–53.1 ms) |
| Crucible: dataset `.to(device)` every step | per step | once | No-op on CPU; removes repeated host→device copies on GPU (not measured) |
| CA `save_predictions` built 5 unused models | 5 models | 0 | Not timed separately |
| Backend re-parsed every CSV/YAML ~3× per page load | uncached | mtime/size-keyed cache | Not timed |
| Tensor allocation per step for NaN checks | `torch.isfinite(torch.tensor(x))` | `math.isfinite(x)` | Not timed |

End-to-end wall time (not a controlled comparison):

* Crucible: 2.99 s → 3.0 s.
* K-scaling: 5.28 s → 5.3 s.
* CA: 1102 s → 1185 s. The refactored CA run overlapped with test, build and
  server checks, and the controlled step benchmark shows no meaningful
  overhead.

**No performance improvement is claimed for the experiments.** The attention
implementation was deliberately left as-is: it is custom by requirement, and
N ≤ 64.

## Hardcoding Removed

* **Model shape.** `d_model 128`, `num_heads 4`, `d_ff 512`, `max_seq_len 64`,
  `α 0.9` and `dropout 0.0` were duplicated across three experiments, three
  scripts and the unused YAML. They now live once, in
  `configs/prototype_small.yaml`.
* **Experiment settings.** Seeds, K values, T values, rule, sequence lengths,
  dataset sizes, validation seed offset, learning rates, weight decay (now
  explicit), max steps, early-stop thresholds, batch size, eval and log
  cadence, benchmark warmup and timing counts, and Crucible pass criteria.
  These are now in `configs/{crucible,k_scaling,cellular_automaton}.yaml`,
  validated strictly and recorded per run.
* **Paths and device.** Output directories were cwd-relative; they now come
  from config plus `TESSERACT_RUNS_DIR` and `--output-dir`. Devices resolve
  through `utils/device.py` from config or `--device`.
* **Hand-typed metrics.** Removed from `generate_presentation_assets.py`, the
  backend and the frontend (parameter counts 210832/207234, latency and BPTT
  arrays, statuses, findings text, "100.0%", "N = 32", "256 / 64", Y-axis
  domains).
* **Summary literals.** K-scaling's summary text ("Answer: YES",
  "1, 2, 4, or 8") and plot label ("K=8 / K=1") are now derived.
* **Tests.** `tests/test_block.py`'s literal `198272` is now an analytic
  formula.

## Remaining Hardcoded Values (intentional)

| Value | Where | Why it stays |
|---|---|---|
| `std=0.02` init of z_H/z_L, fused `3·d_model` QKV, `√d_head` scale, GELU | models | Architectural definition, frozen for Phase 1 |
| `vocab_size = 2`, rule range 0–255, periodic boundary | CA dataset / config validation | Definition of a binary elementary CA |
| `K_VALUES = (1,2,4,8)`, probe batch 4×16 / 2×8 | `validate_prototype.py`, `viva_demo.py` | The depths the health check certifies; probe size is arbitrary |
| `NUM_PREDICTIONS_SHOWN = 8`, `NUM_EXAMPLES_SHOWN = 4`, console `step <= 5` prints | experiments | Display only; no effect on results |
| `OBSERVATION_MARGIN = 5.0` | CA summary | Wording threshold of the text summary only |
| "chance ≈ 50 %" | CA summary, backend finding | Binary balanced targets (verified; vocab 2 enforced by config) |
| Experiment list and raw-file names | `run_all_experiments.py`, backend `RAW_FILES` | Fixed, non-user-supplied file set (no arbitrary path access) |
| Diagnostic defaults `--cells 1:4 8:4`, `--control-train-size 4096` | `diagnose_ca_generalization.py` | CLI-overridable audit diagnostic, not a Phase 1 result |
| Small explicit model sizes in unit tests | tests | Tests should state their inputs |

## Remaining Warnings

1. **Nothing is committed.** Review and commit the working tree. `runs/k_scaling` and
   `runs/cellular_automaton` were regenerated. Their training numbers are
   identical, but latency, plots and the CSV schema differ (new columns
   `stopped_reason`, `eval_step`, `val_*_token_errors`). `runs/crucible_k04/`
   text artifacts are new; `model.pt` stays git-ignored.
2. **CUDA paths untested.** The torch build here has no CUDA. The CUDA
   synchronisation and peak-memory code paths were reviewed but not executed.
   GPU bit-reproducibility (`torch.use_deterministic_algorithms`,
   `CUBLAS_WORKSPACE_CONFIG`) is not configured.
3. **Dashboard not opened in a browser.** `oxlint` and `vite build` pass, and
   every API endpoint was exercised over HTTP against the real artifacts,
   including the CORS allow/deny check. `npm ci` from a clean checkout was not
   run (offline).
4. **Clean-run simulation was partial.** It used a copy of all tracked and
   non-ignored files (no `runs/`, `.venv` or `node_modules`) with the existing
   interpreter, no PYTHONPATH and a foreign working directory. There, pytest
   passed (65 s, while a diagnostic ran concurrently); validation, Crucible,
   K-scaling with `TESSERACT_RUNS_DIR`, presentation assets, the backend
   loader and the notebook all worked. The 20-minute CA grid was run
   in-repo, not in the copy. A fresh `pip install` was only resolved
   offline against installed packages.
5. **Frontend bundle size.** The bundle is 612 kB, so Vite warns about chunk
   size (recharts). This is cosmetic.
6. **`configs/full_7m.yaml` is stale.** It does not match the new config
   schema, and the ~7M dimensions disagree across ROADMAP (512/8/2048) and
   POST_AUG21/VIVA (768/12/3072). Deliberately untouched (future phase).
7. **Stale docs.** `ROADMAP.md` checkboxes are out of date.
8. **Latency is machine-dependent.** Quote it only together with its
   `run_metadata.json`.

## POST-PHASE-1 Items

These should **not** be changed before the next research phase starts, so that
Phase 1 stays comparable:

* **Architecture.** No final LayerNorm, fixed α = 0.9, learnable z_H/z_L
  init, update order, custom attention. Any change is a new, named ablation,
  not an edit.
* **CA configuration** (`t_values [1,2,4,8]`, 256/64 split, schedule, seeds).
  New T values (e.g. 3, 5, 6, 7, which are not two-cell XORs), larger
  training sets or relative positions belong in *new* config files.
* **Metric definitions:** token accuracy, exact match, and the pre-update
  semantics of `final_*`.
* **`prototype_small.yaml` and the three experiment configs** as the Phase 1
  record.

Recommended first steps for Phase 2 (not done here):

* Add a matched unrolled (non-shared) baseline.
* Add a non-power-of-two T grid and a training-set-size sweep.
* Add a CUDA determinism setting before any GPU runs.
