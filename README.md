# Tesseract

**Tesseract — Transformer Engineered for Sequential State Evaluation and Recursive Action**

Tesseract is a research prototype for parameter-efficient recursive reasoning in
transformers. A single shared transformer block is applied K times over dual
latent states (z_H, z_L), and the model is trained with full Backpropagation
Through Time (BPTT). Computation depth can therefore change without adding
trainable parameters.

## Research Question

> **Can shared-weight recursive transformer computation increase computation depth K while keeping trainable parameter count fixed?**

---

## Architecture (as implemented in `models/`)

```text
tokens [B, N]
  → token embedding + learned positional embedding            x      [B, N, D]
  → z_H^(0), z_L^(0) = learnable [D] vectors, broadcast to     [B, N, D]
  → for k = 1..K   (ONE TransformerBlock instance, reused every step)
        z_L^(k) = Block(z_L^(k-1) + z_H^(k-1) + x)
        z_H^(k) = α · z_H^(k-1) + (1 − α) · z_L^(k)
  → Linear(z_L^(K))                                            logits [B, N, V]
```

* `models/attention.py`: custom bidirectional multi-head self-attention (no mask).
* `models/block.py`: pre-LayerNorm block, `x + MHA(LN(x))`, then `+ FFN(LN(·))`.
* `models/recursive_core.py`: the K-step loop. No `.detach()`, so BPTT runs through all K steps.
* z_H^(K) is not consumed by the output head, so it receives no gradient (by design).

### Prototype configuration — `configs/prototype_small.yaml`

| d_model | heads | d_ff | max_seq_len | α | dropout | default K |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4 | 512 | 64 | 0.9 | 0.0 | 4 |

The trainable parameter count is always measured from the instantiated model,
never stored in a config. It is identical for every K: **210,832** at
vocab_size 16 (copy tasks) and **207,234** at vocab_size 2 (cellular
automaton). The per-component breakdown is in `AUDIT_REPORT.md` §A.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                   # full test suite
python scripts/validate_prototype.py        # health check for K = 1, 2, 4, 8
python scripts/run_all_experiments.py --skip-ca   # Crucible + K-scaling (seconds on CPU)
python scripts/run_all_experiments.py             # + cellular automaton (~20 min on a 16-core CPU)
python scripts/viva_demo.py                 # live shape trace, weight sharing, BPTT
```

Run a single experiment:

```bash
python experiments/crucible.py            [--config PATH] [--device auto|cpu|cuda] [--output-dir DIR]
python experiments/k_scaling.py           [...same options...]
python experiments/cellular_automaton.py  [...same options...]
```

Scripts work from any working directory. Each experiment exits non-zero when
it fails, and `run_all_experiments.py` reports the real exit status of each one.

---

## Configuration

Experiment settings have a single source of truth in `configs/`:

| File | Purpose |
|---|---|
| `prototype_small.yaml` | Model architecture (referenced by every experiment through `model.base`) |
| `crucible.yaml` | Copy-task overfitting + BPTT gate; pass criteria |
| `k_scaling.yaml` | K values, latency benchmark settings, optional training diagnostic |
| `cellular_automaton.yaml` | Rule, T/K grid, dataset sizes, training schedule |
| `full_7m.yaml` | Future scaling phase — **not used in Phase 1** |

Configs are validated strictly by `utils/config.py`. Unknown keys, missing
keys and wrong types all raise an error. For example, YAML `3e-4` is parsed
as a string, so write `0.0003`. `--device` and `--output-dir` override the
config, and the override is recorded in the run's saved `config.yaml`.

Device `auto` uses CUDA when available and falls back to CPU otherwise. An
explicit `cuda` request is an error if CUDA is unavailable.

## Experiment Artifacts

Each run writes to `runs/<output_dir>/`:

* `run_metadata.json`: run id, status (`running` / `completed` / `failed`),
  verdict, timestamps, seed, device and thread count, library versions, git
  commit and dirty flag, and command line.
* `config.yaml`: the fully resolved configuration.
* Results, for example `metrics.csv`, `summary.json` and
  `bptt_verification.json` (Crucible), `results.csv` and `summary.txt`
  (K-scaling, CA), and plots.

At the start of a run, the experiment's own previous artifacts are deleted.
A crashed run therefore leaves `status: failed`, not old results that look
like new ones.

## Phase 1 Results

The current numbers live in the artifacts (`runs/*/summary.*`) and in
`PHASE1_FINAL_AUDIT.md`. In brief:

* **Crucible:** PASS. The copy task (16 examples) reaches loss < 0.01 with 100 % exact match, and gradients reach z_L at all K = 4 steps.
* **K-scaling:** the parameter count is identical for K ∈ {1, 2, 4, 8}. Measured shared-block calls equal K. Forward latency grows with K (wall-clock on the recording machine).
* **Cellular automaton (Rule 90):** training exact match is high, but **validation exact match is 0 % in every (T, K) cell**, while validation token accuracy is well above chance. This is a genuine generalization failure, not a metric bug; see the audit.

### Known limitations

* For Rule 90, T ∈ {1, 2, 4, 8} are all powers of two, so every target is `x[i−T] XOR x[i+T]`: T does not increase how many input cells a target depends on.
* Activation norms grow with K (no final LayerNorm), so the initial loss grows with K. This confounds convergence-speed comparisons across K.
* Latency is wall-clock time, not FLOPs.

---

## Evaluation Dashboard (read-only)

The dashboard only visualises experiment artifacts. Every number it shows is
read from `runs/`, and a missing value is shown as "—".

```bash
# Terminal 1 — backend (from repository root)
pip install -r app/backend/requirements.txt
PYTHONPATH=app/backend python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd app/frontend && npm install && npm run dev
```

Open http://127.0.0.1:5173.

## Environment Variables

All are optional; see `.env.example`. Python code does **not** load `.env`
files, so export variables in your shell.

| Variable | Used by | Default |
|---|---|---|
| `TESSERACT_RUNS_DIR` | experiments, scripts, backend | `<repo>/runs` |
| `TESSERACT_CORS_ORIGINS` | backend | `http://localhost:5173,http://127.0.0.1:5173` |
| `VITE_API_BASE_URL` | frontend (Vite reads `app/frontend/.env`) | `http://localhost:8000` |

---

## Repository Structure

```text
tesseract/
├── configs/          # YAML configuration (single source of truth)
├── models/           # attention, block, embeddings, recursive core, full model
├── data/             # synthetic datasets (copy/reverse, cellular automaton)
├── training/         # train step, full-batch loop, BPTT verification, logger
├── evaluation/       # token accuracy, exact match, evaluation helper
├── experiments/      # crucible, k_scaling, cellular_automaton (+ shared CLI)
├── utils/            # config loading, device, paths, seeding, run metadata, param count
├── scripts/          # run-all, validation, viva demo, presentation figures, CA diagnostic
├── tests/            # pytest suite
├── notebooks/        # math sanity checks (imports project modules)
├── app/              # read-only dashboard: FastAPI backend + React frontend
├── runs/             # experiment artifacts
├── AUDIT_REPORT.md       # pre-change audit findings
├── PHASE1_FINAL_AUDIT.md # final Phase 1 audit
├── ROADMAP.md, POST_AUG21.md, VIVA.md
└── requirements.txt
```

## Research Contribution & Novelty Disclaimer

Tesseract is an implementation and experimental investigation of
shared-weight recursive transformer computation. It uses a dual-state
prototype and systematically evaluates how computation depth K affects
latency and gradient dynamics at a fixed trainable parameter count. It does
not claim to have invented recursive transformers or dual latent states.
