# Phase 2 — Batch 01 pre-registration

**Date: 2026-09-27. Committed before any arm in this batch was executed.**

Batch 01 is a set of diagnostics run against one common reference. Each arm
changes **exactly one thing** against that reference, so any movement is
attributable to the thing it changed. Nothing here is part of the Phase 2 grid,
and no result from this batch may be used as evidence for the depth hypothesis.

Constraints carried over unchanged: no frozen-file edits, every run through the
guarded runners, the test split never read, every failure and every interrupted
attempt preserved.

## 1. Reference configuration (REF)

REF is exactly Probe A. It has already been run, so no arm needs to re-run it —
the recorded Probe A artifacts are the comparison baseline.

| Term | Value |
|---|---|
| Benchmark | A₅ cellular automaton, n = 17, vocabulary 60 |
| Depth | **T = 4** (the shallower primary condition) |
| Width | medium, `phase2/prototype_medium`, 837,436 parameters |
| Recursion | K ∈ {1, 8} |
| Seeds | 0, 1, 2 |
| Optimiser | AdamW, lr 1e-3 constant, weight decay 0.01 |
| Batch size | 128 |
| Budget | 20,000 steps, no early stopping |
| Evaluation | every 250 steps, frozen validation split (2,048), test never read |
| Sampling | fresh uniformly random batches every step, val/test inputs excluded |
| Architecture | one shared block, no final LayerNorm |
| Recorded result | **K=1 NOT LEARNABLE, K=8 NOT LEARNABLE** (`runs/phase2/probe_a`) |

**Sampling note.** REF already trains on freshly generated samples every step;
there is no frozen train split. Memorisation and train-set size are therefore
already excluded as explanations for REF's result, and no arm in this batch
needs to test for them.

## 2. What each arm changes

| Arm | Single change against REF | Isolates |
|---|---|---|
| 1 | trains and evaluates on one small **fixed** set | whether the model can optimise at this depth at all |
| 2a | none (measurement only) | the size of the residual-stream confound |
| 2b | final LayerNorm **ON** | whether the residual confound causes the plateau |
| 3 | curriculum over T (Amendment 03) | whether a cold-start gradient is the blocker |
| 4 | LR warmup **and** 2× budget | whether the plateau is an optimisation artefact |
| 5 | the union of whichever of 2b/3/4 moved a cell | whether the levers combine |

Arm 4 changes two settings at once (schedule shape and budget). That is
deliberate and is the one exception to the one-change rule: both are properties
of the optimisation run rather than of the model or the task, and the arm is a
single plateau check rather than a sweep.

## 3. Success rule — identical for every arm

The bars are the P1c probe bars, unchanged, so every arm is directly comparable
with REF.

- **LEARNABLE** — validation token accuracy ≥ **6.6667 %** (chance 1.6667 % + 5
  points), **or** validation loss ≤ **3.99434** nats (ln 60 − 0.1), in at least
  **2 of 3 seeds**.
- **PARTIAL** — above chance on any seed but below the LEARNABLE bar.
- **NOT LEARNABLE** — indistinguishable from chance on all seeds.
- **Above chance** — chance + *z*·SE, SE over the validation split with the
  example as the independent unit, *z* = Φ⁻¹(1 − 0.05/M), M = number of
  validation evaluations.

**A single above-chance seed does not count as movement.**
[PHASE2_PROBE_REPORT.md](PHASE2_PROBE_REPORT.md) established that with a
per-cell false-positive rate of 0.05, one crossing is the modal outcome under
the null across several cells. An arm "moves a cell" only if it reaches
LEARNABLE, or if it reaches PARTIAL on **at least 2 of 3 seeds**.

Arm 1 is judged differently, because it is a fit test rather than a
generalisation test: see §4.

## 4. Arm 1 — fixed-set fit

**Fixed set size: 512 examples.** Justification: 512 × 17 = 8,704 target tokens,
which at log₂60 ≈ 5.9 bits is roughly 51 kbit of information — about 0.2 % of
the 837,436-parameter model's capacity at 32 bits per parameter. A model that
cannot drive training loss down on a set this small at T = 4 cannot be said to
be capable of representing the function at this depth at all. It is also large
enough (4 batches of 128) that fitting it is not a single-batch artefact.

The set is drawn from the training stream with the usual leakage guard, then
frozen for the run. **Training and evaluation both use this set** — this arm
measures fitting, not generalisation, and its numbers must never be reported as
task performance.

**Arm 1 success:** train exact match ≥ 99 % on the fixed set, for a given K.

**Gate:** if **both** K = 1 and K = 8 fail to fit, the batch stops after Arm 1
and is reported. Nothing downstream would be interpretable.

## 5. Arm 2 — residual-stream confound

**Measured problem.** Step-0 validation loss recorded in the probes: **4.43
nats at K = 1** against **33.97 nats at K = 8**. Without a final normalisation
the residual stream grows with the number of block executions, so K = 8 starts
an order of magnitude further from chance than K = 1. This sits directly on the
axis the experiment attributes its effect to.

**Proposed fix: a final LayerNorm applied to the state before the output
projection**, behind a config flag `final_layer_norm`, **default OFF**. It is
applied identically to every K, every family and every baseline when ON.

Rationale for this fix over the alternatives: scaling the residual update by
1/K or 1/√K changes the function the block computes and introduces a new
K-dependent term; a per-step norm changes the recursion itself. A single final
norm leaves the recursion untouched and only removes the scale dependence at
the read-out, which is precisely where the confound enters the loss. This is
option (b) of Amendment 02's open question Q2.

**Arm 2a** measures step-0 loss at K ∈ {1, 2, 4, 8}, with and without the fix.
No training. **Arm 2b runs only if the fix brings K = 8's step-0 loss near
K = 1's** — pre-registered threshold: K=8 step-0 loss within **0.5 nats** of
K=1's. If it does not, Arm 2b is skipped and that is recorded.

## 6. Arm 3 — curriculum (Amendment 03)

**Stage length decision, made now from measured data.** All six probe T = 1
runs reached the 99 % exact-match stop condition at **step 500** (the first
evaluation at which it could fire, since `eval_every` is 250; true convergence
is between 250 and 500). 500 ≪ 5,000, so the pre-registered 5,000-step stages
are **kept unchanged** — they already give 10× the observed convergence time.
No lengthening is warranted and none is applied.

Schedule: 5,000 steps each at T = 1, 2, 4, then 20,000 at the target T = 4 —
35,000 steps per cell. Identical for every K and every baseline. Optimiser
state is not reset between stages.

**Per-stage diagnostics.** At the end of each stage, validation accuracy is
logged at T = 1, 2 and 4. These are diagnostic only; success is judged at the
target T.

**Known risk, recorded in advance.** The T = 2 stage may learn the Amendment 01
shortcut rather than the depth-2 composition: F²(s)[i] collapses to the depth-1
form s[i]·s[i+2] whenever s[i+1]² = e, which holds for 16 of 60 elements. If
T = 2 accuracy plateaus near **26.7 %**, that is the shortcut, not composition,
and the report must say so. A T = 2 stage that plateaus there has taught the
model nothing about depth.

Arm 3 runs with the residual fix **OFF**, so it isolates the curriculum.

## 7. Arm 4 — optimiser plateau check

Pre-registered exactly; no tuning after seeing results.

| Setting | Value |
|---|---|
| Peak learning rate | **1e-3** — unchanged from REF |
| Warmup | linear from 0 to peak over the first **2,000 steps** (5 % of budget) |
| After warmup | cosine decay to **1e-4** (10 % of peak) at step 40,000 |
| Budget | **40,000 steps** (2× REF) |
| Everything else | REF |

Peak LR is deliberately left at REF's value so the arm does not introduce a new
magnitude as a second variable; only the schedule shape and the budget change.

## 8. Arm 5 — combination (conditional)

Runs **only if two or more of Arms 2b, 3, 4 individually move a cell** out of
NOT LEARNABLE, per §3. Otherwise it is skipped and the skip is recorded.

Budget if it runs: the curriculum schedule if the curriculum is among the
winners (35,000 steps), extended to 55,000 if the 2× budget is also among them
(15,000 of warm-up stages plus 40,000 at target). Otherwise 40,000.

## 9. Compute and disk plan

Runtimes from measured per-step times at T = 4, medium width, on the RTX 3050
with deterministic kernels: **K = 1 → 0.01206 s/step**, **K = 8 → 0.07954
s/step** (mean of 3 recorded probe cells each).

| Order | Arm | Steps/cell | Seeds | K=1/cell | K=8/cell | GPU-h |
|---|---|---|---|---|---|---|
| 1 | Arm 1 fixed-set fit | 20,000 | 1 | 4.0 min | 26.5 min | 0.51 |
| 2 | Arm 2a step-0 measurement | — | — | — | — | ~0.02 |
| 3 | Arm 2b REF + residual fix | 20,000 | 3 | 4.0 min | 26.5 min | 1.53 |
| 4 | Arm 3 curriculum | 35,000 | 3 | 7.0 min | 46.4 min | 2.67 |
| 5 | Arm 4 warmup + 2× steps | 40,000 | 3 | 8.0 min | 53.0 min | 3.05 |
| | **Arms 1–4 subtotal** | | | | | **7.78** |
| 6 | Arm 5 combination (conditional) | ≤55,000 | 3 | 11.1 min | 72.9 min | ≤4.20 |
| | **Worst-case total** | | | | | **11.98** |

**The worst-case total is 11.98 GPU-hours, against the 12-hour limit — a margin
of 0.02 h.** It does not exceed the limit, so the batch proceeds, but Arm 5 is
conditional and will not be started without reporting first.

**Disk.** Retention for every diagnostic arm is the final and best checkpoints
only, which is already what the guarded runner writes. 26 cells × 2 checkpoints
× 3.20 MB ≈ 166 MB, plus curves and JSON ≈ 15 MB, for **≈ 181 MB**. Free space
at pre-registration: **1.9 GB**, which is sufficient for the batch but tight
system-wide and worth watching.

## 10. Freezing decision table (Stage 3)

Committed here as given, to be applied mechanically to the Batch 01 results.

| Outcome | Action |
|---|---|
| Arm 1 fails | Stop. Report "cannot optimise at this depth". User decides next steps. |
| Some arm reaches LEARNABLE at T=4 | Draft Amendment 02 as final: fill width, step budget, residual fix (ON if Arm 2 helped **or is neutral**, since the confound must be removed regardless), training protocol (curriculum if Arm 3 was needed), and seeds (propose 3 vs 5 with cost). Then propose a T=8 probe under that exact protocol. **Do NOT mark Amendment 02 frozen.** |
| Only PARTIAL anywhere | Report. Propose the next single lever (e.g. T=3 in the curriculum, or a smaller-group ladder) with pre-registration. No freezing. |
| All arms NOT LEARNABLE | Report. Draft (don't finalise) the path-3 write-up outline: "unlearnable at reachable scale, with curriculum, confound fix and optimiser check all excluded". |

In every case the batch stops after Stage 3 and waits for approval. Freezing
Amendment 02 and launching the grid are not decisions this batch makes.

## 11. Out of scope for Batch 01

Explicitly not run: parameter or width scaling (P1b already showed width alone
does not escape the plateau; scaling is revisited only after something learns),
and **T = 8 in any arm**.

---

# Addendum 01 — Arm 2c, per-step normalisation

**Date: 2026-09-27. Appended after Arm 2a was measured and before Arm 2c was
executed. Nothing above this line has been edited; §§1–11 stand as originally
committed (`7dbd7b6`), including the parts this addendum supersedes.**

## A1.1 Why this addendum exists

§5 proposed a final LayerNorm before the output projection as the residual-
confound fix, and Arm 2a was read as confirming it because step-0 loss became
K-invariant. That reading was wrong. A normalisation immediately before the
vocabulary projection makes step-0 loss K-invariant **by construction**,
whatever the hidden state is doing, so it cannot be evidence that the confound
is gone.

Measuring the quantity the confound is actually about — the RMS of the hidden
state after each recursion step, untrained, 256 validation inputs — shows the
output-side norm leaves it untouched:

| K=8, z_L RMS per step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | growth |
|---|---|---|---|---|---|---|---|---|---|
| no fix | 1.43 | 3.00 | 4.86 | 7.16 | 10.09 | 13.89 | 18.83 | 25.30 | 17.70× |
| output-side norm | 1.43 | 3.00 | 4.86 | 7.16 | 10.09 | 13.89 | 18.83 | 25.30 | 17.70× |

Identical to every digit. **Arm 2b, which is already running, therefore tests
whether read-out normalisation changes learnability, and nothing more. The
report must state that, and must not describe Arm 2b as removing the
confound.**

## A1.2 Arm 2c — per-step normalisation inside the recursion

The chosen fix: **normalise z_L and z_H after every block call.** Rejected
alternatives and why — scaled residual updates (1/K or 1/√K) introduce a
K-dependent term into the function on the very axis under test; reporting-only
leaves every K comparison open to attack.

`models/` is frozen, so the fix does not edit the reasoner. It wraps it,
reusing the same shared `TransformerBlock` and the same learnable initial
states, and reproduces the recorded update rule with a normalisation of each
state at the end of the step. The two LayerNorms are shared across steps, as
the block is, so the added parameter count is independent of K: **+4·d_model**
(1,024 at d_model 256).

**Arm 2c runs in two parts.**

**2c-i — step-0 hidden-state RMS table at K ∈ {1, 2, 4, 8}.** No training.
Pre-registered pass condition: **the RMS must be bounded — growth ratio from
the first recursion step to the last below 1.05 at every K.** A fix that leaves
the stream growing fails here and Arm 2c-ii does not run.

**2c-ii — training.** T = 4, K ∈ {1, 8}, **3 seeds**, 20,000 steps, everything
else REF. Judged by the §3 success rule, unchanged.

## A1.3 Supersedes: the Stage 3 residual clause

The §10 clause "residual fix (ON if Arm 2 helped **or is neutral**, since the
confound must be removed regardless)" is replaced by:

> **a fix that bounds hidden-state growth, verified by per-step norms**

An output-side normalisation does not satisfy this. Only a fix whose measured
per-step RMS is bounded (§A1.2 pass condition) may be recorded in Amendment 02
as the residual fix. The rest of the §10 table is unchanged.

## A1.4 Parameter matching

`match_parameters` now takes the per-d_model cost of any added normalisation
and subtracts it at each candidate width, rather than tolerating the drift. The
cost cannot be folded into the target beforehand, because the baseline's
d_model is what is being solved for and its norms are cheaper than Tesseract's
at the matched width. Measured drift for the param_matched baseline at L = 8
improves from **−0.055 %** to **+0.033 %**.

Scope note: per-step normalisation applies to the shared-block family, which is
the family that has a recursion. The unrolled and parameter-matched baselines
execute distinct blocks and need the equivalent check made separately before
any cross-family comparison under the fix.

## A1.5 Compute

Arm 2c-ii costs **1.53 GPU-h** (identical in shape to Arm 2b); 2c-i is a
structural measurement of the order of a minute.

| | GPU-h |
|---|---|
| Arms 1–4 (§9) | 7.78 |
| **+ Arm 2c** | **9.31** |
| + Arm 5 worst case (4.20) | 13.51 |

**Arm 2c alone keeps the batch at 9.31 GPU-h, under the 12-hour limit, so it is
queued after Arm 4 as instructed.** Adding the conditional Arm 5 afterwards
would reach **13.51 GPU-h and exceed the limit**, so Arm 5 now requires
explicit approval on compute grounds as well as on its own trigger condition.
