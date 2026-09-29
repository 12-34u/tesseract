# Tesseract — write-up draft

**Status: draft. Not a freeze, not a result of record.** Assembled 2026-09-28 on
the `mac-diagnostics` branch while the GPU box was unreachable.

This document collects the results that are already settled, so that the
narrative can be written before Batch 01 finishes. Every section states where its
numbers come from. Sections whose data lives only on the GPU box are marked
**PLACEHOLDER** and contain no invented figures.

Nothing here amends a pre-registration. The authorities remain
[PHASE2_BENCHMARK_DESIGN.md](PHASE2_BENCHMARK_DESIGN.md),
[PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md) and the amendment drafts
in `configs/phase2/`; where this draft and those disagree, those win.

## Provenance legend

| Mark | Meaning |
|---|---|
| **Verified here** | Recomputed on this machine during this session. |
| **From artifacts** | Read from committed artifacts under `runs/`. |
| **From record** | Quoted from a committed report; the underlying artifacts are on the GPU box and were not re-read. |
| **From record (GPU console log)** | Transcribed from console output, not from an artifact: no run id, manifest, commit or status field, and not re-readable. Must be confirmed against artifacts. |
| **PLACEHOLDER** | No data available on this machine. Nothing is filled in. |

---

## 1. The research question, and what narrowed it

The project asks whether shared-weight recursive computation can buy depth
without buying parameters: one transformer block applied K times over dual latent
states (z_H, z_L), trained with full BPTT, so that K varies while the trainable
parameter count does not.

The parameter invariance is not in question and never was. It is measured from
the instantiated model at every K — **210,832** parameters at vocab 16 and
**207,234** at vocab 2, identical for K ∈ {1, 2, 4, 8} (**from artifacts**;
[AUDIT_REPORT.md](AUDIT_REPORT.md) §A). The open question has always been
whether the extra executions *do* anything, and the story of the project so far
is a sequence of benchmarks that turned out unable to answer it — each for a
different and instructive reason.

---

## 2. Phase 1: Rule 90 is degenerate on the T grid

**From artifacts** — `runs/cellular_automaton/summary.txt`, Rule 90,
sequence length 32, 256 train / 64 validation with zero overlap, 207,234
parameters, CPU.

The Phase 1 cellular-automaton experiment varied a transformation depth T
(number of rule applications) against recursive depth K. The headline result was
a clean generalisation failure:

| Validation | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| Exact match, all T | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| Token accuracy, T=1 | 68.8 % | 72.6 % | 79.5 % | 76.5 % |
| Token accuracy, T=2 | 80.4 % | 76.7 % | 82.1 % | 84.0 % |
| Token accuracy, T=4 | 80.1 % | 77.1 % | 79.2 % | 83.5 % |
| Token accuracy, T=8 | 70.7 % | 68.2 % | 74.6 % | 80.6 % |

Training exact match was high (90–99 % in most cells), so the models fit the
training set and did not transfer. Chance token accuracy is 50 %, so the models
learned something real per token and never assembled it into a correct sequence:
the best validation sequence in any cell still had at least one wrong token out
of 32.

**The reason the experiment cannot answer the research question is structural,
and the experiment itself detected it.** Its own task-structure check reports,
for every T ∈ {1, 2, 4, 8}:

```text
target_i == x[i-T] XOR x[i+T]   on all train+val data?   T=1: yes  T=2: yes  T=4: yes  T=8: yes
```

Rule 90 is `x[i-1] XOR x[i+1]`, which is linear over GF(2). Composing it T times
where T is a power of two makes the cross terms cancel, so `F^T` collapses to a
two-cell XOR at a wider spacing. Every T on the grid is a power of two.
Increasing T therefore widened the *span* the model had to reach across without
deepening the *computation* it had to perform — the target never depends on more
than two input cells. A depth experiment whose depth axis does not change
required depth is not a depth experiment.

This is the finding that motivated replacing the benchmark, and it sets the
design requirement the replacement had to meet: the composed function must
actually get harder with T.

*Presentation note:* the dashboard's Phase 1 matrix defaults to token accuracy
and shades cells by margin above chance, because a matrix of sixteen 0.0 % exact
match cells tells a reader nothing about how the cells differ. Commit `a06d42a`;
presentation only, no recorded value recomputed.

---

## 3. The A₅ benchmark, and the T = 2 shortcut (Amendment 01)

The replacement benchmark is iterated multiplication in the alternating group
A₅: states are length-17 sequences over the 60 elements of A₅, the step map is
`F(s)[i] = s[i] · s[i+1]`, and the target is `F^T`. A₅ is non-abelian and
unsolvable, evaluating general A₅ words is NC¹-complete, and a flat lookup over
a (T+1)-cell window would need 60^(T+1) entries — so the three failure modes
that killed Rule 90 (algebraic collapse, finite lookup, span-without-depth) are
addressed by construction. Chance token accuracy is 1/60 = **1.6667 %** and
chance loss is **ln 60 = 4.09434** nats.

**From record** — [PHASE2_BENCHMARK_DESIGN.md](PHASE2_BENCHMARK_DESIGN.md) §8.
The pre-registered gate run (`phase2_gates-20260915T090022Z`) nonetheless
returned **FAIL**, on one cell: gate G2, which caps how well any shallow
candidate may agree with the target, failed at T = 2. The best depth-1 candidate
`s[i]·s[i+2]` agreed with `F²` on **26.65 %** of tokens against a pre-registered
5 % limit.

The mechanism was then verified exactly on sampled states:

```text
F²(s)[i] = s[i] · s[i+1]² · s[i+2]
         = s[i] · s[i+2]            ⇔   s[i+1]² = e
```

In A₅, 16 of the 60 elements square to the identity — the identity itself plus 15
involutions — so on 16/60 ≈ **26.7 %** of tokens the T = 2 target has a depth-1
formula. The original design argument ("a word of 2^T letters") had not accounted
for these cancellations. The same candidate family peaks at **3.57 %** at T = 4
and **1.73 %** at T = 8, both within the limit.

Amendment 01's response was to keep the finding rather than the convenience:
**T = 2 became diagnostic and was removed from primary inference**, the 5 % G2
limit was *not* loosened, no benchmark config was edited, and the original
failing gate run is preserved as recorded. The amended re-run writes to a
separate directory and must reproduce the raw gate numbers exactly. The dashboard
still shows the raw FAIL.

The 26.7 % figure recurs below, and the coincidence is worth flagging in the
final write-up: it is the same number twice for the same algebraic reason, once
as a shortcut a model could exploit and once as the ceiling on partial credit.

---

## 4. P1b: the plateau is not capacity, budget, or instability

**From record** — [PHASE2_PROBE_PREREGISTRATION.md](PHASE2_PROBE_PREREGISTRATION.md)
§1. The P1b capacity/learnability diagnostic established that the A₅ benchmark at
T = 8 is **not learned**, and ruled out the three cheapest explanations:

* **not capacity** — small and medium widths behave alike;
* **not budget** — flat throughout 20,000 steps;
* **not instability** — no divergence or oscillation to blame.

P1b is explicitly a diagnostic, not evidence for or against the architecture
hypothesis, and it is the source of two values Amendment 02 must read rather than
choose: model width and step budget.

> **PLACEHOLDER — P1b per-seed table.** The per-cell numbers live in the GPU
> box's `runs/phase2/` artifacts, which are not present on this machine. Fill in
> width × K × seed with best validation accuracy, each cell's own above-chance
> bar, and final loss.

### The partial-credit analysis (P1c)

**From record** — the same section, and
[PHASE2_PROBE_REPORT.md](PHASE2_PROBE_REPORT.md) §5. The P1c signal analysis
explains *why* the plateau is flat. Predicting a target token from all but one
input cell gives **26.7 %**; from all but two, **13.4 %**; from all but three,
**7.0 %**.

The signal lives at near-full interaction order. A model that has learned most of
the function but not all of it is barely better than one that has learned none of
it, so there is almost no partial-credit gradient to climb. This is the strongest
single explanation the project has for the flatness, and it reframes the failure
from "the architecture cannot represent this" to "this task supplies no usable
learning signal from a cold start" — a weaker and more defensible claim.

It is also the pre-registered justification for Amendment 03's curriculum over T:
if the gradient is missing at T = 4 but present at T = 1, staging the depth is the
principled remedy.

---

## 5. Probes A and B: flat on both axes

**From record** — [PHASE2_PROBE_REPORT.md](PHASE2_PROBE_REPORT.md), classified
mechanically by `phase2/probes.py` against a rule committed (`da95a8e`) before
either probe ran. Medium width, 837,436 parameters, 20,000 steps, seeds 0/1/2,
RTX 3050, torch 2.13.0+cu129, deterministic kernels, TF32 off. LEARNABLE required
accuracy ≥ 6.6667 % or loss ≤ 3.99434 nats in at least 2 of 3 seeds.

* **Probe A** — is T = 4 learnable at all, from a cold start?
* **Probe B** — is T = 8 reachable when warm-started from a solved T = 1 model at
  the same K?

| Probe | K | Classification |
|---|---|---|
| A (T = 4) | 1 | NOT LEARNABLE |
| A (T = 4) | 8 | NOT LEARNABLE |
| B (T = 8) | 1 | NOT LEARNABLE |
| B (T = 8) | 8 | PARTIAL |

All twelve cells sit within roughly 0.13 percentage points of chance. No
accuracy came within 4.7 points of the LEARNABLE bar, and **no loss fell below
ln 60 at all** — every final loss sits *above* chance loss.

**The single PARTIAL is noise, and the report says so on three independent
grounds.** Probe B, K=8, seed 1 scored 1.9273 % against its own 1.9069 % bar, a
margin of 0.0204 points. (1) The bars are set at a per-cell false-positive rate
of 0.05, so across twelve cells the probability of at least one crossing under
the pure null is 1 − 0.95¹² ≈ **46 %**, expected count 0.6 — one crossing is the
modal outcome of no effect whatsoever. (2) That same cell's final loss is 4.0979
nats, *above* chance; genuine partial signal should push loss down. (3) It is not
the deepest or most favoured cell — its sibling seeds scored 1.7923 % and
1.8181 %, below several K=1 cells. No ordering by K or seed fits it.

**The warm start is what makes Probe B interpretable.** Each stage-1 T=1 solution
reached 100 % validation exact match (early-stopping at step 500), then scored
**1.6918 %** on the T=8 split before any T=8 training — chance to four figures,
at a loss of 15.5–16.2 nats. So Probe B's numbers are not inherited T=1
structure. It began from a genuinely solved neighbour and still went nowhere.

**What this establishes:** neither primary depth condition is learnable by a
medium-width model cold within 20,000 steps, and T = 8 is not reachable even from
a solved T = 1 solution at the same K.

**What it does not establish:** that the benchmark is infeasible. A result
uniformly flat across *both* the depth axis and the recursion axis cannot
discriminate "cannot express at this scale" from "provides no usable gradient
cold". The second is the weaker claim and the one P1c points at. Infeasibility is
concluded only if Amendment 03's curriculum also fails.

No benchmark, model, metric or decision rule was changed in response to these
results, and both interrupted attempts of the final Probe B cell were preserved
in sibling directories rather than overwritten.

---

## 6. The output-side LayerNorm reversal

This is the project's clearest instance of a confirmed result being withdrawn,
and it belongs in the write-up as method rather than as an aside.

**The confound.** Nothing bounds z_L between block executions, so its norm grows
with the number of executions. A K=8 model therefore *starts* training in a
different regime from a K=1 model, which contaminates every comparison along the
K axis — the axis the entire project is about. **From record**
([PHASE2_PROBE_REPORT.md](PHASE2_PROBE_REPORT.md) §4), the pre-registered
auxiliary measurement records step-0 validation loss of **4.43 nats at K=1** and
**33.97 nats at K=8**.

**The reversal.** §5 of the Batch 01 pre-registration proposed a final LayerNorm
before the output projection as the fix, and Arm 2a was initially read as
confirming it, because step-0 loss became K-invariant. **That reading was
wrong.** A normalisation immediately before the vocabulary projection makes
step-0 loss K-invariant *by construction*, whatever the hidden state is doing, so
K-invariant step-0 loss is not evidence that the confound is gone. Corrected in
commit `cf78bba`; the alternative fix is Addendum 01 (`b64305f`).

**What settled it** was measuring the quantity the confound is actually about —
the RMS of the hidden state after each recursion step, untrained. The output-side
norm leaves it untouched, to every digit printed:

| K=8, z_L RMS per step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | growth |
|---|---|---|---|---|---|---|---|---|---|
| no fix | 1.43 | 3.00 | 4.86 | 7.16 | 10.09 | 13.89 | 18.83 | 25.30 | 17.70× |
| output-side norm | 1.43 | 3.00 | 4.86 | 7.16 | 10.09 | 13.89 | 18.83 | 25.30 | 17.70× |

**Consequences that the Batch 01 report must respect.** Arm 2b, which tests the
output-side norm, therefore measures whether *read-out* normalisation changes
learnability — and nothing more. It must not be described as removing the
confound. Addendum 01 §A1.3 accordingly replaces the Stage 3 residual clause
with a requirement on **a fix that bounds hidden-state growth, verified by
per-step norms**, which an output-side normalisation does not satisfy at any
measured value.

The methodological lesson generalises: when a fix is validated by a symptom the
fix defines away, the validation is circular. The symptom (step-0 loss) had to be
replaced by the mechanism (per-step RMS) before the question could be answered.

---

## 7. Per-step normalisation bounds the state — Mac CPU diagnostic

**Verified here**, and **not an Arm 2c result.** Arm 2c is a GPU *training* arm.
What follows is an untrained structural measurement, run on CPU because it needs
no training: `diagnostics/per_step_norm.py`, driven by
`diagnostics/run_per_step_norm_rms.py`. Platform: darwin, CPU, torch 2.13.0,
medium width (d_model 256), the T=4 validation split, first 256 examples, seed 0
before each build — the same conditions as `scripts/phase2_residual_norms.py`.

The candidate fix normalises z_L and z_H after *every* block call, inside the
recursion, using two LayerNorms shared across steps (so the parameter cost stays
independent of K). Measured per-step z_L RMS, flag OFF vs ON:

| condition | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | growth |
|---|---|---|---|---|---|---|---|---|---|
| K=1, OFF | 1.43 | — | — | — | — | — | — | — | 1.00× |
| K=2, OFF | 1.43 | 3.00 | — | — | — | — | — | — | 2.10× |
| K=4, OFF | 1.43 | 3.00 | 4.86 | 7.16 | — | — | — | — | 5.01× |
| K=8, OFF | 1.43 | 3.00 | 4.86 | 7.16 | 10.09 | 13.89 | 18.83 | 25.30 | 17.70× |
| K=1, ON | 1.00 | — | — | — | — | — | — | — | 1.00× |
| K=2, ON | 1.00 | 1.00 | — | — | — | — | — | — | 1.00× |
| K=4, ON | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — | 1.00× |
| K=8, ON | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00× |
| K=8, ON, affine=False | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00× |

**Answer: with the flag ON, RMS stays bounded across K.** The final-step z_L RMS
varies by **0.00 %** across K ∈ {1, 2, 4, 8}, against **1669.68 %** with the flag
off. The same holds for z_H (1.000 at every K, against 0.144 → 7.071 off) and for
the largest single activation (4.76–4.80, against 7.00 → 123.40 off), so the
result is not an artefact of averaging.

Three checks support the measurement:

1. **The flag-OFF column reproduces the recorded figures exactly** — 1.43 through
   25.30, growth 17.70× — matching Addendum 01's independently-run table. The
   harness agrees with the GPU box on the condition they share.
2. **Flag OFF is bit-identical to the frozen recursion.** With the flag off
   nothing is wrapped at all, and a test asserts exact equality (`atol=0`)
   against `models/recursive_core.py`.
3. **Independent agreement with `phase2/batch01.py`.** This module re-derives the
   recursion rather than importing the GPU-side implementation; run against it,
   the maximum absolute difference in per-step z_L RMS is **0.0**. The two
   implementations agree exactly, which is a check on both.

Parameter cost is **1024** (two LayerNorms over d_model 256), identical at every
K, so the fix does not disturb the parameter-invariance claim of §1.

**The caveat that keeps this from being the whole answer.** RMS = 1.000 at every
step is close to what LayerNorm does *by definition* at initialisation, where its
gain is 1 and its bias 0. What the measurement shows is that the normalisation is
placed where the growth happens — unlike the output-side norm, which left the
identical 1.43 → 25.30 profile — not that the state stays bounded *after
training*, when the learned gains are free to grow. As pre-registered, Arm 2c must
re-measure per-step norms at the end of training before Addendum 01 §A1.3's
requirement can be recorded as satisfied.

**A non-affine variant would make the bound structural.** With
`elementwise_affine=False` there is no learnable gain to drift, so unit RMS holds
throughout training rather than only at initialisation, and the fix adds **zero**
parameters instead of 1024 (last row of the table above; measured identically).
The two variants are numerically **identical** on an untrained model — maximum
per-step z_L RMS difference 0.0, because gain is 1 and bias 0 at init — so no
untrained measurement can choose between them; the case rests on what each permits
during training, against the loss of a learned per-feature rescaling inside the
recursion. Written up as a proposal in
[PROPOSED_ADDENDUM_02_DRAFT.md](PROPOSED_ADDENDUM_02_DRAFT.md). **Not
pre-registered, not in force, and awaiting a decision on the GPU box before Arm 2c
runs.**

Local suite: 756 passed, 1 skipped, including 14 tests for this module.

---

## 8. Batch 01 — PLACEHOLDER

> **No Batch 01 artifact is reachable from this machine.** `runs/` contains only
> the three Phase 1 directories; there is no `runs/phase2/batch01/`. The arms are
> GPU-only by pre-registration (Amendment 02a P1) and `scripts/phase2_batch01.py`
> exits 2 without CUDA.
>
> §8.1 has since been re-read from the GPU box's artifacts
> (`runs/phase2/batch01/arm1_fixed_set/arm1_report.json`) and no longer depends
> on transcribed console output. Its figures are final. All other sections stay
> empty until they can be filled the same way.
>
> Fill each remaining section from the GPU box's artifacts. Do not fill any of
> them from this draft.

### 8.1 Arm 1 — the optimiser can fit a fixed 512-example set

**From artifacts.** Read back from
`runs/phase2/batch01/arm1_fixed_set/arm1_report.json` on the GPU box, not
transcribed from console output. There is exactly one Arm 1 run of record: an
earlier attempt was interrupted by session teardown at step 10,000 of K=1 and
wrote no artifact, so it left no competing result. The gate field in the
artifact reads `gate_passed: true`.

Design ([PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md) §4, fixed-set arm):
512 examples, justified as 512 x 17 = 8,704 target tokens, about 0.19 % of the
model's capacity at 32 bits per parameter.

| K | seed | fit exact match | token accuracy | loss | steps | status |
|---|---|---|---|---|---|---|
| 1 | 0 | 100.00 % | 100.00 % | 0.004741 | 20,000 | **FITS** — final |
| 8 | 0 | 100.00 % | 100.00 % | 0.006006 | 20,000 | **FITS** — final |

The two loss figures quoted in earlier progress notes, 0.0047 and 0.0060, are
K=1 and K=8 of this single run, not two attempts at the same cell.

**Arm 1 does not fail.** The optimiser fits the fixed set to 100.00 % exact
match and 100.00 % token accuracy at both K, so the Stage 3 row "Arm 1 fails ->
stop, cannot optimise at this depth" is **not** the row the batch is in. Both K
values reached 99 % exact match by step 250, the first evaluation; K=8 fits as
fast as K=1 despite starting 29.5 nats higher in loss, so the residual
confound does not impede fitting.

Caveats, in order of how much they constrain the claim:

* **The lookup-table caveat, which must appear wherever this result is
  reported.** A fixed 512-example set is small enough that fitting it is
  consistent with memorising it. This is evidence about the **optimiser** — that
  gradients at this depth are usable at all — and not about depth, composition or
  generalisation. It is a necessary condition for the batch to continue, never
  evidence for the architecture hypothesis. The benchmark design's coverage
  diagnostic (the fraction of test windows whose exact pattern was seen in
  training — high at T ≤ 2, ≈ 0 at T ≥ 4) is what bounds how much memorisation
  explains, and should be reported alongside.
* **Fit accuracy is not validation accuracy.** 100 % here is measured on the set
  being fitted. Phase 1's Rule 90 cells also reached 90–99 % training exact match
  while sitting at 0.0 % validation exact match (§2), so a fit number carries no
  generalisation claim whatsoever.
* **One seed.** The pre-registered classifications are defined over 3 seeds. A
  single seed cannot produce a classification, and "FITS" above is a description
  of that run, not a pre-registered verdict.
* **One seed at each K**, as Arm 1 was pre-registered — it is a fit gate, not a
  classified arm, so no above-chance bar applies to it.

> **Still needed from the artifacts:** the coverage diagnostic that bounds how
> much of the fit memorisation explains.

### 8.2 Arm 2b — output-side normalisation. PLACEHOLDER

> Per §6: report this as testing whether **read-out** normalisation changes
> learnability. Do not describe it as removing the residual confound, whatever
> the outcome.

### 8.3 Arm 2c — per-step normalisation. PLACEHOLDER

> The training arm. §7 establishes only the untrained structural property.
> **Needed:** learnability outcome, plus per-step norms measured *after* training
> to satisfy Addendum 01 §A1.3.
>
> **Open decision before this arm runs:** whether the normalisation is affine or
> non-affine. [PROPOSED_ADDENDUM_02_DRAFT.md](PROPOSED_ADDENDUM_02_DRAFT.md)
> argues for non-affine — the bound becomes structural rather than
> initialisation-only, at zero parameters — and states the trade-off against it.
> That proposal is not pre-registered and not in force; if it is rejected, this
> arm runs affine as Addendum 01 specifies and owes the post-training
> measurement above.

### 8.4 Arm 3 — curriculum over T. PLACEHOLDER

> **Needed:** per-stage validation accuracy at T = 1, 2 and 4, and explicitly
> whether the T = 2 stage plateaued near **26.7 %** — which would be the
> Amendment 01 shortcut of §3 being learned rather than the composed function,
> a risk recorded in advance in the pre-registration.

### 8.5 Arm 4. PLACEHOLDER

### 8.6 Arm 5. PLACEHOLDER — not run; held pending approval.

### 8.7 Stage 3 classification. PLACEHOLDER

> The §10 table keys every row on an arm outcome. One row can now be excluded:
> **"Arm 1 fails → stop" does not apply**, since Arm 1 fits the fixed set (§8.1).
> No row is *entered*, because every remaining row turns on a LEARNABLE or PARTIAL
> classification at T=4, which is defined over 3 seeds and which no arm has yet
> produced. In particular the batch is **not** in "all arms NOT LEARNABLE", which
> is a finding about measured arms rather than a default for unmeasured ones.
>
> Apply the table only once the artifacts are in hand, and read the residual clause
> in its amended form (§6): the fix recorded in Amendment 02 must bound
> hidden-state growth, verified by per-step norms, which the output-side norm does
> not.

---

## 9. The shape of the argument so far

Stated without the Batch 01 data, the defensible claim is narrow:

1. Parameter count is genuinely invariant in K, and BPTT genuinely reaches every
   step. The mechanism works as described (**from artifacts**).
2. Two benchmarks have now failed to make the K axis informative, for two
   *different* reasons — Rule 90 by algebraic degeneracy that flattened its depth
   axis (§2), A₅ by supplying no low-order gradient to climb (§4).
3. On A₅ at its primary depths, results are flat across both axes, cold and
   warm-started, and the one apparent exception is quantifiably noise (§5).
4. The leading explanation is signal/optimisation rather than representation, and
   it is falsifiable: Amendment 03's curriculum is the pre-registered remedy, and
   infeasibility is concluded only if it fails (§4, §5).
5. A K-axis confound was identified, mis-fixed, caught, and re-fixed at the right
   place, with the correction recorded rather than quietly amended (§6, §7).

What cannot yet be claimed is anything about whether recursive depth helps. No
experiment to date has produced a condition in which it *could* have shown up.
That is the gap Batch 01 exists to close, and §8 stays empty until it does.
