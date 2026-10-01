# Tesseract — write-up draft

**Status: draft. Not a freeze, not a result of record. Not finalised.**
Assembled 2026-09-28 on the `mac-diagnostics` branch while the GPU box was
unreachable; brought onto `main` and updated **2026-09-30**, when the Batch 01
artifacts became reachable.

This document collects the results that are already settled, so that the
narrative can be written before the remaining probes finish. Every section
states where its numbers come from. Sections whose data does not exist yet are
marked **PLACEHOLDER** and contain no invented figures.

**§10's outline is superseded by Batches 03 and 04 — see the dated note in §9a.
It is not rewritten yet, and the reasons are there.**

**Two things were outstanding when this draft was assembled, and the draft must
not be finalised before both land.** First, the Batch 02 T = 3 probe is running as this is written
([PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md)); its outcome decides
whether the learnability boundary claim in §10 is "T = 2" or something else.
Second, §10's outline is an outline — a plan for the path-3 write-up, not the
write-up.

Nothing here amends a pre-registration. The authorities remain
[PHASE2_BENCHMARK_DESIGN.md](PHASE2_BENCHMARK_DESIGN.md),
[PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md),
[PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md) and the amendment drafts
in `configs/phase2/`; where this draft and those disagree, those win. Two claims
of the Batch 01 report are withdrawn by
[PHASE2_BATCH_01_REPORT_ERRATUM_01.md](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) and
must not reappear here.

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

## 8. Batch 01 — complete

**The blocking note that stood here is discharged.** Batch 01 finished
2026-09-30 at 00:08:44 with every arm at `rc=0`, and the artifacts are under
`runs/phase2/batch01/`. §§8.2–8.7 are filled from them, not from the earlier
draft and not from console output.

**Headline: no arm moved.** Not one of the four reached LEARNABLE at T = 4, and
under the Addendum 03 amended rule not one reaches PARTIAL either
([PHASE2_BATCH_01_REPORT.md](PHASE2_BATCH_01_REPORT.md);
`runs/phase2/batch01/amended_classification.json`).

| arm | lever | K=1 original → amended | K=8 original → amended | moves |
|---|---|---|---|---|
| 2b | output-side LayerNorm | PARTIAL → **DEGENERATE** | PARTIAL → **DEGENERATE** | no |
| 3 | curriculum over T | NOT LEARNABLE → NOT LEARNABLE | NOT LEARNABLE → NOT LEARNABLE | no |
| 4 | warmup + 2× budget | NOT LEARNABLE → NOT LEARNABLE | NOT LEARNABLE → NOT LEARNABLE | no |
| 2c | non-affine per-step norm | PARTIAL → **NOT LEARNABLE** | PARTIAL → **DEGENERATE** | no |

Both PARTIALs the original rule produced were artifacts: each rested on a cell
emitting one or two distinct tokens out of sixty and scoring the T = 4
majority-class baseline of 1.9244 % exactly — the defect Addendum 03 was written
to close.

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

### 8.2 Arm 2b — output-side normalisation changes nothing

**From artifacts** (`arm2b_residual_fix__attempt2/arm2b_residual_fix_report.json`,
`amended_classification.json`).

As §6 requires, this is reported as testing whether **read-out** normalisation
changes learnability. It does not, and it does not remove the residual confound:
§3 of the Batch 01 report measures the hidden state directly and the growth
ratio at K = 8 is 17.70× with the output-side norm in place, identical to
without it.

**K = 1 and K = 8 both: PARTIAL → DEGENERATE.** All six cells emitted 1–3
distinct tokens; both seed-2 cells predicted token 11 at all 34,816 positions,
which is the modal class. Their accuracy was real and their learning was nil.

Reporting constraint: per
[the erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) §1, the distinct-token count
is used here only to refuse the above-chance claim. It is **not** to be compared
across arms as a measure of expressiveness — at chance the argmax token is
arbitrary.

### 8.3 Arm 2c — the bound holds exactly, and buys nothing

**From artifacts** (`arm2c_per_step_norm_train/arm2c_per_step_norm_train_report.json`).

The open decision §7 recorded was settled before the arm ran:
[PHASE2_BATCH_01_ADDENDUM_02.md](PHASE2_BATCH_01_ADDENDUM_02.md) adopted the
**non-affine** normalisation, so the bound is structural rather than
initialisation-only and costs zero parameters.

The post-training measurement Addendum 01 §A1.3 owed is in the artifact:
**all six cells recorded a post-training growth ratio of exactly 1.0000×**, so
the bound held after training as well as at initialisation. Arm 2c is the only
arm that removes the confound rather than its symptom.

**K = 1: PARTIAL → NOT LEARNABLE. K = 8: PARTIAL → DEGENERATE.** Learnability
did not change.

Reporting constraint: the earlier reading that per-step normalisation "made
collapse worse" and "cost expressiveness" is **withdrawn**
([erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) §1) and must not appear. The
Addendum 02 draft's predicted trade-off was not observed; it was not
measurable at chance.

### 8.4 Arm 3 — the curriculum solves T = 2 and then goes nowhere

**From artifacts** (`arm3_curriculum/arm3_curriculum_report.json`) for the
classification and the end-of-run per-depth numbers; **from record (GPU console
log)** for the per-stage train-loss trace, which is the deviation below.

The most informative arm in the batch. Train loss at each stage boundary, against
chance = ln 60 = 4.0943:

| cell | T=1 end | T=2 end | T=4 (first stage) | T=4 end |
|---|---|---|---|---|
| K=1 s0 | 0.0001 | 0.0006 | 4.1070 | 4.0958 |
| K=1 s1 | 0.0000 | 0.0023 | 4.1054 | 4.0959 |
| K=1 s2 | 0.0000 | 0.0001 | 4.1058 | 4.0957 |
| K=8 s0 | 0.0001 | 0.0473 | 4.1044 | 4.0969 |
| K=8 s1 | 0.0000 | **4.1009** | 4.1008 | 4.0968 |
| K=8 s2 | 0.0000 | 0.0164 | 4.1296 | 4.0975 |

**T = 1 is solved in every cell and T = 2 in five of six, on freshly generated
samples, and then T = 4 sits at chance from the first stage onward.** The one
exception, K = 8 seed 1, failed the T = 2 stage too, which is worth noting
against any claim that K = 8 has an advantage here. **K = 1 and K = 8 both:
NOT LEARNABLE → NOT LEARNABLE.**

**The 26.7 % shortcut risk did not materialise.** The pre-registration recorded
in advance that a model learning the Amendment 01 depth-2 shortcut instead of
depth-2 composition would plateau near 26.7 % (§3). No cell plateaued there; the
T = 2 stages reached ~100 % train accuracy, which is composition. (Batch 02
measures this directly rather than inferring it from the absence of a plateau —
see §8.8.)

**The earlier solutions are erased.** The per-depth diagnostics on the final
model read T=1 1.66 %, T=2 1.65 %, T=4 1.85 % — chance everywhere. A model that
provably solved T = 1 and T = 2 mid-run ends with no trace of either. The
[erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) §3 offers the interpretation: at
T = 4 the loss rewards partial knowledge almost nothing (the P1c ceilings are
26.7 % / 13.4 % / 7.0 % for windows short by one, two and three cells), so a
near-uniform output is a local optimum and the T = 4 stage moves *away* from the
T = 1 and T = 2 solutions rather than merely failing to build on them.

**Deviation, recorded and since fixed.** Addendum 01 §A1.2 specified logging
validation accuracy at T = 1, 2, 4 *at the end of each stage*; the
implementation measured the final model at each depth once, after training. So
the table above is console output rather than a recorded artifact field, and the
end-of-run numbers describe catastrophic forgetting rather than what each stage
achieved. `phase2.training.train_fixed_steps` now takes a `step_hooks`
parameter and Batch 02 records the measurement at each boundary as a report
field ([PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md) §7). **Arm 3's
own numbers are not retrofitted** — the deviation stands as recorded.

**Do not describe the 2 → 4 gap as a "cliff".** That claim is replaced by the
[erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) §2: T = 2 is solved, T = 4 is at
chance, and the shape between them is what Batch 02 measures.

### 8.5 Arm 4 — the plateau is not the optimiser schedule

**From artifacts** (`arm4_warmup/arm4_warmup_report.json`).

Warmup over 2,000 steps to an unchanged 1e-3 peak, cosine decay to 1e-4, double
the budget (40,000 steps). **No effect: K = 1 and K = 8 both NOT LEARNABLE**,
all six cells at chance, all with 49–60 distinct tokens, so none of them is
degenerate — the arm made a genuine attempt and the task resisted it. The
plateau is not a step-budget or a schedule problem.

### 8.6 Arm 5 — not approved, not run

**From the criteria file**
([PHASE2_BATCH_01_ARM5_CRITERIA.md](PHASE2_BATCH_01_ARM5_CRITERIA.md) §2), which
was committed before any arm beyond Arm 1 reported. Arm 5 is approved only if
**≥ 2** of arms {2b, 3, 4, 2c} individually move. **Zero moved**, so the batch
lands in §4 of the criteria file, "if no arm moves". Recorded rather than
inferred.

The compute constraint would have bound independently: Arms 1–4 plus 2c came to
9.31 GPU-h and Arm 5's worst case would have reached 13.51, over the 12-hour
limit.

### 8.7 Stage 3 — the row that applies

The blocking note that stood here was right not to enter a row early, and the
row is now determined by measured arms rather than assumed for unmeasured ones:

* Arm 1 did not fail (§8.1), so "Arm 1 fails → stop" is excluded.
* No arm reached LEARNABLE at T = 4.
* Under the amended rule there is **no PARTIAL anywhere** in Batch 01.

**The row that applies is "all arms NOT LEARNABLE"**, with the batch's results
being NOT LEARNABLE or the stricter DEGENERATE. Its action: *report, and draft
(do not finalise) the path-3 write-up outline* — which is §10.

The residual clause is read in its amended form (§6): only Arm 2c's per-step
normalisation bounds hidden-state growth, verified by post-training per-step
norms (§8.3). The output-side norm does not, and Arm 2b is not reported as
removing the confound.

### 8.8 Batch 02 — the T = 3 probe. PLACEHOLDER — running

> **Pre-registered in [PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md),
> committed before launch; running as this draft is written.** Arm 3's
> curriculum with the ladder's top moved from 4 to 3 and nothing else moved:
> T = 1 → 2 → 3, 5,000 steps per stage, 20,000 at the target, K ∈ {1, 8},
> 3 seeds, medium width, no fix. ≈ 2.72 GPU-h.
>
> **Needed:** the classification at T = 3 under both rules with T = 3's own
> majority-class floor of **1.8066 %**; the stage-boundary validation accuracies
> at T = 1, 2, 3 (the §8.4 deviation, now a report field); and the shortcut
> attribution.
>
> **This section must report the G2 failure at T = 3 whatever the outcome.**
> `window_product_forward` = s[i]·s[i+1]·s[i+2]·s[i+3] agrees with F³ on
> **26.67 %** of tokens — exactly where (s[i+1]·s[i+2])² = e, and 16 of A5's 60
> elements square to the identity, the same mechanism as the documented T = 2
> failure. The LEARNABLE bar is 6.67 %, so accuracy alone cannot separate
> depth-3 composition from the shortcut, and a LEARNABLE verdict here is not
> interpretable without the per-cell shortcut agreement. Do not report a T = 3
> accuracy without it.

---

## 9. The shape of the argument

1. Parameter count is genuinely invariant in K, and BPTT genuinely reaches every
   step. The mechanism works as described (**from artifacts**).
2. Two benchmarks have now failed to make the K axis informative, for two
   *different* reasons — Rule 90 by algebraic degeneracy that flattened its depth
   axis (§2), A₅ by supplying no low-order gradient to climb (§4).
3. On A₅ at its primary depths, results are flat across both axes, cold and
   warm-started, and the one apparent exception is quantifiably noise (§5).
4. A K-axis confound was identified, mis-fixed, caught, and re-fixed at the right
   place, with the correction recorded rather than quietly amended (§6, §7).
5. **The signal/optimisation explanation has now been tested and four candidate
   mechanisms behind it are excluded** (§8). The pre-registered remedy —
   Amendment 03's curriculum — ran and failed. Under the rule recorded in
   advance ([PHASE2_PROBE_REPORT.md](PHASE2_PROBE_REPORT.md), and the memory of
   the project's own probe-path fork), that is the point at which the
   infeasibility reading becomes available rather than premature.
6. What the curriculum did establish is the sharpest positive result in the
   project: **a depth-2 composition over A₅ is learnable at this width**, in
   five of six cells, on freshly generated samples. The failure is therefore
   located at a depth, not at the task.

What still cannot be claimed is anything about whether recursive depth helps. No
experiment to date has produced a condition in which it *could* have shown up —
K = 1 and K = 8 fail together, and they fail at the same depth. That is the
honest shape of the negative result, and §10 is the plan for writing it.

---

## 9a. NOTE, 2026-10-01: §10's path-3 outline is superseded

**Dated note. Nothing below is rewritten; §10 stands as committed so the change
in the argument is visible rather than edited away.**

§10 outlines a **negative-result** write-up: "unlearnable at reachable scale,
with curriculum, confound fix and optimiser check all excluded". That outline was
correct for the evidence at the time — Batch 01's Stage 3 row was "all arms NOT
LEARNABLE" and nothing in the project had moved at any depth above T = 2.

**Batches 03 and 04 superseded it.** In order:

| | |
|---|---|
| [Batch 03](PHASE2_BATCH_03_REPORT.md) | With the residual stream bounded, **T = 3 is solved exactly** at K = 4 — 100 % exact match in two of three cells — and Batch 02's K = 8 failure is attributed to the residual confound, not to recursion. |
| [Batch 04](PHASE2_BATCH_04_REPORT.md) | With the fix **and** the curriculum, **T = 4 — a primary benchmark depth — is learnable** at K = 4: 87.74 % token accuracy and 12.89 % exact match. **Batch 01 §9's exclusion of the residual confound at T = 4 is withdrawn.** |

So the central claim §10 was built on — that the depth axis could not be made to
move — is false. The project has a positive result at a primary depth.

**What is superseded, specifically:**

* §10's working title and §1's framing ("a recursive-depth hypothesis that its own
  benchmarks could not test").
* §4 "Where learnability stops", which was pending Batch 02 and is now answered
  differently than it anticipated: T = 3 is solved and T = 4 moves.
* §5's claim that four explanations are excluded. The residual confound is **not**
  excluded at T = 4; that exclusion is withdrawn.
* §6's "what is left" — the objective-starvation reading survives as an
  explanation of the *un-fixed* behaviour, but it is no longer the whole story.
* §7's "what would answer the original question", which now has partial answers.
* §9's point 5, that the infeasibility reading had become available.

**What survives.** §2 (two benchmarks, two distinct failure modes), §3 (the A₅
benchmark and the shortcuts at T = 2 and T = 3, now with T = 3's G2 failure
measured), and §8 (the method and integrity record), which is strengthened rather
than weakened: three classification defects were caught before they flattered a
result, two predictions of mine were falsified and recorded, and a reboot
mid-batch was recovered with its provenance disclosed.

**Not rewritten yet, deliberately.** [Batch 05](PHASE2_BATCH_05_PREREG.md) is
running and tests whether the K = 4 result is depth or compute — a question whose
answer changes the shape of any write-up. [Batch 06](PHASE2_BATCH_06_DRAFT.md)
would establish whether T = 4's 12.89 % is a plateau or a budget limit. Rewriting
before those report would mean rewriting twice, and the second rewrite would be
the one that mattered.

---

## 10. Outline of the path-3 write-up — DRAFT, NOT FINALISED

The Stage 3 action for the "all arms NOT LEARNABLE" row is to *draft, and not
finalise, the path-3 write-up outline*. This is that outline: a plan, with each
section's claim, the evidence for it, and the claim it must not make.

**Blocking dependency.** §§4 and 7 below cannot be written until the Batch 02
T = 3 probe reports (§8.8). Under the pre-registered outcomes
([PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md) §6), a NOT LEARNABLE or
DEGENERATE verdict makes the boundary claim "T = 2" and this write-up proceeds;
a LEARNABLE verdict attributed to composition routes to an amendment draft
instead and this outline is superseded; a LEARNABLE verdict attributed to the
shortcut keeps the boundary at T = 2 and adds a finding. **Do not write §4 to
either answer in advance.**

### Working title

*A recursive-depth hypothesis that its own benchmarks could not test: a negative
result over A₅ cellular automata.*

### §1 — What was asked, and why the answer is "not at reachable scale"

Claim: whether shared-weight recursion buys computational depth without
parameters is a well-posed question that this project could not answer, and the
reason is locatable rather than diffuse.

Evidence: §1, §9. Frame the result as a statement about the *benchmark*, not
about the architecture. The architecture hypothesis is untested, not refuted —
this distinction is the spine of the paper and every later section defers to it.

Must not claim: that recursion does not help.

### §2 — Two benchmarks, two distinct failure modes

Claim: the failures are informative because they differ. Rule 90 flattened the
depth axis algebraically (F^T collapses); A₅ supplies no low-order gradient.

Evidence: §2, §3, §4, and the P1c partial-credit ceilings.

This is the section that justifies the whole exercise: a benchmark can fail to
be a test, and the two ways it happened here are worth reporting on their own.

### §3 — What the A₅ benchmark is, and the two shortcuts inside it

Claim: the benchmark is well-gated at its primary depths and its known
weaknesses are documented rather than discovered late.

Evidence: G1 passes at every T tested. G2 passes at T ∈ {4, 8} and **fails at
T ∈ {2, 3}**, by one mechanism: a shallow product agrees with F^T exactly when a
local product squares to the identity, and 16 of A₅'s 60 elements do, giving
26.67 % at both depths (Amendment 01; Batch 02 prereg §2).

Reporting the T = 3 failure *and the fact that it was found by running the gate
before the probe rather than after* is part of the claim. Both are
Amendment-01-style documented failures, not surprises.

Must not claim: that T ∈ {2, 3} are usable primary depths.

### §4 — Where learnability stops — PENDING BATCH 02

Claim (conditional on §8.8): depth-2 composition over A₅ is learnable at this
width and depth-4 is not, with T = 3 the measured point between them.

Evidence: Arm 3's per-stage results (§8.4), Batch 02's stage-boundary
measurements at T = 1, 2, 3 (§8.8), Probe A at T = 4, Probe B at T = 8.

Constraints, both pre-registered in advance of the data: a T = 3 accuracy is
reported only with its shortcut attribution, and **three measured depths do not
establish the shape of a curve** — the "cliff" framing is withdrawn
([erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) §2) and must not return.

### §5 — Four explanations, tested one at a time and excluded

Claim: the T = 4 plateau is not any of capacity-at-depth, the residual-stream
confound, a cold start, or the optimiser schedule.

Evidence: Arm 1 (§8.1), Arms 2b and 2c (§8.2, §8.3), Arm 3 (§8.4), Arm 4 (§8.5).
Each arm changed exactly one thing against a common reference, and every arm was
pre-registered before it ran.

Carry every caveat: Arm 1's fit is a lookup-table result about the *optimiser*;
Arm 2b tests read-out normalisation only; only Arm 2c bounds the stream, and its
bound is verified post-training at 1.0000×.

Must not claim: that the list is exhaustive. §6 exists because it is not.

### §6 — What is left, and why it is the interesting part

Claim: the surviving explanation is that the objective supplies almost no
gradient toward partial knowledge at depth 4, so uniformity is a local optimum.

Evidence: the P1c ceilings (26.7 % / 13.4 % / 7.0 % for windows short by one,
two, three cells) together with the observed *erasure* of the T = 1 and T = 2
solutions during the T = 4 stage ([erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md)
§3). The erasure is the striking datum: the curriculum does not merely fail to
build on what it learned, it moves away from it.

State plainly that this is an interpretation, and state its falsifiable
prediction: erasure should be weaker at a depth where partial credit is worth
more. Batch 02's stage-boundary records test that prediction at T = 3.

Must not claim: representational impossibility. Nothing here distinguishes "the
function is not representable at this width" from "the objective does not lead
to it", and the write-up should say so in those words.

### §7 — What would answer the original question — PENDING BATCH 02

Claim: the question is answerable, by benchmarks this project did not have.

Content: a group ladder (a smaller or solvable group before A₅, to locate the
scale at which the task becomes learnable at all) is the option
[ARM5_CRITERIA](PHASE2_BATCH_01_ARM5_CRITERIA.md) §4 records and this project
did not take. An objective with denser partial credit is the other. Either would
give the K axis a regime in which it could show an effect.

Must not claim: that these would succeed.

### §8 — Method and integrity record

Claim: the negative result is trustworthy because of how it was recorded, and
this is worth a section rather than an appendix.

Content, all of it already on disk: every arm pre-registered before it ran; two
classification defects found and closed in the strict direction *before* the
arms they would have flattered reported (Addendum 03's majority-class floor and
collapse check; the Batch 02 shortcut check); a mis-fix identified and corrected
at the right place (§6, §7); deviations recorded rather than retrofitted (§8.4);
claims withdrawn by dated erratum rather than by editing history
([erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md)); a decision rule for the next
step committed before the results it decides on.

This section should note what the record cost: two of the batch's four PARTIALs
were artifacts that the original rule would have reported as movement, and the
Arm 5 trigger would have fired on them.

### Appendices

A — the A₅ automaton, the tasks, and the gate results at every T including T = 3.
B — full per-cell tables for every probe and arm, original and amended
classification side by side.
C — compute: 9.31 GPU-h for Batch 01, ≈ 2.72 for Batch 02, on one RTX 3050.
D — what is frozen, and the errata against it.

### Sections deliberately absent

No scaling-law section: two widths is not a scaling study. No ablation of the
dual-state update: never varied. No comparison against a non-recursive baseline
at matched depth: the baselines exist in the code but were never run at T = 4,
and saying so is better than implying coverage.
