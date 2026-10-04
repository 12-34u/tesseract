# Phase 2 — Batch 07 DRAFT (T = 4): is K = 8's result depth or compute?

**Date: 2026-10-04. DRAFT — not pre-registered, not approved, not run.**
**Estimated ≈ 13.1 GPU-h in the recommended form, over the 8-hour threshold.**

Replaces [PHASE2_BATCH_07_DRAFT.md](PHASE2_BATCH_07_DRAFT.md), the T = 3 version,
which is left committed and **marked superseded** — its question (arms B and C to
convergence at T = 3) is no longer the priority now that
[Batch 06](PHASE2_BATCH_06_REPORT.md) has solved T = 4, a *primary* depth, at
K = 8. §9 records what the T = 3 draft was for and why it is still worth running
later.

Nothing here is binding.

## 1. The question

Batch 06 found T = 4 solved at K = 8 in 3 of 3 seeds — 98.78 / 90.43 / 84.13 %
exact match — while K = 4 collapsed and K = 1 produced no correct rings. **Steps
were matched; compute was not.** At 90,000 steps K = 8 performs eight times
K = 1's block executions, so the result is consistent with two claims and only one
is the project's hypothesis:

1. **Depth** — applying one block eight times composes something a single
   application cannot, at no parameter cost.
2. **Compute** — eight times the forward FLOPs is what did it, and recursion is
   one way to spend them rather than a better one.

Batch 05 asked this at T = 3 and found the shared arm far ahead at a fixed budget,
but could not reach a verdict because the alternatives had not converged. **At
T = 4 the question matters more**, because T = 4 is a primary benchmark depth and a
positive answer bears on the depth hypothesis proper.

## 2. The three arms

Measured from the instantiated models at medium width (d_model 256, 8 heads,
n = 17), not computed by hand:

| arm | family | parameters | forward FLOPs / sequence | block calls | d_ff |
|---|---|---|---|---|---|
| **A** shared recursion | `tesseract`, K = 8 | **837,436** | 215,615,488 | 8 | 1024 |
| **B** width-scaled | `width_scaled`, matched to K = 8 | 6,383,992 | 215,624,192 | 1 | **11,836** |
| **C** unrolled | `unrolled`, L = 8 | 6,365,756 | 215,615,488 | 8 | 1024 |

* Forward compute matches to within **0.004 %** (arm C exactly; arm B is +8,704
  FLOPs on 215.6 M, the granularity of the integer `d_ff` search).
* **Arm A uses 7.62× fewer parameters than B and 7.60× fewer than C** — twice the
  ratio Batch 05 had at K = 4, because the shared block's count is K-invariant
  while the alternatives' grows with depth.

That ratio is the point. If arm A matches or beats both at equal compute, the
parameter invariance is buying something at 7.6× the density.

### 2.1 Arm A: reuse Batch 06's cells, or rerun?

Batch 06's three K = 8 cells are **bit-identically configured** to what arm A
would be: `tesseract` K = 8, medium width, the fix on, ladder
[1,2,3,4]@[5000,5000,15000,5000] plus a 60,000-step target stage, seeds 0/1/2,
AdamW 1e-3 weight decay 0.01, batch 128, eval every 250. If Batch 07 uses
Batch 06's ladder and budget — as it should, to be comparable — arm A *is* those
cells.

**The case for reuse.** The configuration is identical, and reproducibility has
been demonstrated rather than assumed: Batch 05's arm A reproduced Batch 03's
K = 4 cells to four decimal places across runs days apart, and its T = 3 rung-end
values matched Batch 04's exactly. Reuse saves **≈ 7.4 GPU-h**, which is more than
half the recommended budget. The checkpoints and full curves are on disk.

**The case against.** Batch 03's resume showed what mixing provenance costs: a
report with some cells trained in-batch and others carried in needs per-cell
source tags, and a reader has to track which comparison crosses a batch boundary.
Arms B and C would also run on a different day, under a different GPU thermal
history — and Batch 06's own +23.4 % evaluation overhead against Batch 04's
+5.2 % is probably thermal (§7), which is evidence that the machine's state does
vary across long runs even if the arithmetic is deterministic.

**Recommendation: reuse, plus one fresh arm A cell as a reproduction check.**
Seed 0 rerun costs ~2.0 GPU-h and either confirms the environment has not drifted
(its exact match should land on 98.7793 %) or reveals that it has, which would be
worth knowing before any cross-batch comparison is drawn. Every cell carries a
`source` tag of `reused_from_batch06` or `trained`, as Batch 03's resume did.

## 3. Configuration (proposed)

| | value | source |
|---|---|---|
| Depth | T = 4 | |
| Curriculum ladder | T = 1 (5,000) → T = 2 (5,000) → T = 3 (15,000) → T = 4 (5,000) | **Batch 06's, unchanged** |
| Target stage | T = 4, 60,000 steps | **Batch 06's, unchanged** |
| Total per cell | **90,000** | Batch 06's |
| Residual fix | non-affine per-step normalisation, all arms | Batch 05 established it for `unrolled` |
| Optimiser | AdamW, lr 1e-3, weight decay 0.01 | inherited (§5) |
| Width / batch / eval | medium / 128 / every 250 | |
| **Primary metric** | **exact match** | Batch 03 Note 01 |
| Seeds | 3 per arm | |
| Plateau test | **§4**, corrected | Batch 06 §3 |

The ladder and budget are Batch 06's **unchanged**, which is what makes arm A
reusable and the comparison interpretable. Batch 06 measured T = 4 takeoff needing
12,750–30,250 target-stage steps, so 60,000 is adequate for K = 8; whether it is
adequate for arms B and C is unknown, which is why §4's "still rising" row and
§6's not-at-convergence row both matter.

## 4. The corrected plateau test, argued

Batch 06 §3 showed the pre-registered test — final slope < 0.1 points per 1,000
steps over the last 2,500 — misclassifying all three K = 8 cells as "still rising"
when their accuracy was oscillating without trend near the ceiling. Four
corrections were recorded there. **This draft adopts a conjunction of the original
threshold with a significance test**, and the argument is that the original failed
on exactly one axis.

> **A cell is still rising iff its fitted slope over the last 2,500 steps is
> ≥ 0.1 points per 1,000 steps AND statistically distinguishable from zero at
> t > 2** (slope greater than twice its standard error).

**Why a conjunction rather than replacing the threshold.** The original threshold
encodes *practical* significance — a drift of 0.03 points per 1,000 steps is not
worth calling progress whatever its t-statistic. What it lacked was *statistical*
significance: near the ceiling, per-evaluation noise is 0.36–0.61 points and the
threshold asks an 11-point fit to resolve 0.25 points across the window. Adding
the t-test fixes that axis and nothing else.

**Why not the other three forms.** A headroom-relative threshold moves the wrong
way — at 99.93 % accuracy the headroom is 0.07 points, so any threshold scaled to
it becomes *stricter* and the misclassification worsens. A longer window changes
the quantity being estimated, since it reaches back into genuinely-rising steps. A
test on exact match faces the same noise problem in a steeper metric.

**Verified against every cell of both batches** — 18 cells, and the conjunction
classifies all of them as the trajectories warrant:

| | slope /1k | t | original test | **corrected** |
|---|---|---|---|---|
| B06 K = 8 s0 | +0.322 | 1.78 | still rising ✗ | **plateaued ✓** |
| B06 K = 8 s1 | +0.121 | 0.88 | still rising ✗ | **plateaued ✓** |
| B06 K = 8 s2 | +0.148 | 0.62 | still rising ✗ | **plateaued ✓** |
| B06 K = 1 s1 | +0.037 | 2.31 | plateaued | plateaued ✓ (below the magnitude floor) |
| B06 K = 4 s1 | +0.036 | 2.13 | plateaued | plateaued ✓ (below the magnitude floor) |
| B06 K = 1 s2 | +0.403 | 12.51 | still rising | **still rising ✓** |
| B05 arm B s1 | +1.884 | 25.49 | still rising | **still rising ✓** |
| B05 arm C s0 | +6.701 | 23.28 | still rising | **still rising ✓** |
| B05 arm C s1 | +17.694 | 15.12 | still rising | **still rising ✓** |
| B05 arm A s0–s2 | −0.045 to −0.210 | < 0 | plateaued | plateaued ✓ |

Note the two cells where the t-test alone would have erred — K = 1 s1 and K = 4 s1
at t > 2 on slopes of 0.036–0.037 — which is why the magnitude floor stays.

**This is a change to a pre-registered rule and must be recorded as one.** It is
adopted **before** Batch 07 runs, it is argued from Batch 06 and Batch 05 data
rather than from Batch 07's, and it is strictly a *refinement*: it can only
reclassify a noise-driven "rising" as "plateaued", never the reverse, since it
adds a conjunct. Batch 06's reported outcome stands as published.

## 5. The learning-rate check is dropped, and why

The T = 3 draft included lr ∈ {3e-4, 1e-3} for arms B and C.
**At T = 4 the budget does not allow it, and it is dropped.**

A T = 4 cell is 90,000 steps against T = 3's 35,000, and arms B and C are
7.6× larger, so each cell costs roughly 2.6–3.1× its T = 3 equivalent. Adding a
second rate doubles arms B and C — the whole substance of the batch — taking the
recommended 13.1 GPU-h to ≈ 26.1 (§7). That is not a close call.

**What is lost.** [Batch 05 Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md) records
that 1e-3 was inherited through five batches from a shared-block model at 837,436
parameters, and arms B and C here hold **7.6×** that — a larger mismatch than the
3.8× the note was written about. So "arm A ahead at equal compute" will carry the
same qualifier it does at T = 3: *at a learning rate inherited from arm A's
family*.

**Why that is acceptable here.** The rate question is answered more cheaply at
T = 3, where a cell is a third the cost, and the superseded
[T = 3 draft](PHASE2_BATCH_07_DRAFT.md) is exactly that experiment. **Recommended
sequencing: run this batch at the inherited rate to get the T = 4 answer, and run
the T = 3 rate check separately** — if the rate turns out to matter at T = 3, this
batch's conclusion is qualified accordingly and would need revisiting.

Stating it plainly: **this draft buys the primary-depth answer at the cost of
leaving the rate confound in place.** The alternative — a rate check at T = 4 —
costs 26 GPU-h and answers the same question about a confound that T = 3 can
probe for 4.

## 6. Outcome table (proposed, not in force)

Keyed on **exact match**, with the §4 plateau test, the 1 % floor, and the
not-at-convergence row Batch 05 showed every table needs.

| pattern | reading |
|---|---|
| **A ≈ C, both > B** (A within 5 points of C, both exceeding B by > 5) | **Depth, not compute.** Eight block executions beat one wide layer at equal compute, and sharing costs nothing — the project's claim at a primary depth, with arm A at 7.6× fewer parameters. |
| **A ahead of both B and C** (> 5 points over each) | **Depth, and sharing helps.** Stronger than the hypothesis predicts. This row exists because Batch 05's table lacked it and returned UNRESOLVED when the pattern occurred — it is not a prediction, it is a gap being closed in advance. |
| **A ≈ B ≈ C** | **Compute, not depth.** Batch 06's K = 8 result was the eight-fold FLOPs. |
| **C > A** (> 5 points) | **Depth needs unshared parameters.** Recursion captures part of the benefit; the parameter invariance costs accuracy. |
| **B > A** (> 5 points) | **Width beats depth at equal compute.** A negative result for the architecture, reported as one. |
| **Any arm still rising** under §4 | Reported as still rising; its accuracy is **not** an achieved level and no comparison involving it is drawn. |
| **Not all arms at convergence** | Per-arm results with slopes and t-statistics; **no cross-arm ordering claimed.** |
| **Arm A fails to reproduce 98.7793 %** (the §2.1 check) | The environment has drifted. **Batch failure requiring investigation**, and no cross-batch reuse. |
| Any arm DEGENERATE, or below the exact-match floor | That arm learned nothing usable; excluded from the comparison with the reason recorded. |
| Bounded-growth gate fails, or any cell's post-training growth ≥ 1.05 | That arm is not a fix-on arm; excluded. |

The "differ by more than 5 points" threshold is Batch 05's, with its argument
(§5 of that pre-registration) and the escalation clause: if every arm lands in the
5–95 % band the comparison escalates rather than being adjudicated.

## 7. Compute

**Per-step times for arms B and C at these shapes are not measured.** Measuring
them needs the GPU, which is reserved, so the figures below are **extrapolated**
from Batch 05's T = 3 ratios (arm B 0.726× arm A, arm C 1.036× arm A) applied to
arm A's measured 0.08008 s/step at K = 8, T = 4. Arm B's ratio is the less
trustworthy: it doubles d_ff again to 11,836, and a wider matmul may utilise the
GPU better than the ratio implies, so its cost may be *over*-estimated.

**Measure before pre-registering.** Two minutes of GPU once it is free.

| arm | cells | s/step | basis | total |
|---|---|---|---|---|
| A (1 reproduction cell) | 1 | 0.08008 | **measured** | 2.00 h |
| B, lr 1e-3 | 3 | ~0.0581 | extrapolated | 4.36 h |
| C, lr 1e-3 | 3 | ~0.0830 | extrapolated | 6.22 h |
| | **7** | | | **12.58 h training** |

**Evaluation overhead scaled to step count**, as the task requires. The measured
outturn:

| batch | steps/cell | predicted training | actual | overhead |
|---|---|---|---|---|
| Batch 04 | 40,000 | 4.40 GPU-h | 4.63 | **+5.2 %** |
| Batch 06 | 90,000 | 9.90 GPU-h | 12.22 | **+23.4 %** |

A fixed eval interval should give a *constant* fractional overhead, so the rise
from 5.2 % to 23.4 % is not evaluation count. The likely cause is **sustained-load
thermal throttling**: Batch 06 ran 12 h 13 m against Batch 04's 4 h 38 m on a
laptop GPU. So the correction is not a formula but a regime: **use +23.4 % for any
run of this length.**

| | GPU-h |
|---|---|
| Training, 7 cells × 90,000 steps | 12.58 |
| Overhead at the 90,000-step/long-run rate (+23.4 %) | 2.94 |
| Bounded-growth gate | ~0.02 |
| **Estimated total** | **≈ 15.5 GPU-h** |

That is higher than the 13.1 in the header because the +23.4 % regime is the
honest figure rather than Batch 04's +5.2 %. **Costed reductions:**

| option | cells | ≈ GPU-h | cost |
|---|---|---|---|
| **Full**: arm A rerun ×3 + B, C at both rates | 15 | **~33** | — |
| B and C at both rates, arm A reused | 12 | **~26** | — |
| **Recommended**: B, C at 1e-3, arm A reused + 1 check cell | 7 | **~15.5** | rate confound left in place (§5) |
| Drop the arm A reproduction cell | 6 | **~13.1** | no check that the environment has not drifted before a cross-batch comparison |
| Arm B only (drop arm C) | 4 | **~8.3** | forfeits the weight-sharing question entirely — the comparison Batch 05 could not make |
| 2 seeds for B and C | 5 | **~11.1** | breaks the 2-of-3 classification rule. **Not recommended** |

**Recommendation: approve ≈ 15.5 GPU-h.** The 13.1 option saves 2.4 GPU-h by
removing the one cell that would detect environment drift before the whole batch
rests on a cross-batch reuse, which is a poor trade. Dropping arm C would make
this a narrower experiment than Batch 05 already was.

Disk: arms B and C hold ~6.4 M parameters, so ~25.6 MB per checkpoint; 6 cells × 2
≈ 307 MB, plus arm A's check cell and curves (360 points per cell) ≈ 20 MB, so
**≈ 330 MB**.

## 8. Recorded fields

Batch 06's, plus the two the project has been missing and one new:

* **the resolved optimiser configuration per cell** — optimiser, learning rate,
  weight decay, batch size. [Batch 05 Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md)
  §4 records that no report has ever held these. Required here even though the
  rate is uniform, because §5 leaves it as a named confound and a reader must be
  able to check it from the artifact;
* **the final slope, its standard error and its t-statistic** per cell, for §4;
* **a `source` tag per cell** — `trained` or `reused_from_batch06` (§2.1);
* **parameter count, forward FLOPs and wall-clock seconds per step** per cell, as
  Batch 05 recorded, since equal FLOPs is not equal time.

Partial report written after every cell.

## 9. The superseded T = 3 draft

[PHASE2_BATCH_07_DRAFT.md](PHASE2_BATCH_07_DRAFT.md) proposed running Batch 05's
arms B and C to convergence at T = 3 with a learning-rate check. It is superseded
as the *next* batch, not abandoned: it is the cheap way to answer the rate
question §5 drops, and it would settle whether Batch 05's A-vs-B-and-C gap is
convergence or capability. At ~10.8 GPU-h for 13 cells it is comparable in cost to
this batch and answers a different question.

**Suggested order:** this batch first, because T = 4 is a primary depth and the
depth-versus-compute question there is the one that bears on the project's
hypothesis; then the T = 3 rate check, whose result would qualify or confirm this
batch's conclusion.

## 10. What this batch would still not establish

* **Nothing about the learning rate** (§5). The conclusion carries that qualifier.
* **Nothing about total training compute.** Forward FLOPs are matched; arms B and
  C hold 7.6× the parameters, so their backward pass and optimiser state cost
  far more. A reader wanting total compute matched does not get it.
* **Nothing about why K = 4 collapsed** in Batch 06
  ([Note 01](PHASE2_BATCH_06_REPORT_NOTE_01.md)); K = 4 is not an arm here.
* **Nothing about T = 8**, the other primary depth.
* **No takeoff probability.** Three seeds per arm, and this project has repeatedly
  seen cells at identical settings be bimodal.
* **Nothing about the grid**, which trains at the target depth from scratch with no
  curriculum.

## 11. Open questions

1. **Approve ≈ 15.5 GPU-h**, or take the 13.1 option (no arm A reproduction cell)?
2. **Reuse arm A from Batch 06, or rerun all three cells?** §2.1 recommends reuse
   plus one check cell; rerunning all three costs ~4 GPU-h more and removes the
   cross-batch question entirely.
3. Is the §4 corrected plateau test acceptable as a pre-registered change, given
   it is argued from prior batches and can only reclassify in one direction?
4. Measure arms B and C per-step before pre-registering — agreed? §7's totals move
   if arm B is cheaper than extrapolated.

Nothing in this file is in force.
