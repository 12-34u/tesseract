# Phase 2 — Batch 07 DRAFT: arms B and C to convergence, with a learning-rate check

**Date: 2026-10-03. DRAFT — not pre-registered, not approved, not run.**
**Estimated at ≈ 10.8 GPU-h, over the 8-hour threshold; needs approval.**

[Batch 05](PHASE2_BATCH_05_REPORT.md) found the shared recursion (arm A) solving
T = 3 exactly while the compute-matched alternatives produced almost no correct
rings — but its comparison returned `UNRESOLVED_ESCALATE`, for two reasons
recorded in its §3 and one recorded in its
[Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md):

| | |
|---|---|
| **Not at convergence** | arm A had plateaued (−0.090 /1k) while arm C was still climbing at **+17.694 /1k**. Comparing them measures the budget as much as the arm. |
| **No A > C row** | my outcome table did not contemplate the shared arm beating the unrolled one. |
| **Learning rate inherited from arm A** | all arms ran AdamW 1e-3, traced back through five batches to a rate chosen for a shared-block model at 837,436 parameters. Arms B and C hold 3.8× that. |

Batch 07 removes the first and third. Nothing here is binding.

## 1. The question

**Given enough steps, and a learning rate that is not simply arm A's, do the
compute-matched alternatives reach the solution arm A reaches?**

Two outcomes are worth distinguishing and Batch 05 cannot:

1. **Arm A is faster.** The alternatives converge too, given steps or a better
   rate, and the shared recursion's advantage is step efficiency — real and
   useful, but not a capability claim.
2. **Arm A is alone.** The alternatives plateau below it however long they run
   and whichever of two rates they use, and the advantage is a capability
   difference at 3.8× fewer parameters.

## 2. The arms

| arm | family | lr | seeds | cells | role |
|---|---|---|---|---|---|
| **A** | `tesseract` K = 4 | 1e-3 | 1 | 1 | reference at the longer budget |
| **B** | `width_scaled` | **3e-4** | 3 | 3 | |
| **B** | `width_scaled` | 1e-3 | 3 | 3 | Batch 05's setting, at the longer budget |
| **C** | `unrolled` L = 4 | **3e-4** | 3 | 3 | |
| **C** | `unrolled` L = 4 | 1e-3 | 3 | 3 | Batch 05's setting, at the longer budget |
| | | | | **13** | |

**Why arm A gets one seed.** It converged at 35,000 steps in all three Batch 05
seeds (100.00 / 99.90 / 100.00 % exact match, negative slopes). Re-running it is
insurance that it stays plateaued over 75,000 steps rather than degrading, which
one seed establishes; its three-seed result is already in Batch 05 at the
identical configuration. Arm A is **not** given 3e-4: 1e-3 is its tuned rate and
varying it would be a different experiment.

**Why 3e-4.** A single additional point, roughly a third of 1e-3 — the
conventional direction and magnitude for a 3.8× larger model. This is a *check*,
not a sweep: two points per arm can show that the Batch 05 result is not an
artefact of one rate, and cannot find an optimum. If 3e-4 beats 1e-3 markedly for
either arm, the honest conclusion is "a rate search is needed", not "3e-4 is
right".

## 3. Budget — derived from Batch 05's slopes

Batch 05's final slopes, over the last 2,500 of 35,000 steps, and the
constant-slope steps each rising cell would need to reach ~99 % token accuracy:

| cell | token accuracy | slope | steps to ~99 % at that slope |
|---|---|---|---|
| A s0–s2 | ~100 % | −0.045 to −0.210 /1k | converged |
| B s1 | 70.41 % | +1.884 /1k | **15,180** |
| C s0 | 21.68 % | +6.701 /1k | **11,540** |
| C s1 | 82.84 % | +17.694 /1k | 914 |
| B s0, B s2, C s2 | ~1.8 % | ~0 | never took off |

The binding cell is **arm B seed 1 at 15,180 further steps**, and that figure is a
**lower bound**: slopes flatten near the ceiling, so reaching a *plateau* takes
longer than reaching the accuracy. Arm A offers the one empirical ratio
available — it crossed the shortcut ceiling ~750 steps into its target stage and
had plateaued by the stage's end, about 19,000 steps later, i.e. roughly **25×**
the time-to-cross.

Taking the binding cell's 15,180 and allowing comparable room to flatten:

> **Target stage 60,000 steps** (against Batch 05's 20,000), with Batch 05's
> ladder unchanged at T = 1 → 2 → 3, 5,000 steps per rung. **Total 75,000 steps
> per cell.**

That gives arm B seed 1 roughly **4×** its constant-slope requirement and arm C
seed 0 roughly **5×**. It is not a guarantee of convergence — nothing fixed is —
which is why §5 keeps a "still rising" row.

**The ladder is not lengthened.** Batch 05 measured arms B and C at chance at the
end of the 5,000-step T = 3 rung, and that fixed-point comparison is the result
its §4 rests on. Changing the rung would discard the one measurement that does
not depend on convergence. The question here is what happens *after* the rung.

## 4. Configuration (proposed)

| | value | against Batch 05 |
|---|---|---|
| Depth | T = 3 | same |
| Curriculum | T = 1 → 2 → 3, 5,000 steps per rung | **same, deliberately** (§3) |
| Target stage | **60,000 steps** | 20,000 → **60,000** |
| Total per cell | **75,000** | 35,000 → **75,000** |
| Learning rate | **3e-4 and 1e-3 for B and C; 1e-3 for A** | 1e-3 everywhere |
| Optimiser | AdamW, weight decay 0.01 | same |
| Residual fix | non-affine per-step normalisation, all arms | same |
| Width / batch / eval interval | medium / 128 / every 250 | same |
| Primary metric | **exact match** | same |
| Seeds | 3 for B and C; 1 for A | 3 everywhere |
| Final slope per cell | over the last 2,500 steps | **new, for the plateau test** |

**One lever is varied per cell, but two across the batch** (steps and learning
rate). B and C at 1e-3 isolate the budget against Batch 05; B and C at 3e-4 add
the rate. So budget and rate are separable here, unlike Batch 06 where two stage
lengths move together. That is worth having and is the reason for 13 cells
rather than 7.

## 5. Outcome table (proposed, not in force)

Keyed on **exact match**, with Batch 06's plateau test (final slope < 0.1 /1k)
and the not-at-convergence row.

| pattern | reading |
|---|---|
| **B and C both plateau at ≈ arm A** (within 5 points of exact match) | **Arm A is faster, not better.** Batch 05's result is step efficiency at a given rate; the compute-matched alternatives reach the same solution with more steps. The depth claim narrows to efficiency, and the parameter saving stands on its own. |
| **B and C both plateau far below arm A** (> 5 points) at **both** rates | **Arm A is alone.** At matched compute, neither width nor unshared parameters reaches the shared recursion's solution however long they run, at either rate. The strongest form of the project's claim. |
| **One of B, C plateaus at ≈ A, the other does not** | Partial. Report per arm; the A-vs-that-arm comparison resolves and the other does not. |
| **3e-4 markedly better than 1e-3 for either arm** (> 5 points of exact match) | **The Batch 05 comparison was rate-confounded.** Report that, and that a proper rate search — not this two-point check — is the prerequisite for any arm comparison. §2's caveat applies: this cannot find an optimum. |
| **Any arm still rising at 75,000** | Reported as still rising; its accuracy is **not** quoted as an achieved level, and no comparison involving it is drawn. |
| **Not all arms at convergence** | Per-arm results reported with slopes; **no cross-arm ordering claimed.** The row Batch 05 showed every table needs. |
| **Arm A fails to stay plateaued** | Batch failure requiring investigation, not a finding. |
| Bounded-growth gate fails, or any cell's post-training growth ≥ 1.05 | That arm is not a fix-on arm; excluded from every cross-arm conclusion. |

## 6. Recorded fields

Batch 05's, plus two the project has been missing:

* **the resolved optimiser configuration per cell** — optimiser, learning rate,
  weight decay, batch size. [Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md) §4
  records that no batch report has ever held these, so a reader cannot check from
  the artifact what rate produced a number. This batch *varies* the rate across
  cells, so the artifact would be unreadable without it;
* **the final slope per cell**, for the plateau test.

Partial report written after every cell.

## 7. Compute

From Batch 05's **in-run** measured times (not a separate timing probe): arm A
0.04195, arm B 0.03053, arm C 0.04280 s/step. Learning rate does not change
per-step cost.

| arm | cells | s/step | total |
|---|---|---|---|
| A, 1e-3 | 1 | 0.04195 | 0.87 h |
| B, both rates | 6 | 0.03053 | 3.82 h |
| C, both rates | 6 | 0.04280 | 5.35 h |
| | **13** | | **10.04 GPU-h** |

| | GPU-h |
|---|---|
| Training, 13 cells × 75,000 steps | 10.04 |
| Evaluation overhead (+7.6 %) | 0.76 |
| Bounded-growth gate | ~0.02 |
| **Estimated total** | **≈ 10.82 GPU-h** |

**Over the 8-hour threshold.** Reductions, costed:

1. **Drop the 1e-3 repeats for B and C** (7 cells: A + B/C at 3e-4 only) →
   ≈ **5.9 GPU-h**, under threshold. But then the budget and the rate change
   together and neither is isolated — the Batch 06 defect, reintroduced. **Not
   recommended.**
2. **Target stage 40,000** (55,000 total) → ≈ **7.9 GPU-h**, under threshold.
   Gives arm B seed 1 ~2.6× its constant-slope requirement instead of ~4×. The
   cheapest honest option.
3. **Two seeds for B and C** (9 cells) → ≈ **7.6 GPU-h**, but the classification
   rule needs 2 of 3. **Not recommended.**

**Recommendation: approve ≈ 10.8 GPU-h, with option 2 as the fallback.** Option 2
risks a "still rising" verdict, which would answer nothing and cost 7.9 GPU-h.

Disk: arms B and C checkpoints are ~12.8 MB; 13 cells × 2 ≈ **320 MB**, plus
curves (300 points per cell) ≈ 15 MB.

## 8. What this batch would still not establish

* **No learning-rate optimum.** Two points per arm (§2). If 3e-4 wins, the
  conclusion is that a search is needed.
* **Nothing about total training compute.** Forward FLOPs are matched; arms B
  and C hold 3.8× the parameters, so their backward pass and optimiser state
  cost more.
* **Nothing at T = 4 or T = 8.** T = 3 only, which fails G2.
* **Nothing about arm A at other rates.** It keeps 1e-3 by design (§2).
* **No takeoff probability.** Three seeds per arm-and-rate, and Batch 05's arms
  B and C were both bimodal.
* **Nothing about `param_matched`.**

## 9. Open questions

1. **Approve ≈ 10.8 GPU-h**, or take option 2 (40,000-step target stage,
   ≈ 7.9 GPU-h)?
2. Is **3e-4** the right second point, or would **1e-4** be more informative
   given the 3.8× parameter gap? A third rate costs ~3.6 GPU-h more.
3. Should arm A get a 3e-4 cell after all, as a control that the rate check is
   symmetric? One cell, ~0.9 GPU-h. It would weaken the objection that only the
   alternatives were given a choice of rate.
4. Batch 06 is running and will report before this could start. If it finds T = 4
   plateauing, is a T = 3 arm comparison still the priority, or does the
   grid-resume draft come first?

Nothing in this file is in force.
