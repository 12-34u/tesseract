# Phase 2 — Batch 06 pre-registration: T = 4 with Batch 04's two limits removed

**Date: 2026-10-03. Committed before the batch runs.**

Supersedes [PHASE2_BATCH_06_DRAFT.md](PHASE2_BATCH_06_DRAFT.md), which is left as
committed. The design is the draft's, with the four decisions recorded in its §7
on 2026-10-01 — **before Batch 05 reported** — applied here unchanged. §4's
outcome table additionally gains a row that [Batch 05](PHASE2_BATCH_05_REPORT.md)
showed was missing from every outcome table in this project; §4.1 explains.

Verified immediately before committing: `runs/phase2/batch06/` does not exist, no
`t4_long_report.json` exists anywhere, `systemctl --user` lists no
`tesseract-chain@06` unit, and `tesseract-chain@05` is complete and disabled.

**Approved at ≈ 10.7 GPU-h**, which exceeds the standing 8-hour threshold, on
2026-10-01.

## 1. The two limits, and what the data says each costs

### 1.1 The T = 3 rung starves K = 1

Batch 04 §6 measured T = 3 consolidation at the end of its 5,000-step rung:

| cell | T = 3 at rung end | final T = 4 |
|---|---|---|
| K = 1 s0 | **2.14 %** | 1.92 % (degenerate) |
| K = 4 s1 | 4.85 % | 1.66 % (no takeoff) |
| K = 4 s2 | 15.35 % | **87.74 %** |
| K = 4 s0 | 33.03 % | 32.60 % |

K = 4 extracts 15–33 % of T = 3 from 5,000 steps; K = 1 extracts 2.14 %. So the
K = 1 null at T = 4 is confounded with ladder length — it arrived without the
prerequisite depth — and the two K = 4 cells that learned T = 4 are the two that
had meaningful T = 3, while the one with 4.85 % did not.

**What length does K = 1 need?** From Batch 03's K = 1 cells at T = 3, whose
target stage ran steps 15,001–35,000:

| K = 1 seed | crossed 6.67 % at | crossed 26.63 % at | steps into the stage |
|---|---|---|---|
| 0 | 16,000 | 21,000 | **6,000** |
| 1 | 19,750 | 27,000 | **12,000** |
| 2 | 22,500 | 26,750 | **11,750** |

The worst seed needs **12,000 steps** to pass the 26.63 % shortcut ceiling at
T = 3. **A 15,000-step T = 3 rung** clears that in all three seeds with 3,000
steps of margin, and brings every K = 1 cell to at least 26.6 % — comparable to or
better than the 15–33 % the successful K = 4 cells had.

Two honest qualifications. K = 1 never *solves* T = 3: Batch 03 had it at 43–65 %
token accuracy and ~0 % exact match even after 20,000 steps. So a longer rung
gives K = 1 the same *partial* T = 3 that K = 4 had, which is what Batch 04 §6
suggests is the operative prerequisite — not a solved one. And the T = 3 rung at
15,000 steps is three times the others, which makes the ladder asymmetric in
depth; §1.3 addresses why that is the right asymmetry.

### 1.2 The target stage is too short at T = 4

Batch 04 §3: **every cell that took off was still rising at step 40,000.**

| cell | crossed 26.63 % at | steps into the target stage | slope at the end |
|---|---|---|---|
| K = 4 s0 | 32,750 | 12,750 | +0.732 /1k |
| K = 4 s2 | 30,500 | 10,500 | +3.251 /1k |
| K = 8 s0 | 31,750 | 11,750 | **+5.136 /1k** |

Compare T = 3, where K = 4 crossed the ceiling within ~750 steps of its target
stage starting and had *negative* slopes at 35,000 — converged, with 20,000 steps
ample. At T = 4 the ceiling crossing takes 10,500–12,750 steps and nothing has
plateaued after 20,000.

**What length is enough?** Taking the slowest observed cell: K = 8 seed 0 at
55.96 % rising at +5.136 points per 1,000 steps needs ~8,600 further steps to
approach 100 % at constant slope, and slopes flatten near the ceiling, so more.
A target stage of **60,000 steps** — three times Batch 04's — gives that cell
~48,000 steps past its ceiling crossing, against the ~20,000 that sufficed at
T = 3 for a converged result.

The purpose is not a bigger number. It is that **a plateau is interpretable and a
budget limit is not.** Batch 04 cannot say whether 12.89 % exact match is what
K = 4 achieves at T = 4 or merely where it was when the steps ran out, and no
amount of re-analysis fixes that.

### 1.3 Same rung lengths for every K

The rungs are **not** tuned per K. T = 1 and T = 2 reach 100 % within 5,000 steps
at every K in every batch measured (Batch 02, 03, 04), so 5,000 is ample there and
is kept. T = 3 goes to 15,000 because that is what the *slowest* arm needs, and
the faster arms simply get more than they need — which costs compute but not
validity. Tuning the ladder per K would make the K comparison meaningless, which
is precisely the error §1.1 identifies in Batch 04.

## 2. Configuration

| | value | against Batch 04 |
|---|---|---|
| Target depth | T = 4 | same |
| Curriculum ladder | T = 1 (5,000) → T = 2 (5,000) → **T = 3 (15,000)** → T = 4 (5,000) | T = 3 rung 5,000 → **15,000** |
| Target stage | T = 4, **60,000 steps** | 20,000 → **60,000** |
| **Total steps per cell** | **90,000** | 40,000 → **90,000** |
| Recursion K | {1, 4, 8} | same |
| Seeds | 0, 1, 2 | same |
| Width | medium | same |
| Residual fix | non-affine per-step normalisation | same |
| Optimiser | AdamW, constant 1e-3, weight decay 0.01 | same |
| Batch / eval interval | 128 / every 250 steps | same |
| Metrics | **exact match co-primary** with token accuracy, 1 % floor | same |
| Final slope, per cell | over the last 2,500 steps, for §4's plateau test | **new** |
| Classification | standing rule + Addendum 03 (T = 4 floor **1.9244 %**) | same |
| Shortcut attribution | at T = 2 and T = 3 only; T = 4 passes G2 | same |
| Partial report per cell | yes | same |

Boundaries fall at steps 5,000 / 10,000 / 25,000 / 30,000 / 90,000.

### 2.1 What stays fixed, and why that matters

Everything except the two stage lengths is Batch 04's. That is deliberate: it
makes Batch 06 a two-lever change against Batch 04, and the levers are the two
limits Batch 04 itself recorded. Keeping the bars, the floor, the fix, the width,
the seeds and the K range means a difference between the batches is attributable
to the stage lengths.

**It is still a two-lever change, not one.** Lengthening the T = 3 rung and the
target stage together means a positive result cannot be attributed to either
alone. That is accepted here because Batch 04 gives independent evidence for each
(§1.1, §1.2) and because running them separately doubles an already
over-threshold batch. It should be recorded as a limitation rather than
presented as a clean attribution.

## 3. Compute — over the threshold

Per-step times are **Batch 04's measurements**: same depth, same fix, same width,
same batch size, 120 timed steps after 20 warmup. They were not re-measured when
the draft was written because Batch 05 was training and a timing probe would have
contended with it. Batch 05's in-run figures since confirm the method — its arm A
measured 0.04195 s/step in-run against 0.04051 predicted, a 3.6 % gap that is the
evaluation time included in the per-cell wall clock — so these estimates should
be read as ±5 %.

| K | s/step (measured) | per cell (90,000 steps) | 3 seeds |
|---|---|---|---|
| 1 | 0.01144 | 17.2 min | 51.5 min |
| 4 | 0.04040 | 60.6 min | 181.8 min |
| 8 | 0.08008 | 120.1 min | 360.4 min |
| | | **training total** | **9.90 GPU-h** |

| | GPU-h |
|---|---|
| Training, 9 cells | 9.90 |
| Evaluation overhead (+7.6 %, measured from Batch 02/03 outturn) | 0.75 |
| Bounded-growth gate | ~0.02 |
| **Estimated total** | **≈ 10.67 GPU-h** |

**This exceeds the 8 GPU-h threshold and was approved explicitly on 2026-10-01**
(§6, decision 1). The three reductions below were costed and **not taken**; they
are kept for the record:

1. **Drop K = 8** (K ∈ {1, 4}) → ≈ **4.7 GPU-h**, under the threshold. Batch 04's
   K = 8 was PARTIAL with 0.0488 % exact match while K = 4 was LEARNABLE with
   12.89 %, so K = 4 is where the signal is; but dropping K = 8 forfeits the one
   cell with the steepest slope, which is exactly the cell §1.2 is about.
2. **Target stage of 40,000 instead of 60,000** (70,000 total) → ≈ **8.3 GPU-h**,
   still marginally over. Gives the slowest cell ~28,000 steps past its ceiling
   crossing rather than ~48,000.
3. **Two seeds instead of three** → ≈ **7.1 GPU-h**, under the threshold, but the
   classification rule requires 2 of 3 seeds and would have to change, which
   breaks comparability with every prior batch. **Not recommended.**

Decision 1 took the full 10.67 GPU-h.

Disk: 9 cells × 2 checkpoints × 3.20 MB ≈ 58 MB, plus curves ≈ 12 MB (90,000
steps at an eval every 250 gives 360 points per cell), so **≈ 70 MB**.

## 4. Pre-registered outcomes

Keyed on **exact match**, with the 1 % floor Batch 04 used. Decision 2 (no
adaptive stopping) means nothing in the protocol forces a plateau, so "reaches a
plateau" and "still rising" are **separate rows** rather than one.

| outcome at T = 4 | action |
|---|---|
| **Plateau** (final slope < 0.1 /1k over the last 2,500 steps) **with exact match ≥ 1 %** at some K | T = 4 is learnable to a measurable level and the level is reportable rather than budget-limited. This is the result the batch is for. Propose, **draft only**, resuming the grid with the fix and curriculum. |
| **Exact match ≥ 1 % at some K but still rising** (slope ≥ 0.1 /1k) | Learnable, still budget-limited even at 90,000 steps. Report the slope and state that explicitly; **do not quote the accuracy as an achieved level.** No grid proposal — the number is not yet a number. |
| **K = 1 now learnable where Batch 04 had it not** | Batch 04's K = 1 null was the ladder, not K = 1. Attribute to §1.1 and record that Batch 04's K = 1 row should be read that way. |
| **K = 1 still not learnable** with a 15,000-step T = 3 rung | The ladder was not the explanation; K = 1's T = 4 null stands with that confound removed. |
| **Exact match < 1 % at every K** | "Token accuracy only, no correct rings", as Batch 04 §4 reads it. Report with the token accuracies and slopes. |
| **Nothing learnable anywhere** | Contradicts Batch 04, which found K = 4 LEARNABLE on a shorter budget. **Batch failure requiring investigation**, not a finding. |
| **Not all K at convergence** — see §4.1 | Reported as such. No cross-K comparison is drawn between a converged K and a still-rising one. |
| Bounded-growth gate fails, or any cell's post-training growth ≥ 1.05 | That K is not a fix-on cell; excluded from every confound-related conclusion. |

### 4.1 The row Batch 05 showed was missing

[Batch 05](PHASE2_BATCH_05_REPORT.md) §3 returned `UNRESOLVED_ESCALATE` partly
because its outcome table had no row for the pattern that occurred, and partly
for a reason that generalises to every table in this project:

> the table presupposes arms compared at comparable states, and at 35,000 steps
> they were not — arm A had plateaued at slope −0.090 /1k while arm C was still
> climbing at +17.694 /1k.

A fixed budget does not deliver convergence for every arm, and comparing a
converged arm against a still-rising one measures the budget as much as the arm.
So this table carries the row explicitly:

**If some K values have plateaued (slope < 0.1 /1k) and others have not, the
per-K results are reported with their slopes and no cross-K ordering is
claimed.** The per-K classifications stand — they are per-K, not comparative —
and the comparison is deferred to a batch that runs the slower K values to a
plateau.

This is the one substantive change from the draft, and it is a **restriction**:
it can only withhold a comparison, never produce one.

## 5. What this batch cannot establish

* **Nothing at matched compute.** Steps are matched; K = 8 does eight times
  K = 1's block executions. That is Batch 05's question, at T = 3.
* **Nothing attributable to one lever.** Two stage lengths change together (§2.1).
* **Nothing about T = 8**, the other primary depth.
* **No takeoff probability.** Three seeds per K, and the project has repeatedly
  seen bimodal cells.
* **Nothing about the grid.** The grid trains at the target depth from scratch
  with no curriculum. A curriculum result, however strong, is evidence *for
  amending* the grid's protocol, not a substitute for running it.

## 6. The decisions this pre-registration applies

Recorded in [PHASE2_BATCH_06_DRAFT.md](PHASE2_BATCH_06_DRAFT.md) §7 on
2026-10-01, **while Batch 05 was running with 0 of 9 cells complete and no report
file in existence.** That timing is the point: this design was fixed without
knowing whether Batch 05 would find depth or compute, so it cannot have been
tuned to that answer.

| # | decision | applied |
|---|---|---|
| 1 | Full design at ≈ 10.7 GPU-h: T = 3 rung 15,000, target stage 60,000, K ∈ {1, 4, 8}, 3 seeds. None of the three costed reductions taken. | §2, §3 |
| 2 | **No adaptive stopping rule** — a budget that depends on progress would differ across K and reintroduce the per-K asymmetry §1.3 exists to prevent. Fixed 90,000 steps for every cell; final slopes reported; a still-rising cell reported as such. | §2, §4 |
| 3 | **Keep the 5,000-step T = 4 rung** for schedule comparability with Arm 3 and Batches 02–04, knowingly spending steps Batch 04 showed achieve nothing there. | §2 |
| 4 | **Record the two-lever limitation** as drafted: the T = 3 rung and the target stage lengthen together, so a positive result is not attributable to either alone. | §2.1, §5 |

Decision 2 is why §4 separates "plateau" from "still rising": with a fixed
budget, nothing guarantees either.

## 7. Execution

```
systemctl --user enable --now tesseract-chain@06.service
```

Persistent enabled unit passing `--resume`, so a reboot restarts it and it
continues from the cells already on disk. Batch 05 is complete and its unit
disabled, so there is no GPU contention. This file is committed before that
command runs.
