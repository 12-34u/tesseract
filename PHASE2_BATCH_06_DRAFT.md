# Phase 2 — Batch 06 DRAFT: T = 4 with Batch 04's two limits removed

**Date: 2026-10-01. DRAFT — not pre-registered, not approved, not run.**
**Estimated at ≈ 10.7 GPU-h, which exceeds the 8-hour threshold and needs
explicit approval before it is pre-registered.**

Batch 04 found T = 4 learnable at K = 4 — 87.7384 % token accuracy and 12.8906 %
exact match in one cell — and recorded two limits that bound every number in it.
Batch 06 removes both. Nothing here is binding; if adopted it is committed as
`PHASE2_BATCH_06_PREREG.md` before any cell runs.

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

## 2. Configuration (proposed)

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
same batch size, 120 timed steps after 20 warmup. They are not re-measured here
because Batch 05 is currently training and a timing probe would contend with it
on the GPU, corrupting both numbers.

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

**This exceeds the 8 GPU-h threshold and needs explicit approval.** Three ways to
bring it down, if that is preferred to approving it:

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

My recommendation is to approve the full 10.67 GPU-h, with option 2 as the
fallback. Option 1 removes the arm that best motivates the batch, and option 3
changes a pre-registered rule.

Disk: 9 cells × 2 checkpoints × 3.20 MB ≈ 58 MB, plus curves ≈ 12 MB (90,000
steps at an eval every 250 gives 360 points per cell), so **≈ 70 MB**.

## 4. Pre-registered outcomes (proposed, not in force)

| outcome at T = 4 | reading |
|---|---|
| **Some K reaches a plateau** (final slope < 0.1 /1k) **with exact match ≥ 1 %** | T = 4 is learnable to a measurable level, and the level is reportable rather than budget-limited. This is the result the batch is for. |
| **Exact match ≥ 1 % at some K but still rising** | Learnable, still budget-limited. Report the slope and state that 90,000 steps is also not enough; do not quote the number as an achieved level. |
| **K = 1 now learnable where it was not in Batch 04** | The Batch 04 K = 1 null was the ladder, not K = 1. Attribute it to §1.1 and record that Batch 04's K = 1 row should be read that way. |
| **K = 1 still not learnable with a 15,000-step T = 3 rung** | The ladder was not the explanation. K = 1's T = 4 null stands with the ladder confound removed. |
| **Nothing learnable anywhere** | Contradicts Batch 04, which found K = 4 LEARNABLE on a shorter budget. Treat as a **batch failure requiring investigation**, not a finding. |
| exact match < 1 % at every K | "Token accuracy only, no correct rings", as Batch 04 §4 reads it. |

## 5. What this batch would still not establish

* **Nothing at matched compute.** Steps are matched; K = 8 does eight times
  K = 1's block executions. That is Batch 05's question, at T = 3.
* **Nothing attributable to one lever.** Two stage lengths change together (§2.1).
* **Nothing about T = 8**, the other primary depth.
* **No takeoff probability.** Three seeds per K, and the project has repeatedly
  seen bimodal cells.
* **Nothing about the grid.** The grid trains at the target depth from scratch
  with no curriculum. A curriculum result, however strong, is evidence *for
  amending* the grid's protocol, not a substitute for running it.

## 6. Open questions for the user

1. **Approve ≈ 10.67 GPU-h**, or take option 2 (≈ 8.3 GPU-h, 40,000-step target
   stage)? §3.
2. Is **60,000** the right target stage, or should it be set by a stopping rule —
   e.g. "train until the slope over the last 2,500 steps falls below 0.1 /1k, to
   a cap of 90,000" — which would spend compute only where it buys something? That
   is a protocol change and would need pre-registering as one, including what
   happens when a cell hits the cap.
3. Should the **T = 4 rung** (5,000 steps) be dropped? Batch 04 showed it achieves
   nothing — every cell was at 1.7–2.9 % on T = 4 at its end — and the target stage
   is the same depth. Keeping it preserves the schedule shape used since Arm 3;
   dropping it saves 5,000 steps per cell (~0.6 GPU-h).

Nothing in this file is in force.
