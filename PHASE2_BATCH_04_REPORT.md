# Phase 2 — Batch 04 report: T = 4 is learnable at K = 4, and Batch 01's exclusion is withdrawn

**Date: 2026-10-01.** Chain completed 23:58:15, `rc=0`, all nine cells at 40,000
steps. Runtime 4 h 37 min 49 s against a pre-registered estimate of ≈ 4.75 GPU-h;
actual **4.63 GPU-h**. No interruption, no resume, no recovered cells.

Report: `runs/phase2/batch04/t4_k_sweep/t4_k_sweep_report.json`.
Curves: `runs/phase2/batch04/t4_k_sweep/curve_analysis.json`.
Gate: `runs/phase2/batch04/t4_k_sweep/bounded_growth_gate.json`.
Pre-registered in [PHASE2_BATCH_04_PREREG.md](PHASE2_BATCH_04_PREREG.md),
committed before launch.

## 1. Headline

**The pre-registered §8 outcome is `T4_LEARNABLE_PROPOSE_GRID_RESUME_DRAFT`.**

T = 4 is the benchmark's **first primary depth** — it passes G2, unlike T = 2 and
T = 3. Batch 01 tested four levers against the T = 4 plateau, excluded all four,
and landed on "all arms NOT LEARNABLE". Nothing in this project had ever moved at
T = 4: Probe A and Arm 3 both sat at ~1.85 %.

With the residual fix **and** a curriculum climbing to it — a combination Batch 01
never tried — one K = 4 cell reaches **87.7384 % token accuracy and 12.8906 %
exact match**: 264 correct 17-position rings out of 2,048.

**So Batch 01 §9's exclusion of the residual confound at T = 4 is withdrawn.**
That exclusion rested on Arms 2b and 2c, neither of which combined the fix with
the curriculum, and Addendum 01 §A1.4 had already flagged that the combination
was untested.

**T = 4 is learnable. It is not solved.** K = 4 solved T = 3 at 100 % exact match;
here its best cell gets 12.89 %. The distinction is load-bearing and §4 keeps it.

## 2. Classification and the exact-match co-primary

| K | pre-registered | amended | learnable seeds | best exact match | §4 reading |
|---|---|---|---|---|---|
| 1 | PARTIAL | **NOT LEARNABLE** *(changed)* | — | 0.0000 % | — |
| 4 | LEARNABLE | **LEARNABLE** | 0, 2 | **12.8906 %** | **LEARNABLE_WITH_EXACT_MATCH** |
| 8 | PARTIAL | **PARTIAL** | 0 | 0.0488 % | — |

Per cell, at T = 4 on the best checkpoint:

| K | seed | token accuracy | exact match | distinct tokens | growth |
|---|---|---|---|---|---|
| 1 | 0 | 1.9244 % | 0.0000 % | **2** | 1.0000× |
| 1 | 1 | 1.9244 % | 0.0000 % | **1** | 1.0000× |
| 1 | 2 | 1.8583 % | 0.0000 % | 23 | 1.0000× |
| 4 | 0 | 32.5971 % | 0.0000 % | 60 | 1.0000× |
| 4 | 1 | 1.8469 % | 0.0000 % | 8 | 1.0000× |
| 4 | 2 | **87.7384 %** | **12.8906 %** | 60 | 1.0000× |
| 8 | 0 | 55.9599 % | 0.0488 % | 60 | 1.0000× |
| 8 | 1 | 1.8813 % | 0.0000 % | 39 | 1.0000× |
| 8 | 2 | 1.8756 % | 0.0000 % | 41 | 1.0000× |

The bounded-growth gate passed at all three K, and the post-training bound held
at **1.0000× in all nine cells**, so §8's gating condition is satisfied
throughout.

### 2.1 Addendum 03 caught a majority-class artifact for the third time

K = 1 classified **PARTIAL** under the original rule and **NOT LEARNABLE** under
the amended one. The reason is exact: seeds 0 and 1 both scored **1.9244 %**,
which is the T = 4 majority-class baseline to four decimals, while emitting 2 and
1 distinct tokens respectively. They are constant predictors of the modal token.

The original above-chance bar admitted them; the Addendum 03 floor requires
strictly beating the majority baseline, and 1.9244 % does not beat 1.9244 %. This
is the third time that amendment has rejected a majority-class artifact — after
Arm 2b and Arm 2c in Batch 01 — and the first time it has changed a verdict in a
batch where something else genuinely worked.

### 2.2 The exact-match floor did exactly what it was pre-registered for

**K = 8 seed 0 reached 55.9599 % token accuracy with 0.0488 % exact match — one
correct ring in 2,048.** On token accuracy alone that cell looks like a strong
result; it is partial knowledge that never completes an example. At 55.96 %
per-token accuracy over 17 positions, ~10⁻⁵ exact match is what independence
would predict, and one ring is consistent with that.

Because K = 8 classifies PARTIAL, the floor is not what excludes it. But had its
second seed taken off, K = 8 would have been LEARNABLE at ~0.05 % exact match and
the §4 reading would have reported it as *"learnable on token accuracy only, no
correct rings"* and counted it negative on the depth question. The floor was
fixed at 1 % before the batch ran, and this batch shows why: it is the difference
between K = 4's result and K = 8's, and token accuracy does not see it.

## 3. Every cell that took off is still rising steeply — the dominant caveat

From the recorded curves. The T = 4 target stage spans steps 20,001–40,000.

| K | seed | crossed 6.67 % at | crossed 26.63 % at | slope over last 2,500 steps | still rising |
|---|---|---|---|---|---|
| 4 | 0 | 21,500 | 32,750 | **+0.732** /1k | yes |
| 4 | 2 | 21,750 | 30,500 | **+3.251** /1k | yes |
| 8 | 0 | 24,000 | 31,750 | **+5.136** /1k | yes |
| 1, and the non-takeoff cells | never | never | ~0 | no |

**All three cells that learned anything were still climbing at step 40,000**, and
K = 8 seed 0 at **+5.136 accuracy points per 1,000 steps** has the steepest slope
recorded anywhere in this project. Extrapolating that slope is not legitimate,
but it makes one thing unavoidable:

> **T = 4's numbers are where the budget ran out, not where learning stopped.**

Takeoff occurs 1,500–4,000 steps into the target stage, but crossing 26.63 %
takes until step 30,500–32,750 — i.e. 10,000–13,000 steps of climbing — and none
had plateaued. The 20,000-step target stage is the binding constraint on every
number in §2, and the same was true at T = 3, where Batch 03's K = 1 cells were
still rising at 35,000.

Any grid-resume proposal must therefore choose a budget from this data rather
than inheriting 20,000 steps.

## 4. What "learnable, not solved" means here

| depth | best exact match | at |
|---|---|---|
| T = 3 (Batch 03) | **100.0000 %** | K = 4, two of three seeds |
| T = 4 (this batch) | **12.8906 %** | K = 4, one of three seeds |

T = 3 with the fix is *solved*: two K = 4 cells got every ring right. T = 4 is
*learnable*: one cell got 12.9 % of rings right and was still improving. The gap
may be budget (§3), depth, or both, and this batch cannot separate those.

So the depth boundary claim from Batch 03 — "T = 3 is the boundary with the fix
on" — is **not** what this batch concludes. T = 4 is past that boundary in the
sense that it moves at all, and short of it in the sense that nothing solves it.

## 5. K = 4 beats K = 8 at T = 4, which is new

At T = 3, K = 4 and K = 8 were both essentially perfect and indistinguishable.
At T = 4 they separate, in K = 4's favour, on both metrics:

| K | seeds at bar | best token accuracy | best exact match |
|---|---|---|---|
| 4 | **2/3** | 87.7384 % | **12.8906 %** |
| 8 | 1/3 | 55.9599 % | 0.0488 % |

Two readings, and this batch does not choose between them:

* **Takeoff reliability.** K = 4 took off in 2 of 3 cells, K = 8 in 1 of 3. That
  is a difference of one cell, well inside what three Bernoulli draws produce by
  chance, so it may be nothing.
* **A real ordering.** K = 8's one takeoff had the *steepest* slope in the batch
  (+5.136 /1k), which is the opposite of a capability ceiling — it looks like a
  cell that started late and ran out of budget, consistent with §3 rather than
  with K = 8 being worse.

Three seeds cannot separate these. Recorded as an observation, not a finding.

## 6. Stage boundaries: the ladder starves K = 1, and all T = 4 learning is in the target stage

T = 3 consolidation at the end of its 5,000-step rung (step 15,000), against the
final T = 4 result:

| cell | T = 3 at rung end | final T = 4 |
|---|---|---|
| K = 1 s0 | **2.14 %** | 1.92 % (degenerate) |
| K = 4 s1 | 4.85 % | 1.66 % (no takeoff) |
| K = 4 s2 | 15.35 % | **87.74 %** |
| K = 4 s0 | 33.03 % | 32.60 % |

**The 5,000-step T = 3 rung is not equally short for all K.** K = 4 extracts
15–33 % of T = 3 from it; K = 1 extracts 2.14 %, which is nothing. Batch 03
measured K = 1 needing until step 21,000–27,000 to take off at T = 3 — 6,000 to
12,000 steps *inside* a 20,000-step stage — so K = 1 arrives at the T = 4 stage
without the prerequisite depth.

**So the K = 1 null at T = 4 is confounded with ladder length.** It is not
evidence that K = 1 cannot learn T = 4; it is consistent with K = 1 never having
been given T = 3. The ladder lengths were inherited from Batch 02/03 for
comparability, which was the right choice, and this is its cost.

Partial T = 3 looks necessary but not proportional: the two cells with meaningful
T = 3 (33 %, 15 %) both learned T = 4, the one with 4.85 % did not — but seed 0
had twice seed 2's T = 3 and reached a third of its T = 4.

**All T = 4 learning happens in the target stage.** At step 20,000, the end of the
T = 4 rung, every cell sits at 1.7–2.9 % on T = 4. The 87.74 % is built entirely
in steps 20,001–40,000.

**Erasure remains total and is now extreme.** K = 4 seed 2 finishes with T = 4 at
87.74 % while T = 1, T = 2 and T = 3 sit at ~1.65 % with **losses around 21** — not
merely forgotten but confidently wrong. Every batch has shown this; this is the
first time it appears in a cell that largely mastered a primary depth.

**The ladder rungs learn composition, not the shortcut.** Every cell shows T = 2
at 100 % with shortcut agreement **26.74 %** at its boundary — the exact-solver
signature, since the shallow candidate is right on 26.67 % of tokens. No
attribution is computed at T = 4, which passes G2.

## 7. What this batch cannot establish

* **Nothing at matched compute.** Steps are matched; K = 8 does eight times
  K = 1's block executions. [PHASE2_BATCH_05_DRAFT.md](PHASE2_BATCH_05_DRAFT.md)
  is designed to remove this and is currently blocked (§4.1 of that draft).
* **No converged accuracy at any K.** §3. Every informative cell was still rising.
* **Nothing about K = 1 at T = 4**, beyond that this curriculum does not get it
  there (§6).
* **No takeoff probability.** Three seeds per K, and two of the three K values
  were bimodal.
* **Nothing about T = 8.** Untouched, and it is the other primary depth.
* **Nothing about the grid.** A positive result proposes resuming it; it does not
  substitute for it. The grid trains at the target depth from scratch, with no
  curriculum, which is not what was run here.

## 8. The pre-registered action

§8 of the pre-registration, for this outcome:

> Propose, **draft only**, resuming the pre-registered Phase 2 grid **with the fix
> and the curriculum**, and record that Batch 01 §9's exclusion of the residual
> confound at T = 4 is withdrawn.

The withdrawal is recorded in §1. The draft is **not written here** — this report
stops at applying the table. Four things §3 and §6 say any such draft must
confront, none of which the pre-registration anticipated:

1. **The budget must come from measured takeoff, not from 20,000 steps.** Every
   informative cell was still rising, one at +5.136 points per 1,000 steps.
2. **The curriculum's rung lengths are not neutral across K.** 5,000 steps at
   T = 3 is ample for K = 4 and nothing for K = 1 (§6).
3. **Resuming "the grid with the curriculum" is not the pre-registered grid.**
   The grid trains at the target depth from scratch. A curriculum changes the
   protocol, so this is an amendment to the benchmark design, not a resumption,
   and it needs to be argued as one.
4. **Exact match must be the grid's primary measure**, or K = 8-like cells at
   56 % token accuracy and one correct ring will be read as successes.

## 9. Integrity record

| | |
|---|---|
| Pre-registration committed before launch | yes — `290f927`, with no artifacts in existence |
| Config drift guard against Batch 03 | in the runner; refuses to start on drift |
| Bounded-growth gate before any cell | PASS at all three K, 1.0000× |
| Post-training bound, per cell | held, 1.0000×, 9/9 |
| Exact match pre-registered as co-primary with a 1 % floor | yes, before the batch ran (§2.2) |
| Amended rule applied in-run | yes, and it changed K = 1's verdict (§2.1) |
| Partial report written per cell | yes — the Batch 03 reboot fix, inherited |
| Interruption | none |
| Deviations | none |
| Caveat found during the run and recorded before the result | the ladder starving K = 1 (§6) |

Batch 04 stops here.
