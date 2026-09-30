# Phase 2 — Batch 04 pre-registration: T = 4 with the fix and the full curriculum

**Date: 2026-10-01. Committed before the batch runs.**

Batch 03 showed that with the residual stream bounded, T = 3 is solved exactly at
K ≥ 4 — 100 % exact match in two of three K = 4 cells — and that Batch 02's K = 8
failure was the residual confound. The fix has never been run at **T = 4**, the
benchmark's first primary depth.

Batch 01 excluded the residual confound as an explanation for the T = 4 plateau
on the strength of Arms 2b and 2c, **neither of which combined the fix with the
curriculum**. Batch 03 showed that exact combination turning a flat K = 8 into an
exact solver at T = 3. So the exclusion is not safe, and this batch tests it.

Verified immediately before committing: `runs/phase2/batch04/` does not exist, no
`t4_k_sweep_report.json` exists anywhere, and `systemctl --user` lists no
`tesseract-batch04` or `tesseract-chain@04` unit.

## 1. The question

**With the residual stream bounded and a curriculum that climbs to it, is T = 4
learnable, and at which K?**

Unlike Batch 02 and Batch 03, this is at a **primary depth**. T = 4 passes G2 and
is in the benchmark's `t_values`. A positive result here would be the first
evidence bearing on the depth hypothesis on the benchmark proper — which is why
§8's positive branch proposes resuming the grid rather than drawing a conclusion.

It is still not a grid cell: it is a curriculum diagnostic against a single
reference, the grid's protocol trains at the target depth from scratch, and no
number here enters the grid.

## 2. The configuration

| | value | against Batch 03 |
|---|---|---|
| Target depth | **T = 4** | 3 → **4** |
| Curriculum ladder | **T = 1 → 2 → 3 → 4**, 5,000 steps per stage | one more rung |
| Target stage | T = 4, 20,000 steps | same length |
| **Total steps per cell** | **40,000** | 35,000 → 40,000 |
| Recursion K | **{1, 4, 8}** | K = 2 dropped (§2.1) |
| Seeds | 0, 1, 2 | same |
| Width | medium (`phase2/prototype_medium`, 837,436 params) | same |
| Residual fix | **non-affine per-step normalisation, every K** | same |
| Optimiser | AdamW, constant 1e-3, weight decay 0.01 | same |
| Batch / eval interval | 128 / every 250 steps | same |
| Training data | fresh every step, T = 4 validation inputs excluded | same |
| Test split | never read | same |

The target depth appears twice — as the ladder's last rung and as the 20,000-step
target stage — as in Arm 3, Batch 02 and Batch 03. Boundaries fall at steps
5,000 / 10,000 / 15,000 / 20,000 / 40,000.

### 2.1 Why K = 2 is dropped

Batch 03 §5: K = 2 was bimodal, 1 of 3 seeds taking off, and its classification
turned on takeoff count rather than capability. Three seeds cannot estimate a
takeoff probability, so a fourth K value buys a number that could not be
interpreted, at 13 minutes per cell at T = 4. K ∈ {1, 4, 8} spans the range that
Batch 03 showed matters: K = 1 fails on the strict metric, K = 4 and K = 8
succeed.

This is a reduction in scope, recorded here rather than made silently. It means
**Batch 04 cannot report a trend across four K values** and does not attempt one.

## 3. The bounded-growth gate, first

As Batch 03: per-step hidden-state RMS is measured at K ∈ {1, 4, 8} with the fix
applied, and every K must show growth below **1.05**. **If any K fails, no
training cell runs.** Verified at 1.0000× for all of K ∈ {1, 2, 4, 8} in
Batch 03, so this is expected to pass; it runs anyway because a null result at an
unbounded K would be uninterpretable.

The post-training bound is recorded **per cell** and gates every
confound-related conclusion, as in Batch 03.

## 4. Metrics: exact match is a reported co-primary

[PHASE2_BATCH_03_REPORT_NOTE_01.md](PHASE2_BATCH_03_REPORT_NOTE_01.md) records
that at T = 3 exact match is the more informative measure, and that K = 1's
LEARNABLE verdict there rests on token accuracy at ~0 % exact match — 0.0000 % in
two of three cells at 43–65 % token accuracy. The target is a 17-position ring
and token accuracy averages over positions, so a model making many independent
local errors scores well on it while never producing a correct example.

**So exact match is pre-registered here as a reported co-primary metric.**

| | |
|---|---|
| **The classification rule is unchanged** | token accuracy ≥ 6.6667 % **or** loss ≤ 3.9943 nats, in ≥ 2 of 3 seeds, plus the Addendum 03 majority-class floor and collapse check |
| Majority-class floor at T = 4 | **1.9244 %** (already recorded) |
| **Exact match is recorded for every cell and every stage boundary** | and **no claim that T = 4 is learnable may be made without stating both** |

The rule stays on token accuracy deliberately: it was committed before any probe
ran, every batch since has been judged by it, and changing it now would make
Batch 04 incomparable with Batch 01–03 and would be a rule change made after
seeing Batch 03's numbers. Exact match is therefore **reported, not
classified** — it constrains interpretation, not the verdict.

**Pre-registered reading.** A verdict of LEARNABLE at a K whose exact match is
below **1 %** is reported as *"learnable on token accuracy only, with no correct
rings"*, and counts as a negative result for that K on the depth question. This
threshold is fixed now; 1 % of 2,048 examples is 20 rings, comfortably above the
3 rings K = 1 managed at T = 3.

## 5. No shortcut check at T = 4, and why the ladder still gets one

**T = 4 passes G2** (`runs/phase2/gates/gates_report.json`): its worst shallow
candidate sits at chance, so there is no shortcut to attribute a T = 4 result to,
and no attribution is computed at the target depth.

The ladder passes through T = 2 and T = 3, which **both fail G2** at 26.67 %. So
shortcut agreement is still recorded at those depths in the stage-boundary
diagnostics, where it is free and where it tests whether the ladder rungs learn
composition. Batch 02 and Batch 03 both showed 26.74 % at the T = 2 boundary —
the exact-solver signature — and that check carries over.

## 6. Recorded fields

Per cell, as Batch 03, with T = 4 added to the diagnostic depths:

* **stage-boundary diagnostics** at T ∈ {1, 2, 3, 4} at each of the five
  boundaries: token accuracy, **exact match**, loss, distinct predicted tokens,
  majority-class frequency, and shortcut agreement at T = 2 and T = 3;
* **best-checkpoint diagnostics** at all four depths, including exact match;
* **post-training per-step hidden-state RMS** and its growth ratio;
* **takeoff steps and final slope** from `scripts/phase2_batch02_curves.py`;
* distinct predicted tokens, for the collapse check.

**A partial report is written after every cell** (`t4_k_sweep_partial.json`).
Batch 03 lost nine finished cells to a reboot because the report is written only
at the end; that is fixed and this batch inherits the fix.

## 7. Execution and reboot survival

```
systemctl --user enable --now tesseract-chain@04.service
```

A persistent user unit, enabled, so a reboot restarts it and `--resume` continues
from the cells already on disk — the failure that cost Batch 03 nine cells of
mid-training data. Linger is on. The unit is deliberately **not**
`Restart=on-failure`: a run that dies for a reason other than a reboot should be
looked at, not silently retried.

## 8. Pre-registered outcomes

| outcome at T = 4 | action |
|---|---|
| **LEARNABLE at any K**, with exact match ≥ 1 % at that K | Propose, **draft only**, resuming the pre-registered Phase 2 grid **with the fix and the curriculum**, and record that Batch 01 §9's exclusion of the residual confound at T = 4 is withdrawn. |
| **LEARNABLE at some K but exact match < 1 % at every K** | Not a depth result. Report as "learnable on token accuracy only, no correct rings" (§4). The boundary claim below applies. |
| **NOT LEARNABLE or DEGENERATE at every K** | **T = 3 is the depth boundary with the fix on.** Report that; the grid is not resumed. |
| **Bounded-growth gate fails at any K**, or a cell's post-training growth exceeds 1.05 | That K is not a fix-on cell; its result is excluded from every confound-related conclusion. |
| **K = 1 learnable and K ≥ 4 not** | Would contradict Batch 03's ordering and is treated as a **result requiring investigation**, not a finding: report and stop. |

Nothing in this table is reachable without §3's gate passing at the K concerned.

## 9. What this batch cannot establish

* **Nothing at matched compute.** Steps are matched; K = 8 does eight times
  K = 1's block executions. This is the confound
  [PHASE2_BATCH_05_DRAFT.md](PHASE2_BATCH_05_DRAFT.md) is designed to remove,
  and it is not removed here.
* **Nothing about the grid.** A positive result proposes resuming it; it does not
  substitute for it.
* **No converged accuracy.** Batch 03's K = 1 cells were still rising at 35,000
  steps; 40,000 is not obviously enough either, and a low result at any K may be
  slow takeoff rather than a ceiling.
* **No takeoff probability.** Three seeds per K.
* **Nothing about T = 8.** Untouched.

## 10. Compute

Per-step times **measured on this machine** with the fix applied at T = 4,
medium width, batch 128, deterministic kernels, 120 timed steps after 20 warmup:

| K | s/step | per cell (40,000 steps) | 3 seeds |
|---|---|---|---|
| 1 | 0.01144 | 7.63 min | 22.9 min |
| 4 | 0.04040 | 26.93 min | 80.8 min |
| 8 | 0.08008 | 53.39 min | 160.2 min |
| | | **training total** | **4.40 GPU-h** |

| | GPU-h |
|---|---|
| Training, 9 cells | 4.40 |
| Evaluation overhead (+7.6 %, measured from Batch 02/03 outturn) | 0.33 |
| Bounded-growth gate | ~0.02 |
| **Estimated total** | **≈ 4.75 GPU-h** |

**Under the 8 GPU-h threshold, so the batch proceeds without further approval.**
Disk: 9 cells × 2 checkpoints × 3.20 MB ≈ 58 MB, plus curves and JSON ≈ 7 MB, so
**≈ 65 MB**.

This file is committed before the batch is started.
