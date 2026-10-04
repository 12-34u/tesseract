# Phase 2 — Batch 06 report: T = 4 is solved at K = 8, and K = 4 collapses

**Date: 2026-10-04.** Chain completed 06:07:50, `rc=0`, all nine cells at 90,000
steps. Runtime 12 h 13 min 13 s against a pre-registered estimate of
≈ 10.67 GPU-h; actual **12.22 GPU-h**, 15 % over (§8).

> **[PHASE2_BATCH_06_REPORT_NOTE_01.md](PHASE2_BATCH_06_REPORT_NOTE_01.md)
> (2026-10-04) diagnoses the K = 4 collapse from the existing logs: the transition
> shock is universal at every K, the output collapse is near-universal, and the
> discriminator is whether a cell ever gets below chance loss. The collapse is a
> symptom of a chance plateau, not its cause.** No number or classification
> changes. Nothing below is edited; this notice is the only line added.

Report: `runs/phase2/batch06/t4_long/t4_long_report.json`.
Curves: `runs/phase2/batch06/t4_long/curve_analysis.json`.
Gate: `runs/phase2/batch06/t4_long/bounded_growth_gate.json`.
Pre-registered in [PHASE2_BATCH_06_PREREG.md](PHASE2_BATCH_06_PREREG.md),
committed before launch, with the decisions recorded before Batch 05 reported.

## 1. Headline

**T = 4 — the benchmark's first primary depth — is solved at K = 8, in all three
seeds.**

| cell | T = 4 token accuracy | **T = 4 exact match** | distinct | T = 3 at rung end |
|---|---|---|---|---|
| K = 1 s0 | 1.9244 % | 0.0000 % | 1 | 42.68 % |
| K = 1 s1 | 1.9388 % | 0.0000 % | 6 | 23.14 % |
| K = 1 s2 | 5.7474 % | 0.0000 % | 60 | 18.34 % |
| K = 4 s0 | 1.9244 % | 0.0000 % | **2** | **100.00 %** |
| K = 4 s1 | 1.9244 % | 0.0000 % | **1** | **99.85 %** |
| K = 4 s2 | 1.9244 % | 0.0000 % | **1** | **99.99 %** |
| **K = 8 s0** | **99.9282 %** | **98.7793 %** | 60 | 99.68 % |
| **K = 8 s1** | **99.3940 %** | **90.4297 %** | 60 | 98.45 % |
| **K = 8 s2** | **98.9143 %** | **84.1309 %** | 60 | 70.36 % |

98.7793 % exact match is **2,023 correct 17-position rings out of 2,048**.
Batch 04's best at T = 4 was 12.89 %. Classification: **K = 1 PARTIAL, K = 4
DEGENERATE, K = 8 LEARNABLE** with every bound held at 1.0000×.

**The pre-registered outcome is `T4_LEARNABLE_BUT_STILL_RISING`**, with
`supports_depth_claim: false`, because all three K = 8 cells have final slopes
above the 0.1 /1k plateau threshold. **§3 shows that determination is an artefact
of the test, not a property of the cells** — and the outcome is reported as the
rule produced it rather than overridden.

## 2. Both levers worked, and they had opposite effects by K

The two changes against Batch 04 were a 15,000-step T = 3 rung (from 5,000) and a
60,000-step target stage (from 20,000). Both did what §1.1 and §1.2 predicted, and
the consequences diverge sharply across K.

**K = 1: the rung worked, and K = 1 still failed.** Its T = 3 at rung end went
from 2.14 % in Batch 04 to **18.3–42.7 %** here — above the 15–33 % that preceded
K = 4's Batch 04 successes. Its best T = 4 token accuracy rose from 1.9244 % to
**5.7474 %**, and seed 2's loss fell to 3.9086, below the 3.9943 learnable bar.
So the ladder **was** part of the explanation and removing it moves K = 1
measurably — but not to a single correct ring.

> An earlier framing of mine, that "the ladder was not the explanation", was too
> strong and is corrected here: the ladder was starving K = 1, and relieving that
> is necessary but not sufficient.

**K = 4: the longer rung appears to have destroyed it.** All three cells solved
T = 3 to 99.85–100.00 % in the rung, then collapsed at T = 4 to exactly the
1.9244 % majority-class floor with **1–2 distinct tokens** — DEGENERATE in 3 of 3.
In Batch 04, with a 5,000-step rung giving only *partial* T = 3 (4.85–33.03 %), the
same K reached 32.60 % and 87.74 %.

| K = 4 | T = 3 at rung end | final T = 4 |
|---|---|---|
| Batch 04 (5,000-step rung) | 4.85–33.03 % | 1.66 / 32.60 / **87.74 %** |
| Batch 06 (15,000-step rung) | **99.85–100.00 %** | 1.92 / 1.92 / 1.92 %, all degenerate |

This is systematic — 3 of 3, at the floor, with 1–2 tokens — not seed variance.

**K = 8: solved T = 3 and went on to solve T = 4.** Which refutes the obvious
reading of the K = 4 result. I suggested mid-run that fully solving T = 3 might
harm the T = 4 transition; **that is wrong as stated**, because K = 8 also solved
T = 3 (70.4–99.7 %) and then solved T = 4. Whatever happened to K = 4 is specific
to K = 4, not a general consequence of a well-learned prerequisite.

**So the two levers interact with K, and in opposite directions.** The
pre-registration recorded that the two levers are inseparable (§2.1); what it did
not anticipate is that their *sign* differs by K. The batch therefore cannot be
read as "T = 4 with the limits removed" uniformly — it is a different experiment
at each K.

## 3. The plateau test misfires at the ceiling

The rule says every K = 8 cell is still rising. The data says otherwise, and the
discrepancy is a defect in the test I wrote.

The last 2,500 steps of K = 8 seed 0's token accuracy, at eleven evaluations:

```
99.01  98.89  99.07  99.16  98.31  99.49  98.35  99.22  99.11  99.93  99.90
```

That oscillates over a 1.62-point range with no trend. The fitted slope is
**+0.322 /1k**, and at 0.1005 points of headroom to 100 % **it would exceed 100 %
in 312 steps** — which is impossible. The slope is being fitted through
evaluation noise.

| cell | sd over the last 11 evals | range | fitted slope | steps to exceed 100 % |
|---|---|---|---|---|
| K = 8 s0 | 0.5223 pts | 1.6171 | +0.322 /1k | **312** |
| K = 8 s1 | 0.3578 pts | 1.3499 | +0.121 /1k | 8,416 |
| K = 8 s2 | 0.6053 pts | 2.5592 | +0.148 /1k | 15,321 |

The 0.1 /1k threshold asks the fit to resolve **0.25 points across the window**
against per-evaluation noise of 0.36–0.61 points. It cannot. The threshold was
chosen to separate Batch 04's and Batch 05's climbing cells, which ran at **+1 to
+17 /1k** — a regime where 0.1 is safely small. Near the ceiling it measures
noise.

**Substantively, all three K = 8 cells have plateaued**, around 97.7–99.9 % token
accuracy and 84–99 % exact match.

**The pre-registered outcome stands as produced.** I am not overriding it: the
rule was committed before the batch ran, it was applied mechanically, and
retrofitting it after seeing a result I like is precisely what pre-registration
exists to prevent. What the record gets instead is this section, and a corrected
test for any successor:

* make the threshold **relative to remaining headroom**, not absolute — e.g.
  "slope × window < 10 % of (100 − current accuracy)"; or
* require the slope to **exceed the eval-to-eval standard deviation** over the
  window before calling a cell rising; or
* test the plateau on a **longer window** (10,000 steps, 40 evaluations) so noise
  averages down; or
* test on **exact match**, which at the ceiling moves more than token accuracy
  and is the primary metric anyway.

Any of the four would have classified these three cells as plateaued.

## 4. No cross-K comparison is withheld, and why that is correct here

Batch 05 showed every outcome table in this project needed a
"not all K at convergence" row, and §4.1 of the pre-registration added one. It
did **not** fire: `comparison_withheld: false`.

That is right rather than a miss. The row withholds a comparison only among K
values that *learned something*, and only K = 8 is LEARNABLE. K = 4 is DEGENERATE
and K = 1 is PARTIAL with no cell clearing the accuracy bar, so neither has a
trajectory to converge and neither contributes a verdict — `plateaued_by_k` is
`{1: None, 4: None, 8: False}`. **There is no cross-K ordering to withhold**,
because there is only one K with a result. The per-K verdicts stand on their own,
which is what they are for.

## 5. Takeoff is far later at T = 4 than at T = 3

| | crossed 6.67 % | crossed 26.63 % | steps into the target stage |
|---|---|---|---|
| K = 8 s0 | 36,250 | 42,750 | **12,750** |
| K = 8 s1 | 48,750 | 59,500 | **29,500** |
| K = 8 s2 | 33,250 | 60,250 | **30,250** |

Against T = 3, where arm A crossed the ceiling ~750 steps into its target stage.
So T = 4 needs **12,750–30,250 steps of target-stage training** before the
shortcut ceiling is passed — 17–40× longer. **Batch 04's 20,000-step target stage
was below the requirement for two of these three cells**, which is why it saw
12.89 % where this batch sees 98.78 %, and it vindicates §1.2's reasoning
independently of the slope question.

## 6. What this does to earlier batches

| claim | status |
|---|---|
| Batch 04: T = 4 learnable at K = 4, 12.89 % exact match | **stands for that configuration**, and is now understood as budget-limited: the target stage was shorter than the takeoff requirement |
| Batch 04: K = 4 > K = 8 at T = 4 | **reversed.** With an adequate budget K = 8 solves T = 4 and K = 4 collapses |
| Batch 04 §6: the ladder starves K = 1 | **confirmed and quantified** — 2.14 % → 42.68 % — and shown insufficient |
| Batch 01 §9: the residual confound excluded at T = 4 | remains **withdrawn** (Batch 04 §1) |
| Batch 05: shared recursion solves T = 3 at K = 4 | unaffected; T = 3 is a different depth |
| My mid-run claim that solving T = 3 harms the T = 4 transition | **wrong as stated** (§2) |

## 7. What this batch cannot establish

* **Nothing at matched compute.** Steps are matched; K = 8 does eight times
  K = 1's block executions, and K = 8 is the arm that won. This is the batch's
  largest confound and it is the one [Batch 07](PHASE2_BATCH_07_DRAFT.md) and
  Batch 05 address at T = 3, not here.
* **No attribution to either lever** (§2, prereg §2.1), and now worse than the
  prereg anticipated, since the levers' sign differs by K.
* **Nothing about why K = 4 collapsed.** Three cells at the majority floor after
  solving T = 3 is a clean observation with no mechanism attached.
* **Nothing about the learning rate**, which is 1e-3 everywhere and inherited from
  arm A's family ([Batch 05 Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md)). K = 4's
  collapse in particular could be a rate artefact and this batch cannot say.
* **No takeoff probability.** Three seeds per K.
* **Nothing about T = 8**, the other primary depth.
* **Nothing about the grid.** The grid trains at the target depth from scratch
  with no curriculum; a curriculum result is evidence for *amending* its protocol.

## 8. Compute

| | GPU-h |
|---|---|
| Pre-registered estimate | 10.67 |
| **Actual** | **12.22** (12 h 13 min 13 s) |

15 % over, because the estimate used Batch 04's per-step times and the evaluation
overhead scales with the step count: 90,000 steps at an eval every 250 gives 360
evaluations per cell against Batch 04's 160, so the +7.6 % overhead measured at
40,000 steps understates it at 90,000. Any successor estimating a long run should
scale the eval term with steps rather than applying a flat percentage.

## 9. Integrity record

| | |
|---|---|
| Pre-registration committed before launch | yes — `2038a97`, no artifacts in existence |
| Decisions recorded before Batch 05 reported | yes — draft §7, with Batch 05's state written down |
| Bounded-growth gate | PASS at all three K, 1.0000× |
| Post-training bound, per cell | held, 1.0000×, 9/9 |
| Exact match co-primary with a 1 % floor | yes, pre-registered |
| Amended rule applied in-run | yes, and it changed K = 4 from NOT LEARNABLE to DEGENERATE |
| Partial report written per cell | yes |
| Interruption | none |
| **Pre-registered test found defective** | **yes — the plateau threshold misfires at the ceiling (§3), reported not overridden** |
| Claim of mine falsified by this batch | yes — that solving T = 3 harms the T = 4 transition (§2) |
| Framing of mine corrected | yes — "the ladder was not the explanation" (§2) |

Batch 06 stops here. The pre-registered action for the plateau row is to report
and **not** propose a grid resume, and that is what this report does.
