# Phase 2 — Batch 05 report: shared recursion solves T = 3; the alternatives don't get there

**Date: 2026-10-03.** Chain completed 04:20:22 on 2026-10-02, `rc=0`, all nine
cells at 35,000 steps. A second invocation at 17:37:24 (the enabled unit firing
after a reboot) correctly **skipped** because the report already existed — the
reboot-survival design behaving as intended.

> **[PHASE2_BATCH_05_REPORT_NOTE_01.md](PHASE2_BATCH_05_REPORT_NOTE_01.md)
> (2026-10-03) records that all arms used a learning rate inherited from arm A's
> family, so §4's "faster" is partly confounded with optimiser settings tuned for
> arm A's scale.** No classification or number changes. Nothing below is edited;
> this notice is the only line added after the report was committed.

Report: `runs/phase2/batch05/t3_family/t3_family_report.json`.
Curves: `runs/phase2/batch05/t3_family/curve_analysis.json`.
Gate: `runs/phase2/batch05/t3_family/bounded_growth_gate.json`.
Pre-registered in [PHASE2_BATCH_05_PREREG.md](PHASE2_BATCH_05_PREREG.md),
committed before launch.

## 1. Headline, and the honest version of it

**The mechanical comparison returned `UNRESOLVED_ESCALATE`.** That is the correct
outcome, for two reasons — one of which is a gap in the table I wrote (§3).

| arm | parameters | forward FLOPs | block calls | best exact match | best token accuracy |
|---|---|---|---|---|---|
| **A** shared K = 4 | **837,436** | 108,068,864 | 4 | **100.0000 %** | 100.0000 % |
| **B** width-scaled K = 1 | 3,214,678 | 108,077,568 | 1 | 0.1953 % | 70.4073 % |
| **C** unrolled L = 4 | 3,206,716 | 108,068,864 | 4 | **0.0000 %** | 82.8355 % |

**What is established.** At matched forward compute and a fixed 35,000-step
budget, the shared-weight recursion reaches a **converged, exact** solution —
100 % exact match in all three seeds, with negative final slopes — while neither
alternative produces a single correct ring, despite each holding **3.8× more
parameters**.

**What is not established.** That the alternatives *cannot* get there. Both were
still improving steeply when the budget ended, arm C at **+17.694 accuracy points
per 1,000 steps** — the steepest slope recorded anywhere in this project. The
comparison at 35,000 steps is therefore **not a comparison at convergence**, and
§4 is the finding that does not depend on convergence.

## 2. Per cell

| arm | seed | exact match | token accuracy | loss | distinct | shortcut agr. | growth | s/step |
|---|---|---|---|---|---|---|---|---|
| A | 0 | **100.0000 %** | 100.0000 % | 0.0003 | 60 | 26.63 % | 1.0000× | 0.04218 |
| A | 1 | 99.9023 % | 99.9943 % | 0.0005 | 60 | 26.62 % | 1.0000× | 0.04195 |
| A | 2 | **100.0000 %** | 100.0000 % | 0.0004 | 60 | 26.63 % | 1.0000× | 0.04171 |
| B | 0 | 0.0000 % | 1.8210 % | 4.0952 | **2** | 1.76 % | 1.0000× | 0.03046 |
| B | 1 | 0.1953 % | 70.4073 % | 1.0056 | 60 | 22.53 % | 1.0000× | 0.03064 |
| B | 2 | 0.0000 % | 1.8382 % | 4.0964 | 7 | 1.63 % | 1.0000× | 0.03050 |
| C | 0 | 0.0000 % | 21.6768 % | 2.3132 | 60 | 8.19 % | 1.0000× | 0.04288 |
| C | 1 | 0.0000 % | **82.8355 %** | 0.6702 | 60 | 21.94 % | 1.0000× | 0.04276 |
| C | 2 | 0.0000 % | 1.8411 % | 4.0960 | **3** | 1.72 % | 1.0000× | 0.04276 |

Classification: **A LEARNABLE (3/3), B PARTIAL (1/3), C LEARNABLE (2/3)**, every
attribution COMPOSITION, no arm excluded. The bounded-growth gate passed for all
three arms and the post-training bound held at **1.0000× in all nine cells**, so
all three are legitimate fix-on arms — which required the unrolled-family
normalisation implemented for this batch.

**Arm A replicates Batch 03's K = 4 cells to four decimal places** (100.0000 /
99.9023 / 100.0000), and its T = 3 rung-end values match Batch 04's K = 4
exactly too (33.0308 / 4.8455 / 15.3464). Same seeds, same config, deterministic
kernels, separate runs days apart — so this also confirms the determinism
machinery and that the change I made to `phase2/batch01.py` did not perturb the
shared-block path.

## 3. Why the comparison is UNRESOLVED, including my error

The rule refused to classify, with "the arm pattern matches no pre-registered
row". It was right to refuse, and the gap is mine.

**My outcome table had no row for A > C.** §7 of the pre-registration enumerated
A ≈ C > B, A ≈ B ≈ C, C > A, and B > A. It did not contemplate the shared arm
beating the unrolled arm, because I assumed that an arm with the same structure
and 3.83× more parameters would do at least as well. That assumption was wrong,
and because `DEPTH_NOT_COMPUTE` required A ≈ C, the actual pattern — A ahead of
both — fell through to UNRESOLVED.

**The deeper reason the refusal is correct.** Even with an A > C row, the
comparison would not be interpretable, because §4's premise — that arms are
compared at comparable states — does not hold at 35,000 steps:

| arm | mean takeoff (crossed 26.63 %) | final slope | converged |
|---|---|---|---|
| A | **15,500** | −0.090 /1k | **yes** |
| B | 23,250 | +0.633 /1k | no, 1 of 3 rising |
| C | **31,250** | **+8.126 /1k** | no, 2 of 3 rising |

Arm A had plateaued; arm C had crossed the shortcut ceiling ~3,750 steps before
the budget ended and was climbing at up to +17.694 points per 1,000 steps. So the
table's categories presuppose something the budget did not deliver. A fixed
budget cannot be assumed to reach convergence for every arm, and I should have
written an "arms not at convergence" row. **That is a pre-registration defect,
recorded here rather than patched retroactively.**

## 4. The result that does not depend on convergence

T = 3 accuracy at the end of its **5,000-step rung** (step 15,000), a fixed early
checkpoint identical for every arm:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **A** shared K = 4 | **33.0308 %** | **4.8455 %** | **15.3464 %** |
| B width-scaled K = 1 | 1.6745 % | 1.6573 % | 1.6429 % |
| C unrolled L = 4 | 1.5453 % | 1.7233 % | 1.6199 % |

Chance is 1.6667 %. **The shared recursion extracts 4.8–33.0 % of T = 3 from
5,000 steps; neither alternative leaves chance.** All three seeds of both
alternatives sit within 0.08 points of chance.

This is the finding that survives §3's caveat. It is measured at a fixed point
rather than at the budget's end, so it does not depend on whether any arm
converged. Stated precisely:

> **At matched forward compute, the shared-weight recursion finds the composable
> solution far faster than either spending that compute on width or on
> independent per-step parameters — with 3.8× fewer parameters than either.**

That is a claim about **step efficiency and inductive bias**, not about
representational capacity. Nothing here says arms B or C cannot represent F³; the
evidence is that they do not find it in the same budget, and arm C's trajectory
suggests it may find it later.

## 5. Exact match is near-binary, which limits what it can rank

A methodological point this batch forced, worth recording because it affects how
Batch 04's and Batch 06's numbers should be read.

Exact match over a 17-position ring is a very steep function of token accuracy.
Taking independent errors as a crude baseline:

| token accuracy | expected exact match | expected rings of 2,048 |
|---|---|---|
| 70 % | 0.23 % | 4.8 |
| 82.8 % | 4.04 % | 82.8 |
| 90 % | 16.7 % | 341 |
| 95 % | 41.8 % | 856 |
| 99 % | 84.3 % | 1,726 |

So exact match only becomes informative above roughly 95 % token accuracy, and
**it cannot rank two arms that are both below that.** It did exactly the job it
was adopted for — separating "learned the function" (arm A) from "learned
something local" (arms B and C) — but arms B and C are both at 0.0–0.2 % exact
match and the metric says nothing about which is closer. For that, §3's token
accuracy and slopes are what carry information.

**An asymmetry worth flagging.** Arm B at 70.4 % token accuracy produced 4 rings
against the ~4.8 independence predicts — a close match. Arm C at 82.8 % produced
**0** against ~83 predicted. So arm C's errors are strongly **clustered** across
ring positions where arm B's are not: it gets whole regions wrong rather than
scattered tokens. Independence is a crude baseline for a coupled cellular
automaton and this is an observation rather than a measurement of error
structure, but the contrast between the two arms under the same baseline is real
and points at a qualitatively different failure mode.

## 6. Arm C is the third high-token, zero-exact-match arm

| where | token accuracy | exact match |
|---|---|---|
| Batch 03, K = 1 at T = 3 | 64.5393 % | 0.1465 % |
| Batch 04, K = 8 at T = 4 | 55.9599 % | 0.0488 % |
| **Batch 05, arm C at T = 3** | **82.8355 %** | **0.0000 %** |

Arm C is the most extreme instance: it **classifies LEARNABLE** on the
token-accuracy rule — two of three seeds above the bar — with a best exact match
of **0.0000 %**. Had this batch been keyed on token accuracy, arm C at 82.8 %
would have read as nearly matching arm A, and the comparison would have come out
close to "compute, not depth".

That is three independent occurrences of a verdict that the pre-registered
classification calls learnable and that produces no correct examples. The
Batch 03 Note 01 decision to make exact match primary is vindicated, and
§5 adds the limit on what it can do.

## 7. Wall clock: equal FLOPs is not equal time

Measured in-run, averaged over each arm's three cells:

| arm | FLOPs / sequence | s/step | relative |
|---|---|---|---|
| A shared K = 4 | 108,068,864 | 0.04195 | 1.00× |
| B width-scaled K = 1 | 108,077,568 | **0.03053** | **0.73×** |
| C unrolled L = 4 | 108,068,864 | 0.04280 | 1.02× |

Arm B is **27 % faster** at matched FLOPs, because one wide matmul reaches higher
GPU utilisation than four small sequential ones. The pre-registration predicted
this from the draft and it is confirmed. So arm B is not merely compute-matched,
it is compute-matched *and cheaper in wall clock* — and still produced 4 correct
rings against arm A's 2,048.

## 8. What this batch cannot establish

* **Nothing at convergence for arms B and C** (§3). This is the batch's largest
  limitation and the obvious follow-up: run them to a plateau.
* **Nothing about total training compute.** Forward FLOPs are matched; arms B and
  C hold 3.8× the parameters, so their backward pass and optimiser state cost
  more. A reader wanting total compute matched does not get it here.
* **Nothing about representational capacity.** §4 is about finding a solution,
  not about whether one exists in the hypothesis class.
* **Nothing at T = 4 or T = 8.** T = 3 only, which fails G2.
* **No takeoff probability.** Three seeds per arm; arms B and C were both
  bimodal, and arm B's seed 0 and arm C's seed 2 were DEGENERATE.
* **Nothing about `param_matched`**, which is parameter-matched rather than
  compute-matched and carries a documented head-dimension confound.

## 9. What follows

The escalation §5 of the pre-registration committed to is this report. Two
follow-ups the data points at, neither chosen here:

1. **Run arms B and C to convergence.** Arm C at +17.694 points per 1,000 steps
   is the single most under-budgeted cell in the project. Until it plateaus, "A
   beats C" is a statement about 35,000 steps. Arm A converged by ~16,000, so a
   budget of 70,000–100,000 would test the alternatives on their own terms.
2. **Fix the pre-registration defect for any successor.** An outcome table over
   arms must include a row for "not all arms at convergence", decided from the
   final slopes, rather than assuming the budget delivers comparable states.

## 10. Integrity record

| | |
|---|---|
| Pre-registration committed before launch | yes — `e6e724f`, no artifacts in existence |
| Unrolled per-step norm implemented test-first | yes, and the tests caught an ordering error before any code existed |
| Bounded-growth gate across all three arms | PASS, 1.0000×, with arm B's pass recorded as vacuous |
| Post-training bound, per cell | held, 1.0000×, 9/9 |
| Exact match primary, threshold argued before the run | yes (prereg §5) |
| Arm A replication of Batch 03 | exact, to four decimals |
| Reboot during the batch | yes — the enabled unit re-fired and correctly skipped |
| **Pre-registration defect found** | **yes — no A > C row, and no "not at convergence" row (§3)** |
| Interruption or lost cells | none |

Batch 05 stops here.
