# Phase 2 — Batch 03 report: with the stream bounded, K ≥ 4 solves T = 3 exactly

**Date: 2026-10-01.** Chain completed 19:02:17 UTC, `rc=0`, all twelve cells at
35,000 steps. Interrupted by a machine reboot and resumed; see §0 and
[PHASE2_BATCH_03_ADDENDUM_01.md](PHASE2_BATCH_03_ADDENDUM_01.md).

Report: `runs/phase2/batch03/t3_k_sweep/t3_k_sweep_report.json`.
Curves: `runs/phase2/batch03/t3_k_sweep/curve_analysis.json`.
Gate: `runs/phase2/batch03/t3_k_sweep/bounded_growth_gate.json`.
Pre-registered in [PHASE2_BATCH_03_PREREG.md](PHASE2_BATCH_03_PREREG.md),
committed before launch.

## 0. Provenance: nine cells were recovered, not retrained

A reboot killed the run after nine of twelve cells. Their checkpoints survived;
the report did not, because it is written only once every cell completes.

**Nine cells (all of K = 1, 2 and 4) are `recovered_from_checkpoint`.** Everything
measured on the final model was recomputed from `best_checkpoint.pt`, so those
fields are artifact-backed exactly as a trained cell's are. The curve, the
stage-boundary rows, `steps_run` and the final-checkpoint loss come from the
console log and are tagged `recovered_from_console_log`.

**Three cells (all of K = 8) are `trained` in the ordinary way.**

The integrity check: recomputed best-validation accuracy against what the log
printed, per cell. **All nine agree to within 4.6 × 10⁻⁵ points** — log rounding —
with zero mismatches. A disagreement would have meant the checkpoint and the log
came from different runs.

**Permanently lost**, for the nine recovered cells only: `exact_match`,
`distinct_predicted_tokens` and `majority_class_percent` *inside the
stage-boundary rows*. None of the three enters any classification — the collapse
check uses the best-checkpoint distinct-token count, which is recomputed and
present for every cell. The loss is to the stage-by-stage descriptive picture,
and it falls on K = 1, 2 and 4 but not K = 8, so **that asymmetry lies along the
axis being compared** and is stated here rather than buried.

Addendum 01 also records the fix: the runner now writes a partial report after
every cell, so this cannot recur.

## 1. Headline

**The pre-registered §8 outcome is `BATCH02_K8_FAILURE_ATTRIBUTED_TO_CONFOUND`.**

Same width, same curriculum, same seeds, same 35,000-step budget. The only
change is Addendum 02's non-affine per-step normalisation:

| K = 8 | Batch 02 (`fix=none`) | Batch 03 (fix on) |
|---|---|---|
| seed 0 | 1.8411 % | **99.9856 %** |
| seed 1 | 1.8038 % | **99.5519 %** |
| seed 2 | 1.8526 % | **94.9046 %** |
| classification | NOT LEARNABLE | **LEARNABLE** |

Batch 02's negative K result was the residual confound. Its erratum said the
question was untested; it is now answered.

## 2. Classification, all four K, both rules

| K | pre-registered | amended (Addendum 03) | seeds at bar | degenerate | attribution |
|---|---|---|---|---|---|
| 1 | LEARNABLE | **LEARNABLE** | 0, 1, 2 | — | COMPOSITION |
| 2 | PARTIAL | **PARTIAL** | 1 | seed 2 | COMPOSITION |
| 4 | LEARNABLE | **LEARNABLE** | 0, 1, 2 | — | COMPOSITION |
| 8 | LEARNABLE | **LEARNABLE** | 0, 1, 2 | — | COMPOSITION |

No verdict changes under the amended rule. Every attribution is COMPOSITION, on
agreement figures of 25.75–26.63 % against the ≤ 50 % threshold — and for the
cells at ~100 % accuracy the agreement lands on 26.62–26.63 %, which is the
shortcut ceiling itself and exactly what an *exact* solver of F³ must produce.

The bounded-growth gate passed at all four K at 1.0000×, and the post-training
bound held at **1.0000× in all twelve cells**, so every cell is a legitimate
fix-on cell and §8's gating condition is satisfied.

## 3. Exact match is where the result is sharpest

Token accuracy understates what happened. Exact match — every one of the 17 ring
positions correct — separates the K values qualitatively:

| K | seed | token accuracy | **exact match** |
|---|---|---|---|
| 1 | 0 | 64.5393 % | **0.1465 %** |
| 1 | 1 | 43.5375 % | **0.0000 %** |
| 1 | 2 | 57.3788 % | **0.0000 %** |
| 2 | 1 | 99.4083 % | 90.3809 % |
| 4 | 0 | 100.0000 % | **100.0000 %** |
| 4 | 1 | 99.9943 % | 99.9023 % |
| 4 | 2 | 100.0000 % | **100.0000 %** |
| 8 | 0 | 99.9856 % | 99.7559 % |
| 8 | 1 | 99.5519 % | 93.0176 % |
| 8 | 2 | 94.9046 % | 41.1621 % |

**K = 1 gets 43–65 % of tokens right and essentially never a whole ring.** Two of
its three cells score 0.0000 % exact match. K = 4 gets *every ring* right in two
of three cells.

That is not a difference of degree. A model at 57 % token accuracy and 0 % exact
match has not learned F³ and is making many independent local errors; a model at
100 % exact match has learned it. So K = 1 being classified LEARNABLE is an
artefact of the bar being set on token accuracy: it clears 6.67 % comfortably
while failing the task in the sense that matters.

This is the single most important number in the batch and it was invisible in
Batch 02, whose K = 1 cells were the only ones that learned anything at all.

## 4. Takeoff and convergence — and a correction to what I predicted

From the recorded curves. The T = 3 target stage spans steps 15,001–35,000.

| K | crossed 6.67 % at | crossed 26.63 % at | mean takeoff | final slope /1k | still rising |
|---|---|---|---|---|---|
| 1 | 16,000 / 19,750 / 22,500 | 21,000 / 27,000 / 26,750 | **24,917** | +1.4 to +2.7 | **3/3** |
| 2 | 17,000 (1 cell) | 17,750 (1 cell) | 17,750 | ~0 | 0/3 |
| 4 | 14,250 / 15,250 / 14,500 | 15,000 / 15,750 / 15,750 | **15,500** | −0.2 to −0.01 | 0/3 |
| 8 | 12,250 / 15,250 / 17,000 | 13,000 / 17,000 / 19,500 | **16,500** | −0.1 to +1.5 | 2/3 |

**Higher K takes off *earlier*, not later.** K = 4 crosses the shortcut ceiling at
step 15,500 on average and K = 8 at 16,500, against K = 1 at 24,917. Several
K = 4 and K = 8 cells cross it within 1,000 steps of the target stage beginning.

**I predicted the opposite** while the batch was running — I suggested K = 8 might
simply need longer to take off than K = 1, and the Batch 02 erratum §3 records
that as a caveat. That is now measured and false at T = 3 with the fix on: the
higher-K cells take off sooner. The erratum's caveat was reasonable when written
and is superseded by this measurement; it should not be repeated.

**K = 1 is budget-limited and K ≥ 4 has converged.** All three K = 1 cells were
still climbing at step 35,000 at +1.4 to +2.7 points per 1,000 steps, so their
55 % mean is where the budget ran out. K = 4's slopes are flat or slightly
negative because they are at 100 % and have nowhere to go. So a longer budget
would raise K = 1's token accuracy — but §3 suggests it would be raising a
quantity that is not tracking the task.

## 5. The K = 2 anomaly

K = 2 is the reason §7 reports no trend, so it needs stating plainly rather than
explaining away.

| K = 2 | token accuracy | exact match | distinct tokens |
|---|---|---|---|
| seed 0 | 1.8124 % | 0.0000 % | 8 |
| seed 1 | **99.4083 %** | **90.3809 %** | 60 |
| seed 2 | 1.8066 % | 0.0000 % | **1** |

Seed 1 nearly solved the task. Seeds 0 and 2 never left chance, and seed 2 is
**DEGENERATE** — one distinct token, scoring 1.8066 %, which is the T = 3
majority-class floor to four decimals, i.e. the modal token at every position.
That is precisely the artefact Addendum 03 was written to catch, caught.

So K = 2 is **bimodal in takeoff, not limited in capability**: its one cell that
took off reached 90 % exact match, comparable to K = 8's mean. Its mean token
accuracy of 34.34 % describes neither mode and should not be quoted.

Three seeds cannot estimate a takeoff probability. K = 2's 1-of-3 against K = 4's
and K = 8's 3-of-3 is a difference of two cells, which is well inside what three
Bernoulli draws can produce by chance.

## 6. The §7 trend reading, applied mechanically

Learnable set = **{1, 4, 8}**, which is not a prefix of {1, 2, 4, 8}. The
pre-registered function returns:

> **NON_MONOTONE** — "learnable set is not a prefix of the K range; no threshold
> claim is available."

with the recorded instruction to attribute it to seed variance unless the gap
exceeds Batch 02's K = 1 seed spread of 31.2 points. §5 is that attribution, and
it holds: K = 2's capability matches K = 4's and K = 8's.

**The rule works, and it also under-describes the data.** It is keyed on
learnability because three seeds cannot establish monotonicity of an accuracy —
which was right. But it therefore cannot express the finding in §3, that
learnability at the pre-registered bar is satisfied by K = 1 cells that never
produce a correct ring. Reporting only "NON_MONOTONE" would be true and
misleading. Both are given here; neither replaces the other.

## 7. What this does to Batch 02 and its erratum

| claim | status |
|---|---|
| Batch 02 §1–§3: T = 3 learnable at K = 1, not at K = 8 (`fix=none`) | **stands** as a description of that configuration |
| Batch 02 §8: "eight recursion steps cost depth" | withdrawn by its erratum; **now positively explained** — it was the confound |
| Batch 02 erratum §1: whether recursion or the confound caused the K = 8 failure is untested | **answered: the confound** |
| Batch 02 erratum §3: K = 8 may need longer to take off | **superseded and false at T = 3 with the fix on** (§4) |
| Batch 02 §5: the T = 4 budget argument | **strengthened.** K = 1 is still budget-limited at 35,000 steps here too |
| Batch 02 §7.1 point 2: T = 3 discriminates against more recursion | **reversed.** It discriminates *for* more recursion once the stream is bounded |

## 8. What this batch cannot establish

* **Nothing about the depth hypothesis on the benchmark.** T = 3 fails G2 and is
  an evaluation-only depth. This is a diagnostic, not a grid cell, and no number
  here belongs in the Phase 2 grid.
* **Nothing at matched compute.** Steps are matched; at 35,000 steps K = 8 does
  eight times K = 1's block executions. K ≥ 4 winning at equal steps is partly a
  compute statement. This is the batch's largest open confound and §9 addresses it.
* **Nothing about T = 4 or T = 8.** Untouched. The fix has never been tried at the
  primary depths, and that is now the obvious gap.
* **No takeoff probability.** Three seeds per K (§5).
* **Nothing about K = 1's ceiling.** It had not converged.

## 9. The pre-registered action

§8 of the pre-registration, for this outcome:

> Propose, **draft only**, a K comparison protocol at T = 3 with the fix on —
> matched compute rather than matched steps, and a budget chosen from Batch 02's
> takeoff data.

That draft is not written here; this report stops at applying the table, and the
protocol is the next step. Two things §3 and §4 say it must incorporate that the
pre-registration did not anticipate:

1. **Exact match, not token accuracy, as the primary measure.** A bar on token
   accuracy admits K = 1 cells with 0.0000 % exact match.
2. **A budget set from measured takeoff**, which is ~15,500 steps into the target
   stage for K ≥ 4 and beyond 24,917 for K = 1 — so a matched-compute design must
   not starve the low-K arm of the steps it demonstrably needs.

The more consequential follow-up is outside this batch's remit and is recorded
for the user, not chosen here: **the fix has never been run at T = 4.** Batch 01
excluded four explanations for the T = 4 plateau, and the residual confound was
one of them — excluded on the strength of Arms 2b and 2c, neither of which
combined the fix with the curriculum. This batch shows that combination turning a
flat K = 8 into an exact solver at T = 3.

## 10. Compute

| | GPU-h |
|---|---|
| Pre-registered estimate | 4.80 |
| First run before the reboot (9 cells + ~35 min lost) | ≈ 2.70 |
| Resumed run (9 recoveries ≈ 2 min + 3 K = 8 cells) | **2.37** |
| **Total spent** | **≈ 5.07** |

The overage against the estimate is the ~35 minutes of K = 8 seed 0 discarded by
the reboot. The resume saved ≈ 2.1 GPU-h against a clean rerun, at the provenance
cost recorded in §0.

## 11. Integrity record

| | |
|---|---|
| Pre-registration committed before launch | yes — `db39663` |
| Config drift guard against Batch 02 | in the runner; refuses to start on drift |
| Bounded-growth gate before any cell | PASS at all four K, 1.0000× |
| Post-training bound, per cell | held, 1.0000×, 12/12 |
| Amended rule applied in-run | yes |
| Shortcut attribution recorded per cell | yes, all COMPOSITION |
| Deviation | **yes — the reboot and resume, Addendum 01, committed before the resumed run** |
| Checkpoint-vs-log integrity check | 9/9 agree, 0 mismatches |
| Statistic added after seeing data | the per-K maximum, flagged in code and in `curve_analysis.json` |
| Prediction I made and this batch falsified | K = 8 needing longer to take off (§4) |

Batch 03 stops here.
