# Phase 2 — Batch 02 report: T = 3 is learnable at K = 1 and not at K = 8

**Date: 2026-09-30.** Chain completed 03:23:36, `rc=0`, all six cells at 35,000
steps. Runtime 2 h 40 min 40 s against the pre-registered estimate of
≈ 2 h 45 min (2.72 GPU-h); actual **2.68 GPU-h**.

> **§8's K comparison is corrected by
> [PHASE2_BATCH_02_REPORT_ERRATUM_01.md](PHASE2_BATCH_02_REPORT_ERRATUM_01.md)
> (2026-09-30): this batch ran `fix=none`, so the residual confound was
> uncontrolled on the K axis and the sign of the K effect is unknown, not
> negative.** Nothing below is edited and no number or classification changes;
> §4 of that erratum lists which sections are unaffected. This notice is the only
> line added after the report was committed.

Report: `runs/phase2/batch02/t3_curriculum/t3_curriculum_report.json`.
Curve analysis: `runs/phase2/batch02/t3_curriculum/curve_analysis.json`.
Pre-registered in [PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md),
committed before launch, with
[Addendum 01](PHASE2_BATCH_02_ADDENDUM_01.md) on the per-K combination rule.

## 0. Disclosure: one decision rule was fixed mid-run

Read this before the results.

§6 of the pre-registration keys its outcome table on a single "outcome at T = 3",
but the probe produces one classification per K and §6 never said how the two
combine. [Addendum 01](PHASE2_BATCH_02_ADDENDUM_01.md) closed that gap **while
the probe was running**, adopting the *either-K* convention: the row is
determined by the more favourable K.

At the moment that rule was fixed:

| | |
|---|---|
| K = 1 seed 0 | already reported — **64.0395 %**, above the 6.6667 % bar |
| K = 1 seed 1 | already reported — **49.9914 %**, above the bar |
| K = 1 seed 2 | running, at step 3,000 of 35,000 |
| K = 8, all seeds | not started |
| Any classification | not yet computed; no report file existed |

So two of the three K = 1 seeds were known to be above the bar when the
convention was chosen, and **either-K is the more permissive of the two candidate
conventions**. Its justification is precedent — it is the reading
[PHASE2_BATCH_01_ARM5_CRITERIA.md](PHASE2_BATCH_01_ARM5_CRITERIA.md) §1 already
fixed for Batch 01 ("an arm moves if either of its K values moves") — and not
conservatism.

**It matters to the outcome.** As §7 sets out, the two conventions land on
*different* §6 rows on these numbers. Both are given there, on the same numbers,
so the stricter reading is available to any reader who prefers it. The user
approved keeping either-K with both readings reported.

## 1. Headline

**K = 1: LEARNABLE, 3 of 3 seeds, attributed to composition.**
**K = 8: NOT LEARNABLE, 0 of 3 seeds.**

Neither verdict changes under the Addendum 03 amended rule.

| K | seed | best val T=3 | loss | distinct tokens | shortcut agreement |
|---|---|---|---|---|---|
| 1 | 0 | **64.0395 %** | 1.1401 | 60 | 18.11 % |
| 1 | 1 | **49.9914 %** | 1.6723 | 60 | 14.54 % |
| 1 | 2 | **81.2127 %** | 0.5836 | 60 | 22.06 % |
| 8 | 0 | 1.8411 % | 4.0976 | 58 | 1.80 % |
| 8 | 1 | 1.8038 % | 4.1009 | 60 | 1.80 % |
| 8 | 2 | 1.8526 % | 4.0984 | 51 | 1.73 % |

Against uniform chance 1.6667 %, the T = 3 majority-class floor **1.8066 %**, the
LEARNABLE bar **6.6667 %**, and the shortcut ceiling **26.6257 %**. Chance loss is
ln 60 = 4.0943.

**This is the first result in the project in which the T axis and the K axis both
move.** Every previous Phase 2 result was flat on both.

## 2. Classification, both rules

| K | pre-registered | amended (Addendum 03) | changed | attribution |
|---|---|---|---|---|
| 1 | LEARNABLE | **LEARNABLE** | no | **COMPOSITION** |
| 8 | NOT LEARNABLE | **NOT LEARNABLE** | no | n/a |

K = 1: all three seeds meet the learnable bar and all three are above chance under
both rules. None is degenerate — every cell uses all 60 tokens. K = 8: no seed
above chance under either rule, and **none degenerate either** (58, 60 and 51
distinct tokens), so this is a genuine attempt that sat at chance, not a collapse.
That distinction is the one Addendum 03 §3.2 exists to draw, and it favours the
stronger reading: the task resisted K = 8 rather than K = 8 failing to run.

## 3. The shortcut check, which is what makes K = 1's result readable

G2 fails at T = 3: `window_product_forward` = s[i]·s[i+1]·s[i+2]·s[i+3] agrees
with F³ on exactly 26.67 % of tokens, wherever (s[i+1]·s[i+2])² = e. The
LEARNABLE bar is 6.67 %, so a shortcut-only model would have been classified
LEARNABLE. Accuracy alone therefore could not have read this probe.

Agreement with the shortcut separates the hypotheses, and the observed values sit
on the composition prediction to within 0.6 points in all three cells:

| K=1 seed | accuracy | agreement observed | predicted if composing | predicted if shortcut-only |
|---|---|---|---|---|
| 0 | 64.04 % | **18.11 %** | 17.5 % | ~100 % |
| 1 | 49.99 % | **14.54 %** | 13.9 % | ~100 % |
| 2 | 81.21 % | **22.06 %** | 21.9 % | ~100 % |

(The prediction is p·0.2667 + (1−p)(1−0.2667)/59 for accuracy p: the shortcut and
a correct model coincide where the shortcut happens to be right.) All three are
attributed **COMPOSITION** under the pre-registered ≤ 50 % threshold, with a
73-point margin to the SHORTCUT threshold. **K = 1 learned depth-3 composition,
not the shallow product.**

The same diagnostic independently confirms a Batch 01 claim that was previously
only inferred. At the T = 2 stage boundary, every cell that solved T = 2 shows
shortcut agreement of **26.74 %** — the exact composing signature. Batch 01
argued its T = 2 stages learned composition from the *absence* of a 26.7 %
plateau; this measures it, in both K values.

## 4. Takeoff and slope — 65 % is a budget limit, not a plateau

From the recorded curves; no new runs. The T = 3 target stage spans steps
15,001–35,000.

| K | seed | crossed 6.67 % at | crossed 26.63 % at | slope over last 2,500 steps | still rising |
|---|---|---|---|---|---|
| 1 | 0 | 19,750 | 27,750 | **+3.453** /1k | **yes** |
| 1 | 1 | 17,250 | 23,000 | **+1.051** /1k | **yes** |
| 1 | 2 | 15,750 | 18,250 | **+1.293** /1k | **yes** |
| 8 | 0 | never | never | +0.010 /1k | no |
| 8 | 1 | never | never | +0.012 /1k | no |
| 8 | 2 | never | never | −0.014 /1k | no |

Two findings.

**K = 1 had not converged.** All three cells were still climbing at step 35,000,
at +1.05 to +3.45 accuracy points per 1,000 steps. The mean best of 65.08 % is
where the budget ran out, not where learning stopped. A longer budget would very
likely give a higher number, and **the 64/50/81 % spread across seeds is
plausibly a spread in takeoff time rather than in final capability** — seed 2
took off first (15,750) and scored highest (81.21 %), seed 0 took off last
(19,750) and scored lowest of the three.

**Takeoff needs 750–4,750 steps *into* the target stage**, and reaching the
shortcut ceiling needs 3,250–12,750 steps into it.

## 5. What this does to Batch 01's T = 4 conclusion

This is the most consequential part of the report, and it cuts against the
previous batch.

Batch 01's Arm 3 gave T = 4 a 5,000-step ladder stage plus a 20,000-step target
stage and found it at chance throughout, and
[PHASE2_BATCH_01_REPORT.md](PHASE2_BATCH_01_REPORT.md) §9 listed "no cold-start
gradient" among four excluded explanations. But T = 3 under the identical
schedule needed up to 4,750 steps merely to cross the bar and was still rising
20,000 steps in. **The budget that declared T = 4 at chance is one that T = 3
shows can be marginal at a depth that is in fact learnable.**

So the Arm 3 exclusion should be weakened. It remains true that the curriculum as
pre-registered did not reach T = 4. It is no longer safe to say the curriculum
*cannot*, because the natural follow-up — curriculum plus a longer budget at the
target depth — has never been run at T = 4. Batch 01 §9's fourth exclusion
("optimiser schedule or budget", Arm 4) tested 2× budget **without** a
curriculum; the combination was never tested.

The interpretation recorded in
[the erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) §3 survives and is now
supported rather than merely asserted: it predicted that erasure should be weaker
where partial credit is worth more, and T = 3 — where the partial-credit ceiling
is higher than at T = 4 — is where learning happens.

## 6. Stage boundaries: both K values solve T = 1 and T = 2, and forget them

The fix for Batch 01's recorded deviation. Validation accuracy at every depth at
every stage boundary, as a report field.

| cell | step 5,000 (ends T=1) | step 10,000 (ends T=2) | step 15,000 (ends T=3 rung) | step 35,000 (ends target) |
|---|---|---|---|---|
| K=1 s0 | T1 **100.00 %** | T2 **100.00 %** | T3 1.70 % | T3 **64.04 %** |
| K=1 s1 | T1 **100.00 %** | T2 **99.99 %** | T3 3.43 % | T3 **49.99 %** |
| K=1 s2 | T1 **100.00 %** | T2 **100.00 %** | T3 3.29 % | T3 **81.21 %** |
| K=8 s0 | T1 **100.00 %** | T2 **99.99 %** | T3 1.65 % | T3 1.61 % |
| K=8 s1 | T1 **100.00 %** | T2 1.70 % — failed | T3 1.72 % | T3 1.69 % |
| K=8 s2 | T1 **100.00 %** | T2 **99.43 %** | T3 1.64 % | T3 1.51 % |

**K = 8 is not failing the curriculum.** It solves T = 1 in 3 of 3 and T = 2 in 2
of 3 — the same as K = 1 on T = 1, and one seed short on T = 2. The K = 8 failure
is specific to T = 3. Any explanation that says K = 8 could not optimise, or
could not run the schedule, is excluded by this table.

**Forgetting is total, at every transition, in both K values.** T = 1 falls from
100.00 % to ~1.54 % during the T = 2 stage. T = 2 falls from ~100 % to ~1.6 %
during the T = 3 rung. It happens in the cells that go on to solve T = 3 just as
much as in those that do not, so **catastrophic forgetting is not the mechanism
of the K = 8 failure** — K = 1 forgets identically and still learns T = 3.

The 5,000-step T = 3 rung leaves every cell at chance (1.64–3.43 %); all the
learning happens in the 20,000-step target stage.

## 7. Applying the §6 outcome table

Mechanically, and under both conventions.

**Under the recorded either-K convention (Addendum 01).** K = 1 is LEARNABLE with
attribution COMPOSITION, so the row is:

> **LEARNABLE, attribution COMPOSITION** — Propose an amendment making
> T ∈ {1, 2, 3} the depth range for the K comparison. **Draft only** — written,
> committed as a proposal, not adopted and not executed.

**Under the stricter both-K convention.** K = 8 is NOT LEARNABLE, so the row
would be:

> **NOT LEARNABLE** — The learnability boundary is T = 2. Proceed to the path-3
> write-up.

**The conventions disagree, and the disclosure in §0 is what a reader needs to
weigh that.** The recorded rule selects the first row. This report does not
execute it: per the user's instruction, the table is applied and the batch stops
here.

### 7.1 Two problems the amendment draft must confront

Recorded now so they cannot be skipped later. The pre-registration already
required the first.

1. **Two of the three depths in T ∈ {1, 2, 3} fail G2.** T = 2 fails at 26.67 %
   (`s[i]·s[i+2]`) and T = 3 fails at 26.67 % (`window_product_forward`), by the
   same involution mechanism. Only T = 1 is clean, and T = 1 is the primitive.
   A depth range for the K comparison in which two of three depths admit a
   26.7 % shallow shortcut is not the same object as the primary range
   T ∈ {4, 8}, both of which pass G2.
2. **T = 3 is the only depth in that range that discriminates between K values,
   and it discriminates against more recursion.** T = 1 is solved by both K at
   100 %; T = 2 is solved by both (5 of 6 cells); T = 3 is solved by K = 1 and not
   by K = 8. So a K comparison over T ∈ {1, 2, 3} rests entirely on a depth where
   the answer is that **K = 8 is worse**.

## 8. The K comparison — descriptive only

| K | cells | mean best acc. | mean shortcut agr. | mean takeoff (26.63 %) | mean slope /1k | still rising |
|---|---|---|---|---|---|---|
| 1 | 3 | **65.08 %** | 18.24 % | 23,000 | **+1.932** | 3 / 3 |
| 8 | 3 | 1.83 % | 1.78 % | — | +0.003 | 0 / 3 |

**Descriptive, and it must not be read as a test of the depth hypothesis.** The
reasons, all of which were true before the numbers were seen:

* A curriculum probe is **not a grid cell**. Nothing here is part of the Phase 2
  grid and T = 3 remains an evaluation-only depth that fails G2.
* Three seeds per K cannot separate a K effect from seed variance, and the K = 1
  spread (49.99–81.21 %) is wide.
* **The budget is matched in steps, not in compute.** At 35,000 steps K = 8
  performs eight times the block executions of K = 1. This is not a
  matched-compute comparison, and K = 8 is the arm that got more computation and
  did worse.
* K = 1 had not converged (§4), so the gap's size is not a converged quantity.

What can be said, plainly: **on this benchmark, at this width, under this
schedule and at equal step budget, eight recursion steps did not buy depth — it
cost it.** The project's hypothesis is that shared-weight recursion buys
computational depth without parameters. The first condition in which that could
have shown up produced the opposite sign. That is one data point from six cells
and it is not a refutation, but it is the first evidence on the question, and it
is negative.

## 9. What this probe does not establish

* **Nothing about T = 4.** It does not make T = 4 learnable. It does weaken the
  claim that the curriculum cannot reach it (§5), which is a statement about
  Batch 01's evidence, not a new result about T = 4.
* **Nothing about the shape of the curve.** Three measured depths do not
  establish a shape; the "cliff" framing stays withdrawn
  ([erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md) §2).
* **Nothing about T = 3 as a benchmark depth.** It fails G2.
* **Nothing about the depth hypothesis**, per §8.
* **Nothing about K = 8's ceiling.** K = 8 was flat, not falling; a different
  width, schedule or budget was not tried.

## 10. Integrity record

| | |
|---|---|
| Pre-registration committed before launch | yes — `ef2e7dd`, before the systemd unit existed |
| G2 at T = 3 measured before the rules were written | yes — task-level, no model, artifact at `runs/phase2/gates_t3/` |
| T = 3 floors measured before the probe ran | yes — majority class 1.8066 %, shortcut ceiling 26.6257 % |
| Shortcut check added before the probe ran | yes, and it can only refuse a positive claim |
| Amended rule applied in-run | yes, unlike Batch 01's post-hoc pass |
| Per-K combination rule | fixed mid-run — **see §0** |
| Deviations | none in this batch; Batch 01's recorded deviation is fixed (§6) |
| Smoke run | recorded, `"smoke": true`, separate directory, not classified |

One rule was fixed with partial results in hand, and §0 states which cells were
known and by how much. Everything else was fixed before the data existed.

## 11. Next step is the user's

Batch 02 stops here. The recorded row calls for an amendment **draft only**, and
§7.1 records the two problems any such draft has to confront. The alternatives
that the results now make live:

1. **Curriculum plus a longer target budget at T = 4** — the combination Batch 01
   never tested, which §5 shows is no longer excluded.
2. **A longer budget at T = 3** — K = 1 had not converged, so the achievable
   accuracy at depth 3 is unknown.
3. **A matched-compute K comparison** — the §8 caveat is fixable by equalising
   block executions rather than steps.
4. **The path-3 write-up** — now materially different, since it has a positive
   result at T = 3 and a negative K result to report rather than flatness
   everywhere.

This report does not choose among them.
