# Phase 2 — Batch 01 report (Stages 2 and 3)

**Date: 2026-09-30.** Chain completed 00:08:44; every arm exited `rc=0`.

> **Two claims in §5 are withdrawn or replaced by
> [PHASE2_BATCH_01_REPORT_ERRATUM_01.md](PHASE2_BATCH_01_REPORT_ERRATUM_01.md)
> (2026-09-30).** Nothing below is edited and no number or classification
> changes; read that file alongside §5. This notice is the only line added
> after the report was committed.

Classifications are applied mechanically: the pre-registered rule
([PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md) §3) as each arm applied
it to itself while running, and the amended rule
([PHASE2_BATCH_01_ADDENDUM_03.md](PHASE2_BATCH_01_ADDENDUM_03.md)) applied
afterwards in one pass by `scripts/phase2_reclassify.py`. Both are shown; neither
replaces the other.

## 1. Headline

**No arm moved.** Not one of the four reached LEARNABLE at T = 4, and under the
amended rule not one reaches PARTIAL either.

| arm | what it changed against REF | K=1 original → amended | K=8 original → amended | moves |
|---|---|---|---|---|
| 2b | output-side LayerNorm | PARTIAL → **DEGENERATE** | PARTIAL → **DEGENERATE** | no |
| 3 | curriculum over T | NOT LEARNABLE → NOT LEARNABLE | NOT LEARNABLE → NOT LEARNABLE | no |
| 4 | warmup + 2× budget | NOT LEARNABLE → NOT LEARNABLE | NOT LEARNABLE → NOT LEARNABLE | no |
| 2c | non-affine per-step norm | PARTIAL → **NOT LEARNABLE** | PARTIAL → **DEGENERATE** | no |

Both PARTIALs that the original rule produced were artifacts. Every one of them
rested on a cell emitting one or two distinct tokens out of sixty, scoring the
majority-class baseline of 1.9244 % — the defect Addendum 03 was written to
close. Arm 4's and Arm 3's cells were healthy by that measure (49–60 distinct
tokens) and simply sat at chance.

## 2. Arm 1 — the gate

| K | seed | fit exact match | token accuracy | loss | steps |
|---|---|---|---|---|---|
| 1 | 0 | 100.00 % | 100.00 % | 0.004741 | 20,000 |
| 8 | 0 | 100.00 % | 100.00 % | 0.006006 | 20,000 |

Both reached 99 % exact match at step 250, the first evaluation. **Gate passes**,
so the batch was not stopped here and the downstream arms are interpretable.

**What this does not show.** 512 examples is a lookup table well inside a
837,436-parameter model's capacity. This rules out "cannot optimise at this
depth"; it says nothing about whether the general T = 4 function is
representable, which remains untested.

## 3. Arm 2a — the residual confound

| K | step-0 loss, no fix | output-side norm | per-step norm |
|---|---|---|---|
| 1 | 4.4262 | 4.2569 | 4.2569 |
| 2 | 5.5024 | 4.2569 | 4.2574 |
| 4 | 10.1537 | 4.2568 | 4.2573 |
| 8 | **33.9691** | 4.2568 | 4.2569 |

Step-0 loss is a *symptom*. The quantity the confound is about is the hidden
state, and measuring it directly (z_L RMS, first recursion step → last):

| fix | K=8 trajectory | growth |
|---|---|---|
| none | 1.43 → 25.30 | 17.70× |
| output-side | 1.43 → 25.30 | **17.70× — unchanged** |
| per-step (non-affine) | 1.00 → 1.00 | **1.00×** |

**Arm 2b tests read-out normalisation only.** It does not remove the confound
and must not be described as doing so. Only Arm 2c's per-step normalisation
bounds the stream, and its bound held after training as well as before: all six
Arm 2c cells recorded a post-training growth ratio of exactly 1.0000×.

## 4. Per-arm results

### Arm 2b — output-side LayerNorm

| K | seed | best val acc | its bar | final loss | distinct tokens | above (orig) | above (amended) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.7664 % | 1.8889 % | 4.0944 | **2** | no | no |
| 1 | 1 | 1.7578 % | 1.8913 % | 4.0943 | **1** | no | no |
| 1 | 2 | 1.9244 % | 1.9028 % | 4.0944 | **1** | yes | no |
| 8 | 0 | 1.7693 % | 1.8954 % | 4.0944 | **3** | no | no |
| 8 | 1 | 1.7808 % | 1.8942 % | 4.0943 | **1** | no | no |
| 8 | 2 | 1.9244 % | 1.9028 % | 4.0944 | **1** | yes | no |

**K = 1: PARTIAL → DEGENERATE** — **changed**  
**K = 8: PARTIAL → DEGENERATE** — **changed**  

**Moves under the Arm 5 criteria: NO**

### Arm 3 — curriculum over T

| K | seed | best val acc | its bar | final loss | distinct tokens | above (orig) | above (amended) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.8497 % | 1.9031 % | 4.0951 | 60 | no | no |
| 1 | 1 | 1.8095 % | 1.9119 % | 4.0957 | 60 | no | no |
| 1 | 2 | 1.8153 % | 1.9089 % | 4.0956 | 60 | no | no |
| 8 | 0 | 1.9072 % | 1.9143 % | 4.0963 | 50 | no | no |
| 8 | 1 | 1.9014 % | 1.9126 % | 4.0969 | 59 | no | no |
| 8 | 2 | 1.8497 % | 1.9110 % | 4.0981 | 60 | no | no |

**K = 1: NOT_LEARNABLE → NOT_LEARNABLE**  
**K = 8: NOT_LEARNABLE → NOT_LEARNABLE**  

**Moves under the Arm 5 criteria: NO**

### Arm 4 — warmup + 2x budget

| K | seed | best val acc | its bar | final loss | distinct tokens | above (orig) | above (amended) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.7980 % | 1.9119 % | 4.0946 | 60 | no | no |
| 1 | 1 | 1.8440 % | 1.9154 % | 4.0945 | 60 | no | no |
| 1 | 2 | 1.8698 % | 1.9176 % | 4.0945 | 59 | no | no |
| 8 | 0 | 1.8698 % | 1.9184 % | 4.0946 | 52 | no | no |
| 8 | 1 | 1.8411 % | 1.9173 % | 4.0945 | 49 | no | no |
| 8 | 2 | 1.8296 % | 1.9143 % | 4.0946 | 50 | no | no |

**K = 1: NOT_LEARNABLE → NOT_LEARNABLE**  
**K = 8: NOT_LEARNABLE → NOT_LEARNABLE**  

**Moves under the Arm 5 criteria: NO**

### Arm 2c — non-affine per-step norm

| K | seed | best val acc | its bar | final loss | distinct tokens | above (orig) | above (amended) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.9244 % | 1.9028 % | 4.0955 | **1** | yes | no |
| 1 | 1 | 1.8440 % | 1.8977 % | 4.0955 | 4 | no | no |
| 1 | 2 | 1.7808 % | 1.8980 % | 4.0957 | **2** | no | no |
| 8 | 0 | 1.9244 % | 1.9028 % | 4.0955 | **2** | yes | no |
| 8 | 1 | 1.7894 % | 1.8937 % | 4.0955 | **2** | no | no |
| 8 | 2 | 1.7808 % | 1.8942 % | 4.0955 | **1** | no | no |

**K = 1: PARTIAL → NOT_LEARNABLE** — **changed**  
**K = 8: PARTIAL → DEGENERATE** — **changed**  

**Moves under the Arm 5 criteria: NO**

## 5. What each arm isolated, and what it showed

**Arm 2b — read-out normalisation.** Isolates whether normalising the state just
before the vocabulary projection changes learnability. It does not. Worse, every
one of its six cells collapsed to 1–3 distinct tokens, the most degenerate
outcome in the batch. The arm's original PARTIAL was entirely an artifact of two
cells emitting the modal token everywhere.

**Arm 3 — curriculum over T.** Isolates whether a cold-start gradient is the
blocker. The most informative arm in the batch, and the result is sharp. Train
loss at each stage boundary:

| cell | T=1 end | T=2 end | T=4 (first stage) | T=4 end |
|---|---|---|---|---|
| K=1 s0 | 0.0001 | 0.0006 | 4.1070 | 4.0958 |
| K=1 s1 | 0.0000 | 0.0023 | 4.1054 | 4.0959 |
| K=1 s2 | 0.0000 | 0.0001 | 4.1058 | 4.0957 |
| K=8 s0 | 0.0001 | 0.0473 | 4.1044 | 4.0969 |
| K=8 s1 | 0.0000 | **4.1009** | 4.1008 | 4.0968 |
| K=8 s2 | 0.0000 | 0.0164 | 4.1296 | 4.0975 |

Chance is ln 60 = 4.0943. **T = 1 is solved in every cell and T = 2 is solved in
five of six, on freshly generated samples — and then T = 4 sits at chance from
the first stage onward.** This is not a gradual degradation; it is a cliff
between T = 2 and T = 4. Five cells demonstrably learned a depth-2 composition
and none made any progress at depth 4.

The one exception, K=8 seed 1, failed the T = 2 stage too (4.1009), which is
worth noting against any claim that K = 8 has an advantage here.

**Arm 4 — optimiser plateau check.** Isolates whether the plateau is an
optimisation artefact. Warmup over 2,000 steps to an unchanged 1e-3 peak, cosine
decay, double the budget. No effect: all six cells at chance, all with healthy
token diversity (49–60). The plateau is not a step-budget or a schedule problem.

**Arm 2c — per-step normalisation.** Isolates whether bounding the residual
stream changes learnability. It does not, and it made collapse *worse*: five of
six cells emitted 1–2 distinct tokens. This is the trade-off the Addendum 02
source draft predicted — removing the learnable gain removes the recursion's
per-feature rescaling, and the model appears to answer by collapsing. The bound
itself worked exactly as designed (growth 1.0000× before and after training);
what it bought was nothing, and it cost expressiveness.

## 6. Things that contradict expectations, and deviations

**The curriculum's success at T = 2 was not carried forward at all.** The
per-depth diagnostics on the final model read T=1 1.66 %, T=2 1.65 %, T=4 1.85 %
— chance everywhere. A model that provably solved T = 1 and T = 2 mid-run ends
with no trace of either. Whatever the T = 4 stage does, it destroys the earlier
solutions rather than building on them.

**Deviation, recorded.** Addendum 01 §A1.2 specified logging validation accuracy
at T = 1, 2, 4 *at the end of each stage*. The implementation measures the final
model at each depth instead, once, after training. So the numbers above describe
catastrophic forgetting at the end of the run; they do **not** establish what the
model scored at each stage boundary. The train-loss trace in §5 is the evidence
for what each stage achieved, and it is console output rather than a recorded
artifact field.

**The T = 2 shortcut did not appear.** Amendment 01 predicts that a model
learning the depth-1 shortcut instead of depth-2 composition would plateau near
26.7 %. No cell plateaued there — the T = 2 stages reached ~100 % train accuracy,
which is composition, not the shortcut. The risk recorded in advance did not
materialise.

**Arm 1's K = 8 fit as fast as K = 1** despite starting 29.5 nats higher in loss,
so the residual confound does not impede fitting. Combined with Arms 2b and 2c,
the confound now looks irrelevant to this failure in both directions: removing
it changes nothing, and leaving it in place blocks nothing.

## 7. Stage 3 — applying the pre-registered table

The Stage 3 rows are: Arm 1 fails; some arm reaches LEARNABLE; only PARTIAL
anywhere; all arms NOT LEARNABLE.

* Arm 1 did not fail.
* No arm reached LEARNABLE at T = 4.
* Under the amended rule there is **no PARTIAL anywhere** in Batch 01.

**The row that applies is "all arms NOT LEARNABLE"** — with the batch's results
being NOT LEARNABLE or the stricter DEGENERATE. Its action: *report, and draft
(do not finalise) the path-3 write-up outline.*

## 8. Arm 5 — the criteria applied

[PHASE2_BATCH_01_ARM5_CRITERIA.md](PHASE2_BATCH_01_ARM5_CRITERIA.md) §2 approves
Arm 5 only if **≥ 2** of arms {2b, 3, 4, 2c} individually move.

| arm | moves |
|---|---|
| 2b | no |
| 3 | no |
| 4 | no |
| 2c | no |

**Zero arms moved. Arm 5 is not approved and is not run.** This is recorded
rather than inferred: it lands in §4 of the criteria file, "if no arm moves".

The compute constraint would have bound independently — Arms 1–4 plus 2c came to
9.31 GPU-h, and Arm 5's worst case would have reached 13.51, over the 12-hour
limit.

## 9. What is now excluded

Four candidate explanations for the T = 4 plateau have been tested one at a time
against a common reference, and all four are excluded:

1. **Cannot optimise at this depth** — Arm 1 fits a fixed set to 100 %.
2. **The residual-stream confound** — Arms 2b and 2c; removing it changes nothing.
3. **No cold-start gradient** — Arm 3; the curriculum reaches T = 4 with T = 1
   and T = 2 solved and still goes nowhere.
4. **Optimiser schedule or budget** — Arm 4; warmup and 2× steps change nothing.

Memorisation and train-set size were never available as explanations: the
reference already trains on freshly generated samples every step.

**What remains.** The batch's sharpest finding is the cliff between T = 2 and
T = 4 in Arm 3. That is consistent with the P1c signal analysis — all-but-one-cell
subsets predict 26.7 %, all-but-two 13.4 %, all-but-three 7.0 % — meaning the
partial-credit gradient that carries T = 2 has essentially vanished by T = 4. The
batch does not establish that T = 4 is unlearnable at reachable scale; it
establishes that it is not reached by any of the four levers tested.

## 10. Next step is the user's

Per the criteria file §4, the choice among these belongs to the user, and this
report does not make it:

1. **T = 3 in the curriculum** — the cliff is between T = 2 and T = 4, and T = 3
   is the untested point inside it. This is where the Arm 3 result points most
   directly.
2. **A group ladder** — a smaller or solvable group before A₅.
3. **The negative-result write-up** — the Stage 3 action for this row.

Batch 01 stops here.
