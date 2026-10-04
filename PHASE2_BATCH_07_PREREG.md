# Phase 2 — Batch 07 pre-registration: is K = 8's T = 4 result depth or compute?

**Date: 2026-10-04. Committed before the batch runs.**

Supersedes [PHASE2_BATCH_07_T4_DRAFT.md](PHASE2_BATCH_07_T4_DRAFT.md), which is
left as committed, and which carries the derivation of the arm shapes and the
reuse argument in full. The earlier T = 3 draft
([PHASE2_BATCH_07_DRAFT.md](PHASE2_BATCH_07_DRAFT.md)) remains superseded as the
next batch and not abandoned.

Verified immediately before committing: `runs/phase2/batch07/` does not exist, no
`t4_family_report.json` exists anywhere, and `systemctl --user` lists no
`tesseract-chain@07` unit.

**Approved at ≈ 15.1 GPU-h for the main run on 2026-10-04**, with the conditional
follow-up (§7) approved if triggered at ≈ 12.6 GPU-h, both over the standing
8-hour threshold.

## 1. The question

Batch 06 solved T = 4 — a **primary** benchmark depth — at K = 8 in 3 of 3 seeds
(98.78 / 90.43 / 84.13 % exact match), with steps matched and **compute not**. At
90,000 steps K = 8 performs eight times K = 1's block executions, so that result is
consistent with *depth* or with *eight times the FLOPs*. This batch holds forward
compute fixed and varies how it is spent.

## 2. The three arms

Measured from the instantiated models; forward compute matches to **0.004 %**:

| arm | family | parameters | forward FLOPs / seq | block calls | d_ff |
|---|---|---|---|---|---|
| **A** | `tesseract` K = 8 | **837,436** | 215,615,488 | 8 | 1024 |
| **B** | `width_scaled` → K = 8 compute | 6,383,992 | 215,624,192 | 1 | **11,836** |
| **C** | `unrolled` L = 8 | 6,365,756 | 215,615,488 | 8 | 1024 |

**Arm A is 7.62× smaller than B and 7.60× smaller than C** — twice Batch 05's
ratio, because the shared block's count is K-invariant while the alternatives'
grows with depth.

### 2.1 Arm A: three cells reused, one reproduced

Batch 06's K = 8 cells are bit-identically configured to arm A, so **they are
reused** (`source: reused_from_batch06`), and **one fresh cell at seed 0 is trained
as a reproduction check** (`source: trained`). Approved as recommended in the
draft §2.1.

The check is pass/fail and pre-registered: **the fresh cell's T = 4 exact match
must land within 0.01 points of 98.7793 %.** Outside that, the environment has
drifted, the reuse is invalid, and §6's last row applies. Reproducibility has been
demonstrated before — Batch 05's arm A matched Batch 03 to four decimals — so this
is insurance, not a prediction.

## 3. Configuration

Batch 06's ladder and budget **unchanged**, which is what makes arm A reusable:

| | value |
|---|---|
| Depth | T = 4 |
| Ladder | T = 1 (5,000) → T = 2 (5,000) → T = 3 (15,000) → T = 4 (5,000) |
| Target stage | T = 4, 60,000 steps |
| Total per cell | **90,000** |
| Boundaries | 5,000 / 10,000 / 25,000 / 30,000 / 90,000 |
| Residual fix | non-affine per-step normalisation, all three arms |
| Optimiser | AdamW, weight decay 0.01, batch 128 |
| **Learning rate** | **1e-3 in phase 1; 3e-4 in the conditional phase 2 (§7)** |
| Seeds | 3 per arm |
| Primary metric | **exact match** |
| Eval interval | every 250 steps |
| **Evaluation batch** | **512, not the 2,048-example split in one pass (§3.1)** |

### 3.1 The evaluation batch is 512, for memory

Arm B's `d_ff` of 11,836 needs **3.33 GiB** to evaluate the 2,048-example split in
one forward pass and **OOMs on this 3.68 GiB card**; at 512 the peak is 0.88 GiB.
Evaluation is chunked, so this is a memory setting rather than a change to what is
measured — and it was verified rather than assumed:

| arm A seed 0, Batch 06 checkpoint | eval batch 2048 | 1024 | 512 | 256 |
|---|---|---|---|---|
| exact match | 98.7793 % | 98.7793 % | 98.7793 % | 98.7793 % |
| token accuracy | 99.9282 % | 99.9282 % | 99.9282 % | 99.9282 % |
| loss | 0.003027 | 0.003027 | 0.003027 | 0.003027 |

Identical throughout. **The one case where it is not equivalent** is a cell sitting
at chance: there the logits are near-tied, so a different reduction order can flip
an argmax, which can move a distinct-token count marginally. That affects only the
Addendum 03 collapse check, whose role is to *refuse* a positive claim, so it
cannot manufacture one. The same batch is used for every arm, including arm A's
reused cells, which are re-evaluated at it rather than carried over.

**The resolved optimiser configuration is recorded per cell** in the report —
optimiser, learning rate, weight decay, batch size. [Batch 05
Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md) §4 records that no batch report has
ever held these; this batch varies the rate across phases, so the artifact would be
unreadable without it.

A partial report is written after every cell.

## 4. The bounded-growth gate

Per-step hidden-state RMS measured for all three arms with the fix applied;
growth must be below **1.05** or no cell trains. The post-training bound is
recorded per cell and gates every cross-arm conclusion. Batch 05 established that
the fix applies to `unrolled` at all; this re-verifies it at L = 8.

## 5. The corrected plateau test — a pre-registered refinement

Batch 06 §3 showed the previous test — slope ≥ 0.1 points per 1,000 steps over the
last 2,500 — misclassifying all three K = 8 cells as "still rising" when their
accuracy was oscillating without trend near the ceiling. **Accepted for Batch 07
onward and explicitly not retroactive: Batch 06's recorded outcome stands as
published.**

> **A cell is still rising iff its slope over the last 2,500 steps is ≥ 0.1 points
> per 1,000 steps AND its t-statistic exceeds 2** (slope > 2 × standard error).

**Both conjuncts are load-bearing**, and each alone misclassifies a real cell:

| cell | slope /1k | t | magnitude alone | t-test alone | **conjunction** |
|---|---|---|---|---|---|
| B06 K = 8 s0 | +0.322 | 1.78 | rising ✗ | plateaued ✓ | **plateaued ✓** |
| B06 K = 1 s1 | +0.037 | 2.31 | plateaued ✓ | rising ✗ | **plateaued ✓** |
| B06 K = 1 s2 | +0.403 | 12.51 | rising ✓ | rising ✓ | **rising ✓** |
| B05 arm C s1 | +17.694 | 15.12 | rising ✓ | rising ✓ | **rising ✓** |

Verified against **all 18 cells of Batches 05 and 06**; the conjunction classifies
every one as its trajectory warrants, and
`tests/phase2/test_batch07.py` pins the real cells as fixtures.

The refinement is **one-directional**: adding a conjunct can only reclassify a
noise-driven "rising" as "plateaued", never the reverse. The other three forms
recorded in Batch 06 §3 were rejected — a headroom-relative threshold moves the
wrong way at 99.93 % accuracy, a longer window changes the quantity estimated, and
exact match faces the same noise in a steeper metric.

## 6. Phase 1 outcome table

Keyed on **best exact match per arm**, with Batch 05's 5-point difference
threshold and its escalation clause, and the §5 plateau test.

| pattern | reading |
|---|---|
| **A ahead of both B and C** (> 5 points over each) | **Depth, and weight sharing helps.** Stronger than the hypothesis predicts, at 7.6× fewer parameters. *This row exists because Batch 05's table lacked it and returned UNRESOLVED when the pattern occurred.* |
| **A ≈ C, both > B** | **Depth, not compute.** Eight block executions beat one wide layer at equal compute; sharing costs nothing. |
| **A ≈ B ≈ C** | **Compute, not depth.** Batch 06's K = 8 result was the eight-fold FLOPs. |
| **C > A** (> 5 points) | **Depth needs unshared parameters.** |
| **B > A** (> 5 points) | **Width beats depth at equal compute.** A negative result, reported as one. |
| every arm in the 5–95 % band | **UNRESOLVED** — report and escalate. |
| **Any arm still rising** under §5 | Reported as still rising; its accuracy is **not** an achieved level and no comparison involving it is drawn. |
| **Not all arms at convergence** | Per-arm results with slopes and t-statistics; **no cross-arm ordering claimed.** |
| **Arm A's reproduction cell misses 98.7793 % by > 0.01** | The environment has drifted. **Batch failure**; the reuse is invalid and no comparison is drawn. |
| Any arm DEGENERATE or below the 1 % exact-match floor | Learned nothing usable; excluded with the reason recorded. |
| Bounded-growth gate fails, or a cell's post-training growth ≥ 1.05 | Not a fix-on arm; excluded. |

## 7. The conditional learning-rate follow-up

1e-3 was inherited through six batches from a shared-block model at 837,436
parameters; arms B and C hold **7.6×** that ([Batch 05
Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md)). The check is therefore **conditional
and pre-registered**, not dropped:

> **Phase 2 runs arms B and C at lr 3e-4, three seeds each, automatically and only
> if phase 1's reading is `A_AHEAD_OF_BOTH` or `DEPTH_NOT_COMPUTE`** — the two
> readings in which arm A beats both, and so the two whose interpretation the
> inherited rate would undermine.

It does **not** run on a reading that already goes against arm A: there is nothing
for a rate check to rescue. It also does not run if phase 1's comparison was
withheld for non-convergence — spending 12.6 GPU-h requires knowing what is being
checked.

### 7.1 How the follow-up is read

| pattern | reading |
|---|---|
| Either alternative at 3e-4 comes within 5 points of arm A | **`RATE_CONFOUNDED`.** Phase 1's ordering was a learning-rate artefact. A proper rate search — not this two-point check — is the prerequisite for any arm comparison, and phase 1's conclusion is withdrawn. |
| 3e-4 improves either alternative by > 5 points without reaching arm A | **`RATE_MATTERS_BUT_ORDERING_HOLDS`.** Report both, and that the gap is smaller than phase 1 implied. |
| 3e-4 changes neither materially | **`RATE_DOES_NOT_EXPLAIN_THE_GAP`** at this second point. |

**Two points cannot find an optimum.** Whatever phase 2 returns, the conclusion
carries that limit; a `RATE_CONFOUNDED` result means a search is needed, not that
3e-4 is right.

## 8. Thermal instrumentation

`nvidia-smi` temperature, clocks, power and any throttle reasons are logged **every
60 seconds** to `runs/phase2/batch07_thermal.csv` for the run's duration, and the
report states **whether throttling explains the evaluation overhead**.

The overhead needs explaining. A fixed eval interval should give a constant
fractional overhead, and it does not:

| batch | steps/cell | wall clock | overhead over predicted training |
|---|---|---|---|
| Batch 04 | 40,000 | 4 h 38 m | **+5.2 %** |
| Batch 06 | 90,000 | 12 h 13 m | **+23.4 %** |

Sustained-load throttling on a laptop GPU is the hypothesis; this batch is the
first to measure it rather than infer it.

**It may also bear on the unexplained hard stop of 2026-09-30.** That boot's
journal ends abruptly at 21:36:22 with no shutdown sequence, no thermal event and
no MCE recorded, during a long sustained-load run; the cause was never
established. If this batch shows temperatures approaching the limit, that becomes
a candidate. If it does not, the hard stop stays unexplained and thermal causes are
weakened as an explanation. Either way the record gains a measurement where it
currently has none.

## 9. Compute

Per-step times **measured on this machine** on 2026-10-04 with the fix applied at
T = 4, medium width, batch 128, deterministic kernels, 120 timed steps after 20
warmup:

| arm | s/step | per cell (90,000) |
|---|---|---|
| A `tesseract` K = 8 | 0.08093 | 121.40 min |
| B `width_scaled` | **0.05213** | 78.19 min |
| C `unrolled` L = 8 | 0.08395 | 125.93 min |

The draft extrapolated arm B at 0.0581 from Batch 05's ratio; measured it is
**0.05213**, 11 % cheaper — a wider matmul utilises the GPU better than the ratio
implied, as the draft suspected.

**Phase 1** — 7 trained cells (arm A reproduction ×1, B ×3, C ×3):

| | GPU-h |
|---|---|
| Training | 12.23 |
| Overhead at the long-run rate (+23.4 %, §8) | 2.86 |
| Bounded-growth gate | ~0.02 |
| **Phase 1 total** | **≈ 15.11** |

**Phase 2, if triggered** — 6 cells (B ×3, C ×3 at 3e-4):

| | GPU-h |
|---|---|
| Training | 10.21 |
| Overhead (+23.4 %) | 2.39 |
| **Phase 2 total** | **≈ 12.60** |

| | GPU-h |
|---|---|
| Phase 1 only (if the trigger does not fire) | **≈ 15.11** |
| **Both phases (if it does)** | **≈ 27.71** |

Reusing arm A's three cells saves ≈ 7.5 GPU-h against training them. Disk: arms B
and C hold ~6.4 M parameters, so ~25.6 MB per checkpoint; 13 cells across both
phases ≈ 650 MB, plus curves ≈ 25 MB.

## 10. What this batch cannot establish

* **No learning-rate optimum** (§7.1). Two points at most.
* **Nothing about total training compute.** Forward FLOPs are matched; arms B and C
  hold 7.6× the parameters, so their backward pass and optimiser state cost far
  more. A reader wanting total compute matched does not get it here.
* **Nothing about why K = 4 collapsed** in Batch 06
  ([Note 01](PHASE2_BATCH_06_REPORT_NOTE_01.md)); K = 4 is not an arm.
* **Nothing about T = 8**, the other primary depth.
* **No takeoff probability.** Three seeds per arm, and this project has repeatedly
  seen cells at identical settings be bimodal.
* **Nothing about the grid**, which trains at the target depth from scratch with no
  curriculum. A curriculum result is evidence for *amending* its protocol.
* **Nothing about arm A at other rates.** It keeps 1e-3 by design, so the rate
  check is offered only to the alternatives — an asymmetry recorded rather than
  hidden.

## 11. Execution

```
systemctl --user enable tesseract-chain@07.service
systemctl --user start --no-block tesseract-chain@07.service
```

`enable` then `start --no-block` rather than `enable --now`: with a `Type=oneshot`
unit, `--now` blocks until ExecStart finishes, which on previous batches tied up
the launching shell for the whole multi-hour run.

This file is committed before those commands are run.
