# Phase 2 — Batch 05 pre-registration: depth or compute?

**Date: 2026-10-01. Committed before the batch runs.**

Supersedes [PHASE2_BATCH_05_DRAFT.md](PHASE2_BATCH_05_DRAFT.md), which is left as
committed. The draft's §4.1 blocker is resolved by implementing option 1; §3
below records what was implemented and measured.

Verified immediately before committing: `runs/phase2/batch05/` does not exist, no
`t3_family_report.json` exists anywhere, and `systemctl --user` lists no
`tesseract-chain@05` unit.

## 1. The question

Batch 03 found that at T = 3 with the stream bounded, K = 4 solves the task
exactly while K = 1 reaches 43–65 % token accuracy and ~0 % exact match. Batch 03
§8 records why that is incomplete:

> **Nothing at matched compute.** Steps are matched; at 35,000 steps K = 8 does
> eight times K = 1's block executions.

Two claims survive that result and only one is the project's hypothesis:

1. **Depth** — applying one block four times composes something a single
   application cannot, without growing the parameter count.
2. **Compute** — four times the forward FLOPs is what did it, and any model
   spending that much would do as well.

**Batch 05 holds forward compute fixed and varies how it is spent.**

## 2. The three arms

All at T = 3, all with the non-affine per-step fix on, all with Batch 03's
curriculum and budget. Parameters and FLOPs are **measured from the instantiated
models**, not computed by hand:

| arm | family | parameters | forward FLOPs / sequence | block executions | d_ff |
|---|---|---|---|---|---|
| **A** shared recursion | `tesseract`, K = 4 | **837,436** | 108,068,864 | 4 | 1024 |
| **B** width-scaled | `width_scaled`, matched to K = 4 | 3,214,678 | 108,077,568 | 1 | **5658** |
| **C** unrolled | `unrolled`, L = 4 | 3,206,716 | 108,068,864 | 4 | 1024 |

* Forward compute matches to within **0.008 %** (arm C exactly; arm B is +8,704
  FLOPs on 108.07 M, the granularity of the integer `d_ff` search).
* **Arm A uses 3.84× fewer parameters than B and 3.83× fewer than C.**
* For reference, K = 1 at this width is 837,436 parameters and **27,408,896**
  FLOPs — a quarter of the compute of all three arms. It is not an arm; it is what
  makes the matching meaningful.

### 2.1 What each comparison isolates

* **A vs B** — same compute, spent on four applications of a narrow block versus
  one application of a wide one. This is the depth question.
* **A vs C** — same compute *and* same number of block executions, differing only
  in weight sharing. Isolates sharing from depth.
* **B vs C** — near-identical parameter counts at identical compute, differing in
  whether the compute is one wide layer or four narrow ones. A control.

All three are reported. They are not independent and no one of them is selected
after the fact.

## 3. The draft's blocker, resolved

[PHASE2_BATCH_05_DRAFT.md](PHASE2_BATCH_05_DRAFT.md) §4.1 found
`apply_per_step_norm` **refusing** on the unrolled family, which had **4.9777×**
hidden-state growth across four blocks — far outside the 1.05 gate and on exactly
the axis arm C controls for. Running A bounded against C unbounded would have
reproduced the defect that invalidated Batch 02's K comparison.

**Option 1 was implemented, test-first.** `phase2/batch01.py` gains
`apply_per_step_norm_unrolled`, and `apply_per_step_norm` dispatches to it so a
caller cannot bound one family and silently skip another.

Measured at medium width with the T = 3 validation inputs:

| arm | growth before | growth after | parameters added |
|---|---|---|---|
| A `tesseract` K = 4 | 5.0099× | **1.0000×** | **0** |
| B `width_scaled` | 1.0000× (1 step) | 1.0000× | **0** |
| C `unrolled` L = 4 | **4.9777×** | **1.0000×** | **0** |

Four properties are pinned by `tests/phase2/test_unrolled_per_step_norm.py`,
written before the implementation:

1. **It is the same fix.** The unrolled version reproduces
   `PerStepNormalisedReasoner` step for step, including the ordering — z_H is
   formed from the **un-normalised** z_L and then normalised. Taking z_H from the
   normalised z_L is a different function, and arms A and C would then be under
   two different fixes, making any A-vs-C difference unattributable. My first
   draft of the test had this backwards and was corrected against the reference.
2. **Zero parameters.** Non-affine, so arm C's count does not move and the
   comparison with arm B is unaffected.
3. **Existing blocks are reused**, not rebuilt — a rebuilt block would
   reinitialise weights.
4. **Both states are bounded at every step**, at every depth in {1, 2, 4, 8}.

**Arm B's bound is vacuous, not earned.** It executes one block, so there is no
between-step growth to bound; its 1.0000× is trivially true and is reported as
such. A test pins that it records exactly one recursion step.

### 3.1 The bounded-growth gate

Runs before any training cell, across **all three arms**, requiring growth below
**1.05**. If any arm fails, no cell runs. The post-training bound is recorded
**per cell** and gates every cross-arm conclusion: an arm whose bound broke
during training is not a fix-on arm.

## 4. Configuration

| | value |
|---|---|
| Depth | T = 3 |
| Curriculum | T = 1 → 2 → 3, 5,000 steps per rung, then 20,000 at the target |
| Total steps per cell | **35,000 — identical to Batch 03**, so arm A is directly comparable to Batch 03's K = 4 cells |
| Seeds | 0, 1, 2 |
| Width | medium base; B and C derive their shapes from it |
| Residual fix | non-affine per-step normalisation, all three arms |
| Optimiser | AdamW, constant 1e-3, weight decay 0.01 |
| Batch / eval interval | 128 / every 250 steps |
| **Primary metric** | **exact match** |
| Secondary | token accuracy, loss, distinct tokens, shortcut agreement |
| Classification | the standing rule, plus Addendum 03's floor (T = 3: **1.8066 %**) and collapse check |
| Shortcut attribution | **yes** — T = 3 fails G2 at 26.67 %, so every arm gets it |

Arm A is a replication of Batch 03's K = 4 cells under identical conditions. That
is deliberate: if arm A does not reproduce ~100 % exact match, the batch has a
problem and §7's last row applies.

## 5. The "≈" threshold, with the argument

The outcome table turns on whether two arms differ. A bare number would be
arbitrary, so here is the reasoning and then the number.

**Within-arm variation is bimodal, not Gaussian.** Across Batches 03 and 04, a
cell either takes off or does not, and the within-arm spread depends entirely on
which:

| arm | cells | exact match | within-arm spread |
|---|---|---|---|
| Batch 03, K = 4, T = 3 | all 3 took off | 100.00 / 99.90 / 100.00 | **0.10 points** |
| Batch 03, K = 2, T = 3 | 1 of 3 took off | 0 / 90.38 / 0 | 90.4 points |
| Batch 04, K = 4, T = 4 | 1 of 3 took off | 0 / 0 / 12.89 | 12.9 points |

A single margin cannot serve both regimes. **So the comparison is split in two**,
and only the first carries a threshold:

**(a) Capability — best exact match per arm.** Among cells that take off, the
observed within-arm spread was **0.10 points**. A threshold of **5 points** is
**50× that spread**, so a difference exceeding it cannot plausibly be seed noise
among takeoffs. It is also small against the effect the batch is looking for:
at T = 3 the gap between a solving and a non-solving configuration was 100 % vs
0.15 %, so 5 points is comfortably inside the signal. Ten points would risk
calling an 8-point real difference "≈"; one point would be only 10× the spread
and invites arguing about measurement noise. **Arms differ iff their best exact
match differs by more than 5 points.**

**(b) Reliability — takeoff count per arm.** Reported, never thresholded.
Batch 03 showed K = 2 at 1-of-3 and K = 4 at 3-of-3 with *identical* capability
(90.4 % and 100 % exact match), so a takeoff-count difference is not a capability
difference. With three seeds, a difference of one cell is well inside what three
Bernoulli draws produce and **is not evidence of anything**.

**The threshold is probably not load-bearing, and if it is, that is the finding.**
At T = 3 with the fix, arms that learn the task reach ~100 % exact match and arms
that do not reach ~0 %. Any threshold between 5 and 50 points gives the same
answer in that regime. It only matters if an arm lands in the middle — and if two
arms land 5 to 20 points apart, pre-registering a number to adjudicate that would
be pretending to a precision three seeds do not support. So:

> **If every arm's best exact match falls in the 5–95 % band, the comparison is
> reported as UNRESOLVED and escalated to the user**, rather than decided by the
> threshold.

## 6. Recorded fields

Per cell: exact match and token accuracy at T ∈ {1, 2, 3} at each of the four
stage boundaries and on the best checkpoint; loss; distinct predicted tokens;
shortcut agreement at T = 2 and T = 3; **post-training per-step RMS and growth
ratio**; parameter count and forward FLOPs; **wall-clock seconds per training
step**, because four sequential narrow block calls and one wide call have the
same FLOPs and different latency (§9).

**A partial report is written after every cell**, as in Batches 03 and 04.

## 7. Pre-registered outcomes

Keyed on **best exact match per arm**, with the §5 threshold and the §5 escalation
clause.

| pattern | reading |
|---|---|
| **A ≈ C, both > B** | **Depth, not compute.** Four block executions beat one wide layer at equal compute, and sharing costs nothing — the project's claim supported, with arm A using 3.84× fewer parameters. |
| **A ≈ B ≈ C** | **Compute, not depth.** Batch 03's K = 4 result was the four-fold FLOPs; recursion is one way to spend them, not a better one. |
| **C > A** (by more than 5 points) | **Depth needs unshared parameters.** Recursion captures part of the benefit; parameter invariance costs accuracy, and the gap is the number to report. |
| **B > A** (by more than 5 points) | **Width beats depth at equal compute.** A negative result for the architecture, reported as one. |
| every arm in the 5–95 % band | **UNRESOLVED** — report and escalate (§5). |
| **arm A below 50 % exact match** | Arm A fails to replicate Batch 03's K = 4 (~100 %). Treat as a **batch failure requiring investigation**, not a finding; report and stop. |
| any arm DEGENERATE, or below the 26.6257 % shortcut ceiling, or attributed SHORTCUT | That arm learned nothing usable; exclude it from the comparison and report why. |
| bounded-growth gate fails, or any cell's post-training growth ≥ 1.05 | That arm is not a fix-on arm; exclude it from every cross-arm conclusion. |

## 8. Compute

Per-step times **measured on this machine** with the fix on, T = 3, medium width,
batch 128, deterministic kernels, 120 timed steps after 20 warmup:

| arm | s/step | per cell (35,000) | 3 seeds |
|---|---|---|---|
| A tesseract K = 4 | 0.04051 | 23.63 min | 70.9 min |
| B width-scaled K = 1 | **0.02940** | 17.15 min | 51.4 min |
| C unrolled L = 4 | 0.04196 | 24.48 min | 73.4 min |
| | | **training total** | **3.26 GPU-h** |

| | GPU-h |
|---|---|
| Training, 9 cells | 3.26 |
| Evaluation overhead (+7.6 %, measured from Batch 02/03 outturn) | 0.25 |
| Bounded-growth gate | ~0.02 |
| **Estimated total** | **≈ 3.53 GPU-h** |

**Under the 8 GPU-h threshold, so the batch proceeds without further approval.**

The draft guessed arm B might be *faster* than A despite matched FLOPs, because
one wide matmul reaches higher GPU utilisation than four small sequential ones.
**Measured: 27 % faster** (0.02940 against 0.04051 s/step). That prediction is
recorded as confirmed, and it is also why §6 records wall-clock per step: equal
FLOPs is not equal time, and a reader comparing arms on latency needs the
measurement rather than the FLOP count.

Disk: arms B and C hold ~3.2 M parameters, so ~12.8 MB per checkpoint against
arm A's 3.2 MB: 6 × 2 × 12.8 + 3 × 2 × 3.2 ≈ **173 MB**, plus ~7 MB of JSON.

## 9. What this batch cannot establish

* **Nothing at T = 4 or T = 8.** T = 3 only, and T = 3 fails G2.
* **Nothing about total training compute.** **Forward** FLOPs per sequence are
  matched. Backward cost and optimiser state differ, and arms B and C hold 3.8×
  the parameters, so they use more memory and more optimiser compute. Matching
  forward compute is a choice; a reader may reasonably want total compute matched
  instead, and this batch does not answer that version of the question.
* **Nothing about inference latency**, beyond the per-step wall clock §6 records.
* **No takeoff probability.** Three seeds per arm, and the project has seen cells
  at identical settings be bimodal.
* **Nothing about scaling.** One width.
* **Nothing about `param_matched`.** It is parameter-matched, not
  compute-matched (837,516 parameters, 27,992,064 FLOPs — a quarter of the
  compute), and it carries a documented head-dimension confound under
  Amendment 02 D3. It belongs in its own batch.

## 10. Execution

```
systemctl --user enable --now tesseract-chain@05.service
```

Persistent enabled unit with `--resume`, so a reboot restarts it and it continues
from the cells on disk. This file is committed before that command runs.
