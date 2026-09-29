# Batch 01 — Arm 5 approval criteria and the next step after it

**Date: 2026-09-29. Committed before any arm beyond Arm 1 reported.**

At the time of committing, the only Batch 01 results in existence were Arm 1
(`arm1_fixed_set/arm1_report.json`, gate PASS) and the Arm 2a measurements.
Arm 2b was running at step 1,500 of its first cell and had produced no report;
Arms 3, 4 and 2c had not started. So none of the rules below could have been
written with knowledge of the outcomes they decide.

This records the user's decision rule. It is applied **mechanically** alongside
the Stage 3 table in [PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md) §10,
as superseded by Addendum 01 §A1.3.

## 1. What counts as an arm "moving"

An arm moves if it reaches **PARTIAL or LEARNABLE** under the §3 success rule,
**and that classification does not rest on a single above-chance seed.**

The second clause is the operative one. [PHASE2_PROBE_REPORT.md](PHASE2_PROBE_REPORT.md)
established that with a per-cell false-positive rate of 0.05, one above-chance
crossing is the *modal* outcome under the null when several cells are tested:
across twelve probe cells the probability of at least one crossing was 46 %,
with an expected count of 0.6. A PARTIAL resting on one seed is therefore not
evidence of movement, and does not count here.

Concretely, for an arm at a given K:

* **LEARNABLE** — counts as moving (it already requires 2 of 3 seeds).
* **PARTIAL on ≥ 2 of 3 seeds** — counts as moving.
* **PARTIAL on exactly 1 of 3 seeds** — does **not** count as moving.
* **NOT LEARNABLE** — does not count as moving.

An arm moves if either of its K values moves.

## 2. Arm 5 approval

> **Arm 5 is approved only if ≥ 2 of arms {2b, 3, 4, 2c} individually move**,
> in the sense of §1.

If fewer than two arms move, Arm 5 is **not run**, and the skip is recorded in
the Stage 2 report with the per-arm classifications that produced it.

Two standing constraints apply even when the criterion is met:

* Arm 5 is **not** started without explicit approval. Addendum 01 §A1.5 records
  that Arms 1–4 plus Arm 2c total 9.31 GPU-h, and that adding Arm 5's worst case
  of 4.20 GPU-h would reach **13.51 GPU-h, exceeding the 12-hour limit**. So Arm
  5 needs approval on compute grounds as well as on this trigger.
* Arm 5 runs the **union of the levers that moved**, at T = 4, K ∈ {1, 8},
  3 seeds, per the pre-registration §8.

## 3. If only Arm 3 moves

No Arm 5. The next step is a **pre-registered probe of the curriculum combined
with Arm 2c's fix**:

| | |
|---|---|
| Levers | curriculum over T **and** non-affine per-step normalisation |
| Depth | T = 4 |
| Recursion | K ∈ {1, 8} |
| Seeds | 3 |
| Success rule | the §3 bars, unchanged, with the §1 single-seed clause |

It is pre-registered before it runs, as every arm in this batch has been. Note
this is a *two-lever* probe and so is not attributable in the way Batch 01's
arms are; it is a follow-up to a specific positive result, not a diagnostic.

## 4. If no arm moves

No Arm 5. The choice among these three belongs to the user, not to this
document or to me:

1. **T = 3 in the curriculum** — a non-power-of-two intermediate depth, to test
   whether the ladder's gaps are what the curriculum fails to bridge.
2. **A group ladder** — a smaller or solvable group before A₅, to locate the
   scale at which the task becomes learnable at all.
3. **The negative-result write-up** — the Stage 3 "all arms NOT LEARNABLE" row,
   whose outline is drafted but not finalised: unlearnable at reachable scale,
   with curriculum, confound fix and optimiser check all excluded.

Batch 01 stops after Stage 3 and this file are applied, and reports. It does not
choose among these.

## 5. Record

Applying this file produces, for each of arms 2b, 3, 4 and 2c: the per-K
classification, the number of above-chance seeds behind it, whether the arm
moved under §1, and the resulting Arm 5 decision with the branch of §2, §3 or §4
that it lands in.
