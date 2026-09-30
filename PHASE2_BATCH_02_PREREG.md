# Phase 2 — Batch 02 pre-registration: the T = 3 learnability probe

**Date: 2026-09-30. Committed before the probe runs.**

Batch 01 ended on the Stage 3 row "all arms NOT LEARNABLE"
([PHASE2_BATCH_01_REPORT.md](PHASE2_BATCH_01_REPORT.md) §7), which hands the
next choice to the user. The choice taken is option 1 of
[PHASE2_BATCH_01_ARM5_CRITERIA.md](PHASE2_BATCH_01_ARM5_CRITERIA.md) §4:
**T = 3 in the curriculum.** This file records the probe and its decision rules
before any of it is executed, in the same way every arm of Batch 01 was
recorded.

Verified immediately before committing: **no `t3_curriculum_report.json` exists
anywhere** on disk, and `systemctl --user` lists no `tesseract-batch02` unit. So
nothing below could have been written with a result in hand.

One GPU run did happen first, and it is recorded rather than omitted:
`scripts/phase2_batch02.py --smoke`, 40 steps at K = 1 seed 0, to check that
the code path completes before the real run spends three GPU-hours reaching a
crash. Its output is in `runs/phase2/batch02/t3_curriculum_smoke/`, its report
carries `"smoke": true` and `steps: 40`, and it is not a result and not
classified — after 40 steps every depth sits at chance by construction. §10
records what it verified.

Two things were measured *before* writing the rules, and both are measurements
of the **task**, not of any model — no model is trained, loaded or evaluated by
either, so neither could steer an outcome rule toward a result:

* the G1 and G2 gates at T = 3 (§2), which had never been run there;
* the T = 3 validation split's majority-class frequency (§4.1).

Both are reported in full below, including the part that is bad news.

## 1. The question

Batch 01's Arm 3 solved T = 1 in every cell and T = 2 in five of six, on
freshly generated samples, and then sat at chance from the first T = 4 stage
onward. T = 3 is the only untested depth between the depth that is solved and
the depth that is not.

**Is T = 3 learnable under the same curriculum that solves T = 1 and T = 2?**

This is a single-lever change against Arm 3 — the ladder's top moves from 4 to
3 and nothing else moves — so it is attributable in the way Batch 01's arms
were. It is a diagnostic. **No number it produces is evidence for the depth
hypothesis or belongs in the Phase 2 grid**, and T = 3 remains an
evaluation-only depth of the benchmark (`interpolation_t_values` in
`configs/phase2/benchmark.yaml`) unless an amendment says otherwise.

## 2. G2 fails at T = 3, and this is the most important thing in this file

The gates had only ever been run at T ∈ {1, 2, 4, 8}. Run at T = 3 on CPU
(`phase2.gates.evaluate_group("A5", [3], 17, …)`, the committed gate config,
seed 7, 10,000 states):

| | result |
|---|---|
| G1 uniformity | **pass** — TV distance to uniform 0.00796 (limit 0.02) |
| G1 sensitivity | **pass** — every in-window cell changes the output in ≥ 97.9 % of cases; both out-of-window cells in 0.0 % |
| G1 cycles | **pass** — no repeated global state in 1,024 steps |
| G2 shortcuts | **FAIL** — `window_product_forward` agrees with F³ on **26.71 %** of tokens (limit 5 %) |

Twenty-four of the 25 shallow candidates sit at chance (1.6–1.7 %). The one
that does not is s[i]·s[i+1]·s[i+2]·s[i+3].

**The mechanism, verified exactly and not on paper alone.** F(s)[i] =
s_i·s_{i+1}, so

* F²[i] = s_i s_{i+1}² s_{i+2}
* F³[i] = s_i s_{i+1}² s_{i+2} s_{i+1} s_{i+2}² s_{i+3}

and F³[i] equals s_i s_{i+1} s_{i+2} s_{i+3} **iff (s_{i+1} s_{i+2})² = e**.
Sixteen of A5's sixty elements square to the identity, giving 16/60 = 26.667 %.
Checked token-for-token against `phase2.automaton.iterate` over 20,000 states:
the agreement mask and the involution mask are identical everywhere, and
`tests/phase2/test_batch02.py` re-checks it.

This is **the same mechanism as the documented T = 2 failure**, where
s[i]·s[i+2] agrees with F² iff s_{i+1}² = e, also 26.667 %. Amendment 01 handled
that by designating T = 2 a diagnostic depth and recording the mechanism; T = 3
is likewise a diagnostic, evaluation-only depth, so it lands in the same
category and does not touch the amended gate verdict
(`PASS_WITH_AMENDMENT_01`), which was over the primary depths T ∈ {4, 8}. Both
of those pass G2.

Maximum *sequence* agreement at T = 3 is 0.0 %: the shortcut is never right
across a whole ring, only on 26.7 % of positions.

### 2.1 Why this changes the decision rule and not the probe

The pre-registered LEARNABLE bar is chance + 5 points = **6.667 %** token
accuracy. The shortcut scores **26.6 %**. **A model that learned nothing but
the shortcut would be classified LEARNABLE.**

That is the same class of defect Addendum 03 closed for the majority-class
floor, and it has to be closed the same way — before the probe runs, and in the
strict direction only. Accuracy cannot separate the two hypotheses at T = 3, so
§5 adds a measurement that can, and §6 pre-registers how the result is read.
The probe's training configuration is untouched by this.

## 3. The probe

Arm 3's configuration with exactly one field changed. `PROBE_CONFIG` in
`phase2/batch02.py` holds it as data, and
`tests/phase2/test_batch02.py::test_probe_matches_arm3_except_for_the_target_depth`
compares it field by field against Arm 3's committed `ARM_CONFIG["3"]`, so
silent drift fails a test rather than producing an uninterpretable comparison.

| | value | same as Arm 3 |
|---|---|---|
| Curriculum ladder | T = 1 → 2 → 3, 5,000 steps per stage | shape yes, top 4 → **3** |
| Target stage | T = 3, 20,000 steps | shape yes, depth 4 → **3** |
| Total steps per cell | 35,000 | yes |
| Recursion K | {1, 8} | yes |
| Seeds | 0, 1, 2 | yes |
| Width | medium (`phase2/prototype_medium`, 837,436 params) | yes |
| Residual fix | none | yes |
| Optimiser | AdamW, constant 1e-3, weight decay 0.01 | yes |
| Batch / eval interval | 128 / every 250 steps | yes |
| Training data | freshly generated every step, validation inputs excluded | yes |
| Test split | never read | yes |

**The target depth appears twice** — as the ladder's last rung and again as the
20,000-step target stage — because that is how Arm 3's schedule was built
(stages [1, 2, 4] then T = 4 for 20,000, so T = 4 received 25,000 steps in
total). Batch 02 keeps that shape so the budget at the target depth matches
Arm 3's exactly. Boundaries fall at steps 5,000 / 10,000 / 15,000 / 35,000.

**Reading of the instruction, recorded.** "Stages matching Arm 3's lengths, then
the target-length stage at T = 3" is taken as three 5,000-step ladder stages
[1, 2, 3] plus the 20,000-step target stage, totalling 35,000 — identical to
Arm 3. The alternative reading, a two-rung ladder [1, 2] plus the target stage
(30,000 steps), was rejected because it would give Batch 02 a smaller budget
than Arm 3 and confound the depth comparison with a budget difference.

### 3.1 The T = 3 validation split

T = 3 was an evaluation-only depth, so a T = 3 *validation* split did not exist
and is generated here. It comes from the same `SplitStore` machinery as every
other split, keyed by (data_seed 1729, split=val, T=3, n=17), which is an
independent stream from the T = 3 test split (split=test):

| | |
|---|---|
| `A5/main/val/T3/n17/N2048` | sha256 `5e5d4b4951328614…`, recorded in the splits manifest |
| Overlap with the T = 3 test split | none (asserted in `tests/phase2/test_batch02.py`) |
| Training stream | excludes every validation input, as all arms do |

The T = 3 **test** split is not read by this probe.

## 4. Classification

The pre-registered rule
([`configs/phase2/probe_preregistration.yaml`](configs/phase2/probe_preregistration.yaml),
`PHASE2_PROBE_PREREGISTRATION.md`) **and** the amended rule
([PHASE2_BATCH_01_ADDENDUM_03.md](PHASE2_BATCH_01_ADDENDUM_03.md)) are both
applied and both reported, neither replacing the other, exactly as Batch 01
reports them.

Unlike Batch 01, the amended rule is applied **in-run** rather than post-hoc.
Addendum 03 was applied post-hoc only because Arm 3 was already in flight when
it was written; it has been committed since `ade9a6c` and applying it in-run is
what keeps every Batch 02 cell judged by one rule.

### 4.1 T = 3's own floors

Measured on the T = 3 validation split before the probe runs:

| floor | value |
|---|---|
| Uniform chance | 1.6667 % |
| **Majority-class baseline (T = 3)** | **1.8066 %** |
| **Shortcut ceiling (T = 3)** | **26.6257 %** |
| LEARNABLE accuracy bar | 6.6667 % |
| LEARNABLE loss bar | 3.9943 nats (ln 60 − 0.1) |

For comparison, the majority-class baselines already recorded are 1.9244 % at
T = 4 and 1.8354 % at T = 8. The shortcut's exact-match rate on the T = 3
validation split is 0.0000 %.

### 4.2 The rules, unchanged

* **LEARNABLE** — accuracy ≥ 6.6667 % or loss ≤ 3.9943 nats, in ≥ 2 of 3 seeds.
* **PARTIAL** — above chance on ≥ 1 seed but below the LEARNABLE bar.
* **NOT LEARNABLE** — indistinguishable from chance on all seeds.
* **Amended (Addendum 03 §3.1)** — a seed counts as above chance only if it also
  beats **1.8066 %**, T = 3's own majority-class frequency.
* **Amended (Addendum 03 §3.2)** — a cell predicting ≤ 3 distinct tokens is
  **DEGENERATE** regardless of accuracy and counts as neither above chance nor
  learnable; an arm-and-K whose every seed is degenerate reports as DEGENERATE.
* **Arm 5 single-seed clause** ([ARM5_CRITERIA](PHASE2_BATCH_01_ARM5_CRITERIA.md)
  §1) — a PARTIAL resting on exactly one above-chance seed does not count as
  movement.

No bar is lowered and no seed count is relaxed. §5 and §6 add a check that can
only ever refuse a positive claim, never create one.

## 5. The shortcut check

Recorded per cell, on the T = 3 validation split, from the best checkpoint:

**Shortcut agreement** — the percentage of tokens on which the model and
s[i]·s[i+1]·s[i+2]·s[i+3] emit the same token. This is not an accuracy and is
not compared against any accuracy bar.

It separates the two hypotheses by about 73 points where accuracy separates them
by nothing:

| a model that computes… | its accuracy | its shortcut agreement |
|---|---|---|
| F³ exactly | 100 % | **26.67 %** (where the shortcut happens to be right) |
| the shortcut exactly | 26.63 % | **100 %** |
| nothing | ~1.8 % | ~1.7 % |

Pre-registered thresholds, fixed here and not revisable after seeing output:

| agreement | attribution |
|---|---|
| ≥ 90 % | **SHORTCUT** |
| ≤ 50 % | **COMPOSITION** |
| strictly between | **UNRESOLVED** — reported and escalated to the user, not branched on |

The gap between the hypotheses is 73 points wide, so a value inside it is not
something a threshold should be asked to settle; committing in advance to
escalate is the honest handling. Attribution applies **only to seeds the
amended rule already counted**, so it can never promote a seed
(`tests/phase2/test_batch02.py::test_attribution_cannot_promote_a_seed_the_amended_rule_did_not_count`).
A verdict of NOT LEARNABLE or DEGENERATE has no attribution, because there is
nothing to attribute.

The same diagnostic is recorded at T = 2 against its own documented candidate
s[i]·s[i+2]. That is a free test of a claim Batch 01 made but did not measure —
that its T = 2 stages learned composition rather than the 26.7 % shortcut.

## 6. Pre-registered outcomes

The user's decision, recorded before the probe runs. The first and third rows
are the two the user specified; the middle row is the branch §2.1 forces to
exist, since the outcome it describes is one the user's two-way split would
otherwise route into the amendment row.

| outcome at T = 3 | action |
|---|---|
| **LEARNABLE**, attribution COMPOSITION | Propose an amendment making T ∈ {1, 2, 3} the depth range for the K comparison. **Draft only** — written, committed as a proposal, not adopted and not executed. |
| **LEARNABLE**, attribution SHORTCUT | **Does not support that amendment.** Report that the depth-3 shortcut is learnable and that the boundary for genuine composition remains T = 2. Proceed to the write-up, with the shortcut result as a finding in it. |
| **LEARNABLE**, attribution UNRESOLVED | Report the agreement numbers and stop for the user's decision. No amendment is drafted and the write-up is not finalised. |
| **PARTIAL** | Not learnable in the sense this probe asks about. The single-seed clause applies. Report; the boundary claim stays as the NOT LEARNABLE row has it, and the write-up proceeds with T = 3 recorded as PARTIAL. |
| **NOT LEARNABLE** or **DEGENERATE** | **The learnability boundary is T = 2.** Proceed to the path-3 write-up. |

An amendment drafted under row 1 would have to address separately that T = 3
fails G2, since a depth range for the K comparison that includes a depth with a
26.7 % shallow shortcut is not the same object as one that does not. That is
noted here so the draft cannot quietly skip it.

## 7. The Batch 01 deviation, fixed

Batch 01 §6 recorded this deviation:

> Addendum 01 §A1.2 specified logging validation accuracy at T = 1, 2, 4 *at the
> end of each stage*. The implementation measures the final model at each depth
> instead, once, after training. […] The train-loss trace in §5 is the evidence
> for what each stage achieved, and it is console output rather than a recorded
> artifact field.

Batch 02 fixes it. `phase2.training.train_fixed_steps` gains a `step_hooks`
parameter — a mapping from step to a measurement taken at the end of that step,
before the next batch is drawn — and the probe registers one at each of the four
stage boundaries. Results land in `TrainOutcome.hook_records` and are written to
the report as `detail[].stage_boundary_validation`, one entry per boundary,
each holding for **every** depth T ∈ {1, 2, 3}:

`token_accuracy`, `exact_match`, `loss`, `distinct_predicted_tokens`,
`majority_class_percent`, and `shortcut_agreement_percent` at the depths where
G2 found a candidate (T = 2 and T = 3).

The hook is called at the end of the boundary step, so each record describes the
model the finishing stage produced rather than the one after it. When
`step_hooks` is None the loop behaves exactly as before, which
`tests/phase2/test_batch02.py::test_no_hooks_means_no_hook_records` pins, so the
grid and every Batch 01 arm are unaffected.

`best_checkpoint_validation_by_t` records the same fields for the best
checkpoint, which is the Batch 01 measurement — kept, so the two are
comparable, rather than replaced.

## 8. Compute

Per-step times as pre-registered in
[PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md) §9 (RTX 3050,
deterministic kernels): K = 1 → 0.01206 s/step, K = 8 → 0.07954 s/step. Model
cost does not depend on T — depth enters only through target generation on
CPU — so Arm 3's measured cost transfers directly.

| | |
|---|---|
| Steps per cell | 35,000 |
| K = 1 per cell | 7.0 min |
| K = 8 per cell | 46.4 min |
| 3 seeds × 2 K values | **2.67 GPU-h** |
| Stage-boundary evaluations (12 per cell, +8.6 % of eval work) | ≈ 0.05 GPU-h |
| **Estimated total** | **≈ 2.72 GPU-h, about 2 h 45 min wall clock** |

The prediction is checked against a measurement: Arm 3 ran the identical shape
in **2 h 40 min 24 s** (chain log, 16:52:54 → 19:33:18), against a predicted
2.67 GPU-h = 2 h 40 min 12 s.

**Budget.** The 12-hour limit in Batch 01's §9 is that batch's budget, and
Batch 01 closed at 9.31 GPU-h with Arm 5 skipped. Batch 02 is a new batch of one
probe at 2.72 GPU-h, well inside a comparable limit. Recorded so the figure is
not carried over by accident: **Batch 02's cost is not additive to Batch 01's
closed budget.**

**Disk.** 6 cells × 2 checkpoints × 3.20 MB ≈ 38 MB, plus curves and JSON
≈ 4 MB, so **≈ 42 MB**.

## 9. What this probe cannot establish

* **Nothing about the depth hypothesis.** A diagnostic against a single
  reference, not a grid cell.
* **Nothing about T = 4.** A T = 3 result does not make T = 4 learnable or
  unlearnable; Batch 01 already measured T = 4.
* **Nothing about the shape of the curve between measured depths.** Even with
  T = 3 measured, three points do not establish a shape — the correction in
  [PHASE2_BATCH_01_REPORT_ERRATUM_01.md](PHASE2_BATCH_01_REPORT_ERRATUM_01.md)
  §2 applies to this report in advance as well.
* **Nothing about T = 3 as a benchmark depth.** It fails G2, and the amended
  gate verdict covers the primary depths T ∈ {4, 8} only.
* **Nothing from a LEARNABLE verdict alone.** Per §2.1 and §6, at T = 3 that
  verdict is not interpretable without the attribution.

## 10. Execution

### 10.1 The smoke run, and what it verified

Run before this file was committed, at 40 steps, K = 1, seed 0, into a separate
directory. It confirmed the mechanics and nothing about T = 3:

* the four stage-boundary hooks fire at the right steps and are attributed to
  the right stages — recorded as steps 10 / 20 / 30 / 40 ending stages
  T = 1 / 2 / 3 / 3, matching the schedule's shape;
* each boundary record carries all three depths with every field §7 lists;
* the T = 3 validation split loads at the sha256 in §3.1 and the T = 1 and T = 2
  splits at their existing ones;
* the shortcut diagnostic evaluates at T = 2 and T = 3 and is absent at T = 1,
  as it must be;
* the classification, the Addendum 03 amendment and the attribution are all
  produced, and the attribution is `null` because no seed was counted — which is
  the behaviour §5 requires of a chance-level result.

It cannot verify any claim about learnability at 40 steps, and none is made
from it.

### 10.2 The probe

```
systemd-run --user --unit=tesseract-batch02 --same-dir \
    scripts/phase2_batch02_chain.sh
```

A service, not a scope, for the reason recorded in the Batch 01 driver: a scope
dies with the shell that started it. The driver skips a probe whose report
already exists and never overwrites a partially written directory, sending a
retry to a sibling `__attemptN`.

This file is committed before that command is run.
