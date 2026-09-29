# Addendum 03 — majority-class floor and collapse check

**Date: 2026-09-29. Committed before Arms 3, 4 and 2c reported.**

A separate dated file, as Addendum 02 was.
[PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md) (`7dbd7b6`), Addendum 01
(`b64305f`) and Addendum 02 (`5bdbb77`) are not edited and stand as committed,
including the parts this supersedes.

## 1. Reports in existence at commit time

Verified immediately before committing:

| report | exists |
|---|---|
| `arm1_fixed_set/arm1_report.json` | yes |
| `arm2a_step0/arm2a_report.json` | yes |
| `arm2a_residual_norms/residual_norms_report.json` | yes |
| `arm2b_residual_fix__attempt2/arm2b_residual_fix_report.json` | yes |
| Arm 3 | **no** — running, 3 of 6 cells, no report written |
| Arm 4 | **no** — not started |
| Arm 2c | **no** — not started |

So Arms 3, 4 and 2c are decided in advance by this file, exactly as the
pre-registration intends. Arm 2b and Probes A/B are amended retroactively, and
both classifications are reported side by side for every one of them.

## 2. Why

Arm 2b classified PARTIAL at both K. Both crossings came from seed 2, whose
numbers were identical to four decimals across K=1 and K=8 — which two
independently trained models should not produce. Investigating found two
defects in the original rule.

**The above-chance bar sat below the majority-class baseline.** It is built from
*uniform* chance, 100/60 = 1.6667 %, and landed at 1.8889–1.9028 % across the
Arm 2b cells. Emitting the most frequent target token of the T=4 validation
split scores **1.9244 %**. A model with no input dependence at all therefore
cleared the bar.

**Nothing detected collapse.** Both Arm 2b seed-2 cells predicted **token 11 at
all 34,816 positions** — a constant, and that constant is the modal class. Their
accuracy was real and their learning was nil. The other Arm 2b cells were nearly
as bad: 1 to 3 distinct tokens out of a 60-token vocabulary.

## 3. What changes

**3.1 Majority-class floor.** A seed counts as above chance only if it beats
**both** the existing bar **and** the most-frequent-target frequency of that T's
validation split.

| split | majority-class baseline | uniform chance |
|---|---|---|
| T = 4 | **1.9244 %** | 1.6667 % |
| T = 8 | **1.8354 %** | 1.6667 % |

**3.2 Collapse check.** Distinct predicted tokens are recorded per cell. A cell
predicting **≤ 3 distinct tokens** is classified **DEGENERATE** regardless of
accuracy, and counts neither as above chance nor as learnable. Three rather than
one allows slightly more than a pure constant without admitting anything that
could be called a solution over 60 tokens.

An arm-and-K whose every seed is degenerate is reported as **DEGENERATE** rather
than NOT LEARNABLE — the distinction matters, because a degenerate result says
the run collapsed, not that the task resisted a genuine attempt.

**3.3 Applies uniformly.** To every arm of Batch 01 and retroactively to Probes
A and B. Every report shows the original classification and the amended one side
by side; neither replaces the other in the record.

Nothing else changes: the LEARNABLE bars, the seed counts, the budgets, the
Stage 3 table and the Arm 5 criteria are untouched.

## 4. How it is applied

`phase2/reclassify.py` re-derives the classification from recorded artifacts,
and `scripts/phase2_reclassify.py` applies it to every arm and probe in one
pass. Distinct predicted tokens are recomputed from each cell's best checkpoint,
so the collapse check reaches arms whose reports predate this addendum.

**The running pipeline is not changed.** Arms classify themselves as
pre-registered while they run; the amendment is applied afterwards. That is
deliberate: Arm 3 was already in flight when this was written, and editing the
classifier would have left Arm 3 judged by one rule and Arms 4 and 2c by
another. Applying it post-hoc in a single pass is what makes it uniform.

## 5. Result of applying it to what exists

| source | K | original | amended | |
|---|---|---|---|---|
| Probe A (T=4) | 1 | NOT LEARNABLE | NOT LEARNABLE | — |
| Probe A (T=4) | 8 | NOT LEARNABLE | NOT LEARNABLE | — |
| Probe B (T=8) | 1 | NOT LEARNABLE | NOT LEARNABLE | — |
| Probe B (T=8) | 8 | PARTIAL | **PARTIAL** | survives |
| Arm 2b (T=4) | 1 | PARTIAL | **DEGENERATE** | changed |
| Arm 2b (T=4) | 8 | PARTIAL | **DEGENERATE** | changed |

**Arm 2b does not move** under the Arm 5 criteria, and now for a stronger reason
than the single-seed clause: every one of its six cells collapsed.

**Probe B's PARTIAL survives.** Its seed 1 uses all 60 tokens and beats the T=8
majority baseline, so it is not the same artifact. It remains what
[PHASE2_PROBE_REPORT.md](PHASE2_PROBE_REPORT.md) called it — consistent with
noise across twelve cells — but it is not degenerate, and the amendment does not
reclassify it.

**Probe A is untouched.** Its cells use 45 to 60 distinct tokens; nothing there
was collapsed or majority-class.

## 6. What this does not do

It does not lower any bar or make a positive result easier to obtain — both
changes are strictly stricter. It cannot turn a NOT LEARNABLE into a PARTIAL or
a LEARNABLE. It does not alter what any arm trained, only how a recorded result
is read.
