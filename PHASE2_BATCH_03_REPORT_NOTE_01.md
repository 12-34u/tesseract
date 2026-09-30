# Note 01 — exact match is the more informative metric at T = 3

**Date: 2026-10-01.** A dated note on
[PHASE2_BATCH_03_REPORT.md](PHASE2_BATCH_03_REPORT.md) (`c6e1e50`). That file is
not edited and stands as committed. **No classification changes**, no bar moves,
and no number is revised: this records which of two already-reported metrics
should carry the interpretation.

## 1. The finding

Batch 03 §3 reports both metrics. This note states the conclusion that follows
and that the report left implicit:

> **At T = 3, exact match is the more informative measure, and token accuracy is
> misleading at the low end.**

| K | seed | token accuracy | exact match |
|---|---|---|---|
| 1 | 0 | 64.5393 % | **0.1465 %** |
| 1 | 1 | 43.5375 % | **0.0000 %** |
| 1 | 2 | 57.3788 % | **0.0000 %** |
| 4 | 0 | 100.0000 % | 100.0000 % |
| 4 | 2 | 100.0000 % | 100.0000 % |

**K = 1's LEARNABLE verdict rests entirely on token accuracy, at approximately
0 % exact match.** Two of its three cells produce no correct ring at all in 2,048
validation examples, and the third produces three.

Why token accuracy misleads here: the target is a 17-position ring and the metric
averages over positions, so a model making many independent local errors scores
well on it while never getting a whole example right. Exact match is
conjunctive — it requires all 17 — and so it is the measure that tracks whether
the function was learned rather than approximated. A model that had learned F³
would score highly on both, and K = 4 does.

## 2. What this does and does not change

**Does not change.** The classification of every arm and K in every batch. The
pre-registered rule is defined on token accuracy and loss
(`configs/phase2/probe_preregistration.yaml`), it was committed before any probe
ran, and it is not amended here or retroactively. K = 1 at T = 3 remains
**LEARNABLE** in the record.

**Does change how that verdict is read.** Wherever "K = 1 is learnable at T = 3"
appears, it must be accompanied by its exact match. Reporting the token accuracy
alone overstates it in the direction that matters, and the batch's central
finding — that K ≥ 4 solves T = 3 and K = 1 does not — is visible only on the
strict metric.

This is the same class of defect as the two the project has already closed: the
above-chance bar sitting below the majority-class baseline (Addendum 03) and the
LEARNABLE bar sitting below the shortcut ceiling (Batch 02 prereg §2.1). In each
case a bar was clearable by something that had not learned the task. It is
recorded as a note rather than an amendment because, unlike those two, it does
not change any verdict — it changes which number is quoted beside one.

## 3. Consequence for later batches

[PHASE2_BATCH_04_PREREG.md](PHASE2_BATCH_04_PREREG.md) pre-registers exact match
as a **reported co-primary metric** alongside the existing token-accuracy rule,
for exactly this reason. The pre-registered classification stays on token
accuracy so that Batch 04 remains comparable with every batch before it; exact
match is reported for every cell and every stage boundary, and any claim that a
depth is learnable must state both.

The recorded stage-boundary fields already included `exact_match`, so this costs
nothing new to measure. The nine cells Batch 03 recovered from checkpoints lack
it *inside their stage-boundary rows* only (Addendum 01 §4); their
best-checkpoint exact match is present, which is where §1's table comes from.
