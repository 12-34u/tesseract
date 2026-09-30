# Erratum 01 — the Batch 02 K comparison is confounded

**Date: 2026-09-30.** This corrects §8 of
[PHASE2_BATCH_02_REPORT.md](PHASE2_BATCH_02_REPORT.md) (`98e3432`). That file is
not edited and stands as committed, including the sentences withdrawn here. No
number changes and no classification changes: the correction is entirely about
what the K = 1 versus K = 8 difference is allowed to mean.

Written before Batch 03 runs, and before any Batch 03 result exists.

## 1. The defect

**Batch 02 ran `fix=none`.** It inherited Arm 3's conditions, and Arm 3 ran
without the residual-stream fix. So every K = 8 cell in Batch 02 carried the
unbounded residual stream that Batch 01 measured directly:

| | K = 1 | K = 8 |
|---|---|---|
| z_L RMS growth, first recursion step → last | ~1.00× | **17.70×** |
| Step-0 validation loss (nats) | 4.43 | **33.97** |

Source: [PHASE2_BATCH_01_REPORT.md](PHASE2_BATCH_01_REPORT.md) §3, from
`runs/phase2/batch01/arm2a_step0/` and `arm2a_residual_norms/`.

**The confound lies on the K axis itself.** It is not a general source of noise
that happens to affect both arms; it grows with the number of block executions,
which is the very quantity the comparison varies. Batch 01 said so in as many
words — the confound is "a difference in optimisation conditions sitting on
exactly the axis the experiment attributes its effect to" — and Batch 02
nonetheless drew a K conclusion without controlling it.

This was a failure of the report, not of the probe. Batch 02's *pre-registered
purpose* was T = 3 learnability, and for that purpose `fix=none` was correct: it
made the probe a single-lever change against Arm 3. The error was in §8, which
went on to read a K result off a design that was never built to support one.

## 2. What is withdrawn

**§8's concluding claim:**

> on this benchmark, at this width, under this schedule and at equal step budget,
> **eight recursion steps did not buy depth — it cost it.** […] The first
> condition in which that could have shown up produced the opposite sign.

**Replace with:**

> **K = 8 failed to learn T = 3 under this schedule with the residual confound
> uncontrolled. Whether the recursion depth or the confound caused that failure
> is untested.**

The two hypotheses are not distinguished by anything Batch 02 measured. A K = 8
cell beginning training 29.5 nats further from chance, with a hidden state
growing 17.7× across its recursion, is not a clean instance of "more recursion".
The sign of the effect is therefore unknown, not negative.

Everything else in §8 stands, including the caveats already recorded there — that
a curriculum probe is not a grid cell, that three seeds cannot separate a K
effect from seed variance, that the budget is matched in steps and not in compute,
and that K = 1 had not converged. Those were necessary and remain so; they were
not sufficient, and this is the one that was missing.

## 3. A second caveat, also missing

**K = 8 may simply need longer to take off than K = 1 did.** §4 records that
K = 1's three cells crossed the LEARNABLE bar between 750 and 4,750 steps into
the 20,000-step target stage, and that all three were **still rising** at step
35,000. A flat K = 8 cell at a fixed step budget is therefore not evidence of a
ceiling: the budget was shown by K = 1 itself to be marginal even at a depth that
is learnable.

So "K = 8 is flat" should be read as "K = 8 had not taken off within 20,000
target-stage steps", which is a statement about the budget as much as about K.

## 4. What this does to the rest of the report

| section | effect |
|---|---|
| §1–§3 (classification, shortcut attribution) | **unaffected.** K = 1 LEARNABLE / COMPOSITION and K = 8 NOT LEARNABLE are per-K verdicts under pre-registered rules, not comparisons. |
| §4 (takeoff, slopes) | unaffected, and §3 above strengthens its relevance. |
| §5 (what this does to the T = 4 claim) | **unaffected.** That argument is about budget at a given depth and does not compare K values. |
| §6 (stage boundaries) | unaffected. That K = 8 solves T = 1 and T = 2 is a fact about K = 8 alone, and it remains the reason its T = 3 failure is specific rather than general. |
| §7 (the §6 outcome table) | **unaffected.** Both conventions turn on the per-K classifications, which do not change. The recorded either-K row still applies. |
| §8 (K comparison) | **corrected here.** |
| §9 (what it does not establish) | extended: add the confound to the list. |
| §7.1, point 2 | **weakened.** It says the only depth discriminating between K values "discriminates against more recursion". Replace with: discriminates between K values *with the confound uncontrolled*. The point that T = 3 carries the whole K comparison stands. |

## 5. What settles it

[PHASE2_BATCH_03_PREREG.md](PHASE2_BATCH_03_PREREG.md): Batch 02's configuration
with one change, Addendum 02's non-affine per-step normalisation on for every K,
across K ∈ {1, 2, 4, 8}. That fix is the only one Batch 01 showed actually bounds
the stream — verified at a post-training growth ratio of exactly 1.0000× in all
six Arm 2c cells — and the output-side LayerNorm does not, so Arm 2b is no
substitute.

Pre-registered before it runs, with the outcomes for both directions fixed in
advance.
