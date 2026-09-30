# Erratum 01 — two claims withdrawn from the Batch 01 report

**Date: 2026-09-30.** This corrects two statements in
[PHASE2_BATCH_01_REPORT.md](PHASE2_BATCH_01_REPORT.md) (`6dfb3c0`). That file is
not edited and stands as committed, including the sentences withdrawn here. The
underlying numbers are unchanged and no classification moves: both corrections
are about what the numbers were said to *mean*.

Written before any Batch 02 result exists, and before Batch 02 is run.

## 1. Withdrawn: "Arm 2c made collapse worse" and "it cost expressiveness"

**What the report said** (§5, Arm 2c):

> It does not, and it made collapse *worse*: five of six cells emitted 1–2
> distinct tokens. […] what it bought was nothing, and it cost expressiveness.

**Why that is withdrawn.** At chance the logits are near-uniform, so the argmax
token is arbitrary — it is decided by differences far smaller than the
differences that would encode a decision. Which token wins, and therefore how
many distinct tokens appear across 34,816 positions, is not a behavioural
property of the model at that point. Comparing Arm 2c's distinct-token count
against Arm 3's or Arm 4's is comparing two arbitrary quantities, and the
ordering it produces carries no information about expressiveness.

**What the collapse check is still good for.** It remains valid in the one
direction Addendum 03 introduced it for: as a guard against a *false
above-chance claim*. A cell that clears the accuracy bar while emitting one or
two distinct tokens has cleared it by emitting the modal class, and that is a
sound reason to refuse the claim. The check licenses "this above-chance result
is not real". It does not license "this arm is more collapsed than that one", and
that is the inference the withdrawn sentences made.

So Addendum 03 is unaffected — §3.2 defines DEGENERATE as a veto on a positive
result, which is exactly the surviving direction. What falls away is the Arm 2c
commentary built on top of it, including the suggestion that the Addendum 02
draft's predicted trade-off had been observed. Arm 2c's finding is what §3 of the
report says it is: the bound held exactly (growth 1.0000× before and after
training) and learnability did not change.

The report's Arm 2b commentary is affected in the same way where it calls six
collapsed cells "the most degenerate outcome in the batch". Arm 2b's substantive
result — that both of its PARTIALs rested on cells scoring the majority-class
baseline exactly — does not depend on the comparison and stands.

## 2. Replaced: "a cliff between T = 2 and T = 4"

**What the report said** (§5, Arm 3):

> This is not a gradual degradation; it is a cliff between T = 2 and T = 4.

**Replace with:** *T = 2 is solved and T = 4 is at chance; T = 3 was not
tested.* "Cliff" asserts a shape for a curve measured at two points, and a shape
cannot be read off two points. Whether the fall is abrupt at some depth or
gradual across 2 → 3 → 4 is open, and it is the question Batch 02 exists to
answer. §9 of the report is already careful here — it calls T = 3 "the untested
point inside it" — so this correction brings §5 into line with §9 rather than
changing the conclusion.

## 3. Added: why the T = 4 stage erases the earlier solutions

The report records (§6) that a model which provably solved T = 1 and T = 2
mid-run ends at chance on both, and says only that "whatever the T = 4 stage
does, it destroys the earlier solutions rather than building on them". The
interpretation it should have offered, consistent with the P1c signal analysis
already cited in §9:

At T = 4 the loss rewards partial knowledge almost nothing. The signal analysis
gives the ceiling for predicting from an incomplete window: all-but-one-cell
subsets reach 26.7 %, all-but-two 13.4 %, all-but-three 7.0 %. Composing three
of the four required steps correctly therefore buys very little, and each step
of partial progress buys less than the last. Under that objective a
near-uniform output is a local optimum: it is what the loss prefers over any
partially-correct function the model can reach by a small change. The T = 4
stage then does not merely fail to build on the T = 1 and T = 2 solutions — it
actively moves away from them, because they are worse under the T = 4 loss than
uniformity is.

This is an interpretation, not a measurement. It predicts that the erasure
should be *weaker* at a depth where partial credit is worth more, which is one
of the things Batch 02's stage-boundary measurements will show at T = 3.

## 4. What does not change

| | |
|---|---|
| Every number in the report | unchanged |
| Every classification, original and amended | unchanged |
| The Stage 3 row that applies ("all arms NOT LEARNABLE") | unchanged |
| The Arm 5 decision (not approved, not run) | unchanged |
| The four explanations §9 excludes | unchanged |

Nothing here makes any result more positive, and nothing here was written with
a Batch 02 result in hand.
