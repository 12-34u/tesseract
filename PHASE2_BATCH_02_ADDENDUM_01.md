# Addendum 01 — how the two K classifications combine into a §6 row

**Date: 2026-09-30. Committed while the probe is running, before K = 8 has
started.**

A separate dated file, as Batch 01's addenda were.
[PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md) (`ef2e7dd`, amended by
`c68a1f6`) is not edited and stands as committed, including the gap this closes.

## 1. What exists at commit time

Verified immediately before committing, from
`runs/phase2/batch02_t3_curriculum.console.log`:

| cell | state |
|---|---|
| K = 1, seed 0 | reported — best val 64.0395 %, shortcut agreement 18.11 %, 60 distinct tokens |
| K = 1, seed 1 | reported — best val 49.9914 %, shortcut agreement 14.54 %, 60 distinct tokens |
| K = 1, seed 2 | **running**, at step 3,000 of 35,000 |
| K = 8, all seeds | **not started** |
| `t3_curriculum_report.json` | **does not exist** — no classification has been computed for anything |

So **K = 8 is decided in advance by this file in full**, and so is K = 1's third
seed. What is not decided in advance is K = 1's classification, and §4 states
that plainly rather than leaving it to be noticed.

## 2. The gap

[PHASE2_BATCH_02_PREREG.md](PHASE2_BATCH_02_PREREG.md) §6 keys its outcome table
on "**outcome at T = 3**", in the singular. But the probe produces **two**
classifications, one per K, exactly as every Batch 01 arm did. §6 does not say
how the two combine into one row.

That is a defect in a pre-registration, not a question to settle after seeing
which K did better. With K = 1's two reported seeds both above the LEARNABLE bar
it is already foreseeable that the two K values might land on different rows,
which is precisely why the rule has to be written now rather than at the end.

## 3. The rule

> **The row is determined by the more favourable K.** If either K value reaches
> a given row, that row applies. Both K classifications, and both attributions,
> are reported in full alongside it; neither is replaced by the other.

Row order, most to least favourable, from §6's own table:

1. LEARNABLE, attribution COMPOSITION
2. LEARNABLE, attribution UNRESOLVED
3. LEARNABLE, attribution SHORTCUT
4. PARTIAL
5. NOT LEARNABLE or DEGENERATE

Rows 2 and 3 are ordered that way because row 2 stops for a decision and row 3
does not; taking the row that stops is the conservative choice when one K is
unresolved and another is attributed to the shortcut.

**This is not a new rule.** It is the reading
[PHASE2_BATCH_01_ARM5_CRITERIA.md](PHASE2_BATCH_01_ARM5_CRITERIA.md) §1 already
fixed for Batch 01, in its own words: "An arm moves if either of its K values
moves." This addendum states that the same convention governs §6's table, so the
two batches are read the same way.

**It is also the more permissive of the two candidate conventions, and with
K = 1's first two seeds already above the bar it is the more favourable one to
adopt at this moment.** That has to be said outright rather than dressed up.
The justification for it is precedent and nothing else: the stricter convention
— requiring *both* K values to reach a row — is the one that would be new here,
and introducing it now would itself be a rule change made with K = 1's numbers
in hand. The safeguard is §3's second sentence: **both per-K classifications and
both attributions are reported in full**, so any reader who prefers the stricter
reading can apply it to the reported numbers without needing anything this
probe did not record.

## 4. What is retroactive here, and what is not

**For K = 8: fully prospective.** Not one K = 8 step had run at commit time.

**For K = 1: partly retroactive.** Two of its three seeds had reported. The
arithmetic of the pre-registered rule over those two seeds was therefore already
determined before this file was written — 64.0395 % and 49.9914 % both clear the
6.6667 % bar, both cells use all 60 tokens so neither is degenerate, and both
shortcut agreements sit below the 50 % COMPOSITION threshold. K = 1's own
classification is not in this addendum's gift and is not altered by it.

What this addendum decides is only the **combination** of two per-K
classifications into one row. It cannot change either K's classification, it
cannot promote a seed, and it cannot move any bar.

It can, however, select a more favourable row than the stricter convention
would, and for K = 1 it does so partly retroactively. A reader should discount
it accordingly, and §3 gives them what they need to: every per-K classification
and attribution is reported, so the stricter reading remains available on the
same numbers. The disclosure in this section exists so that this judgement can
be checked rather than taken on trust — which is the point of writing it down
while K = 8 has not run, rather than at the end.

## 5. What does not change

The bars, the seed counts, the majority-class floor of 1.8066 %, the collapse
threshold, the shortcut-attribution thresholds of 90 % and 50 %, the budget, the
probe's configuration and the content of every §6 row are untouched. Nothing
here makes a positive result easier to obtain at any K.
