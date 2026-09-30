# Phase 2 — Batch 03 pre-registration: T = 3 across K, with the residual fix on

**Date: 2026-09-30. Committed before the batch runs.**

Batch 02 found T = 3 learnable at K = 1 (3 of 3 seeds, attributed to composition)
and not learnable at K = 8. It ran `fix=none`, so the residual-stream confound
was uncontrolled on exactly the axis that comparison varies, and
[PHASE2_BATCH_02_REPORT_ERRATUM_01.md](PHASE2_BATCH_02_REPORT_ERRATUM_01.md)
withdrew the K conclusion for that reason. Batch 03 controls it.

**One change against Batch 02: Addendum 02's non-affine per-step normalisation is
on, for every K.** The K range widens to {1, 2, 4, 8} so that a trend can be
read rather than a two-point difference.

Verified immediately before committing: `runs/phase2/batch03/` does not exist, no
`t3_k_sweep_report.json` exists anywhere, and `systemctl --user` lists no
`tesseract-batch03` unit. No Batch 03 cell has run.

## 1. The question

**With the residual stream bounded, is T = 3 learnable at K > 1, and how does
learnability vary across K ∈ {1, 2, 4, 8}?**

This is the experiment the Batch 02 erratum says would settle whether recursion
or the confound caused the K = 8 failure. It is still a diagnostic: T = 3 fails
G2 (§5), a curriculum probe is not a grid cell, and **no number this batch
produces is evidence for the depth hypothesis or belongs in the Phase 2 grid.**

## 2. The configuration

Batch 02's configuration exactly, with one field changed and the K range widened.
`BATCH03_CONFIG` in `phase2/batch03.py` holds it as data, and
`tests/phase2/test_batch03.py` compares it field by field against Batch 02's
committed `PROBE_CONFIG`, so anything else drifting fails a test rather than
producing an uninterpretable comparison.

| | value | same as Batch 02 |
|---|---|---|
| **Residual fix** | **non-affine per-step normalisation, every K** | **no — the one change** |
| **Recursion K** | **{1, 2, 4, 8}** | widened from {1, 8} |
| Curriculum ladder | T = 1 → 2 → 3, 5,000 steps per stage | yes |
| Target stage | T = 3, 20,000 steps | yes |
| Total steps per cell | 35,000 | yes |
| Seeds | 0, 1, 2 | yes |
| Width | medium (`phase2/prototype_medium`) | yes |
| Optimiser | AdamW, constant 1e-3, weight decay 0.01 | yes |
| Batch / eval interval | 128 / every 250 steps | yes |
| Training data | fresh every step, T = 3 validation inputs excluded | yes |
| Test split | never read | yes |

The normalisation is **non-affine**, as
[PHASE2_BATCH_01_ADDENDUM_02.md](PHASE2_BATCH_01_ADDENDUM_02.md) adopted: an
affine gain is a free parameter that nothing in the objective stops growing with
depth, so it would bound the state at initialisation only. Non-affine costs zero
parameters, which keeps the parameter count K-invariant — the property the whole
project rests on.

**Why not the output-side LayerNorm.** It does not bound the stream. Batch 01 §3
measured K = 8 growth at 17.70× both with and without it. Only per-step
normalisation bounds it, verified post-training at exactly 1.0000× in all six
Arm 2c cells.

## 3. The bounded-growth gate, first

Before any training cell runs, and reusing Arm 2c-i unchanged: per-step hidden
state RMS is measured at K ∈ {1, 2, 4, 8} with the fix applied, and every K must
show a growth ratio below **1.05**. **If any K fails, no training cell runs.**

This is not a formality. If the fix did not bound the stream at some K, a null
result at that K would be uninterpretable in exactly the way Batch 02's was, and
the batch would have reproduced the defect it exists to remove.

## 4. Recorded fields

Per cell, as Batch 02 recorded them:

* **stage-boundary diagnostics** — validation accuracy, exact match, loss,
  distinct predicted tokens, majority-class frequency and shortcut agreement at
  T ∈ {1, 2, 3}, at each of the four stage boundaries, via `step_hooks`;
* **shortcut agreement** on the best checkpoint at T = 3 and T = 2;
* **distinct predicted tokens**, for the Addendum 03 collapse check;
* **takeoff steps and final slope** — first evaluation crossing 6.6667 % and
  26.6257 %, and the least-squares slope over the last 2,500 steps, from
  `scripts/phase2_batch02_curves.py`, which reads recorded curves only.

And one field Batch 02 did not need:

* **post-training per-step hidden-state RMS**, per cell, with its growth ratio.
  Addendum 01 §A1.3 requires the bound to hold *after* training and not only at
  initialisation, and a cell whose bound broke during training is not a
  fix-on cell. **Recorded per cell, not once for the batch.**

## 5. T = 3 still fails G2, and the shortcut check still applies

Unchanged from Batch 02 and not re-litigated here:
`window_product_forward` = s[i]·s[i+1]·s[i+2]·s[i+3] agrees with F³ on exactly
26.67 % of tokens, wherever (s[i+1]·s[i+2])² = e, because 16 of A5's 60 elements
square to the identity (`runs/phase2/gates_t3/gates_t3_report.json`).

The LEARNABLE bar is 6.6667 % and the shortcut scores 26.6257 %, so accuracy
alone cannot separate depth-3 composition from the shallow product at any K. The
pre-registered attribution thresholds are therefore carried over **unchanged**:

| shortcut agreement | attribution |
|---|---|
| ≥ 90 % | SHORTCUT |
| ≤ 50 % | COMPOSITION |
| strictly between | UNRESOLVED — reported and escalated, not branched on |

## 6. Classification

The pre-registered rule and the Addendum 03 amended rule, both applied in-run and
both reported, with T = 3's own majority-class floor of **1.8066 %**. Unchanged
bars: LEARNABLE at ≥ 6.6667 % accuracy or ≤ 3.9943 nats loss in ≥ 2 of 3 seeds;
DEGENERATE at ≤ 3 distinct predicted tokens regardless of accuracy.

**Per-K combination.** [PHASE2_BATCH_02_ADDENDUM_01.md](PHASE2_BATCH_02_ADDENDUM_01.md)
fixed the either-K convention for a two-value K range. This batch has four, and
§8 below keys its outcomes on **specific K values** rather than on a combined
row, so no combination rule is needed and none is used. Every K's classification
is reported separately. Where §8 needs a summary it names the Ks explicitly.

## 7. How a trend across K is read

Fixed now, before any cell runs, because "is there a trend" is exactly the
question that invites a post-hoc reading.

Let L = the set of K values classified LEARNABLE (amended rule), and for each K
let a(K) be the mean best validation token accuracy over its three seeds.

| pattern | reading |
|---|---|
| **L = {1, 2, 4, 8}** | Learnability is K-independent at T = 3 with the fix on. Report a(K) and whether it varies; a monotone decline in a(K) with every K still learnable is a **quantitative** effect, not a learnability boundary. |
| **L is a proper prefix** ({1}, {1,2} or {1,2,4}) | **Monotone decline.** There is a K threshold above which T = 3 is not learnable at this budget. Report the largest learnable K and the smallest non-learnable one; the threshold lies between them and is not further localised by this batch. |
| **L = {} (not even K = 1)** | The fix has broken something Batch 02 could do, since Batch 02's K = 1 was LEARNABLE with the fix off. Treat as a **failure of the batch**, not a result about K, and report the per-step RMS and stage boundaries as the first place to look. |
| **L non-empty and not a prefix** (e.g. {1, 4} but not 2) | **Not monotone.** No threshold claim is available. Report as non-monotone and attribute it to seed variance unless the gap exceeds the K = 1 seed spread, which Batch 02 measured at 49.99–81.21 %, a spread of **31.2 points**. |

Two constraints on any trend claim, recorded in advance:

* **Three seeds per K cannot establish monotonicity of a(K).** With a 31-point
  seed spread at K = 1, a difference in a(K) smaller than that is not a trend.
  A trend claim is admissible only about the *learnability* pattern (the table
  above), never about a(K) alone.
* **Equal steps is not equal compute.** At 35,000 steps K = 8 performs eight
  times the block executions of K = 1. Any decline across K is therefore
  measured at increasing compute, which makes a decline *harder* to explain away
  and an improvement *easier*. This must be stated wherever a trend is reported.

## 8. Pre-registered outcomes

The two the user specified, plus the cases §7 forces to exist.

| outcome | action |
|---|---|
| **K = 8 LEARNABLE** (attribution COMPOSITION) | The Batch 02 K = 8 failure is **attributed to the residual confound**. Propose, **draft only**, a K comparison protocol at T = 3 with the fix on — matched compute rather than matched steps, and a budget chosen from Batch 02's takeoff data. |
| **K = 8 NOT LEARNABLE or DEGENERATE**, with the bounded-growth gate passed and the post-training bound holding | **The negative K result stands with the confound excluded.** Report it as such, with the §7 trend reading and the §9 caveats. |
| **K = 8 LEARNABLE, attribution SHORTCUT or UNRESOLVED** | Not a composition result. Report the agreement figures; no attribution to the confound is made and no protocol is drafted. |
| **The bounded-growth gate fails at any K**, or a cell's post-training growth exceeds 1.05 | That K is **not a fix-on cell**. Its result is reported and excluded from every confound-related conclusion. If K = 8 is affected, neither of the first two rows applies and the batch answers nothing about the confound. |
| **K = 1 not LEARNABLE** | §7's third row: a failure of the batch. No K conclusion; investigate. |

Nothing in this table can be reached without the gate in §3 passing at the K it
concerns.

## 9. What this batch cannot establish

* **Nothing about the depth hypothesis.** A diagnostic at a depth that fails G2,
  not a grid cell.
* **Nothing about matched-compute K scaling.** Steps are matched, compute is not.
* **Nothing about a converged accuracy.** Batch 02's K = 1 cells were still rising
  at 35,000 steps and this batch uses the same budget, so a(K) is a
  budget-limited quantity at every K. A low a(K) at high K may be slow takeoff
  rather than a ceiling — the caveat the Batch 02 erratum §3 added.
* **Nothing about T = 4.** Untouched.
* **Nothing about the fix's effect at T = 4 or T = 8.** T = 3 only.

## 10. Compute

Per-step times **measured on this machine** with the fix applied, at 128 batch,
medium width, T = 3, deterministic kernels, 120 timed steps after 20 warmup:

| K | s/step (fix on) | per cell (35,000 steps) | 3 seeds |
|---|---|---|---|
| 1 | 0.01134 | 6.61 min | 19.8 min |
| 2 | 0.02120 | 12.36 min | 37.1 min |
| 4 | 0.04011 | 23.40 min | 70.2 min |
| 8 | 0.07952 | 46.38 min | 139.1 min |
| | | **training total** | **4.44 GPU-h** |

Evaluation overhead is taken from Batch 02's measured outturn rather than
estimated: its 6 cells took 2 h 40 min 40 s against 2.49 GPU-h of measured
training time, i.e. **+7.6 %**.

| | GPU-h |
|---|---|
| Training, 12 cells | 4.44 |
| Evaluation overhead (+7.6 %) | 0.34 |
| Bounded-growth gate (§3) | ~0.02 |
| **Estimated total** | **≈ 4.80 GPU-h** |

**Under the 8 GPU-h threshold the user set, so the batch proceeds without
further approval.** Disk: 12 cells × 2 checkpoints × 3.20 MB ≈ 77 MB, plus
curves and JSON ≈ 8 MB, so **≈ 85 MB**.

## 11. Execution

```
systemd-run --user --unit=tesseract-batch03 --same-dir \
    scripts/phase2_batch03_chain.sh
```

A service, not a scope. The driver skips a probe whose report already exists and
never overwrites a partially written directory, sending a retry to a sibling
`__attemptN`.

This file is committed before that command is run.
