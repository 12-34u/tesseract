# Note 01 — the K = 4 collapse is a symptom of a chance plateau, not its cause

**Date: 2026-10-04.** A dated note on
[PHASE2_BATCH_06_REPORT.md](PHASE2_BATCH_06_REPORT.md) (`8c90984`). That file is
not edited and stands as committed. **No number and no classification changes.**
Everything here is read from the existing console log and report artifacts; no
cell was rerun and the GPU was not used.

## 1. What the report left unexplained

§7 recorded "nothing about why K = 4 collapsed. Three cells at the majority floor
after solving T = 3 is a clean observation with no mechanism attached." This note
narrows that, and the main result is that **the collapse was the wrong thing to be
looking at.**

## 2. The transition shock is universal, at every K

At the T = 3 → T = 4 switch (step 25,000 → 25,250), **every one of the nine cells
takes the same hit**, in a single 250-step interval:

| cell | train loss 25,000 → 25,250 | grad max 25,000 → 25,250 |
|---|---|---|
| K = 4 s0 | 0.0004 → **4.3171** | 0.123 → **27.186** (221×) |
| K = 8 s0 | 0.0293 → **4.3206** | 1.041 → **34.042** (33×) |
| K = 8 s1 | 0.0610 → **4.2948** | 0.619 → **76.526** (124×) |
| K = 8 s2 | 1.0339 → **4.2001** | 4.333 → **32.173** (7×) |

Train loss jumps from near zero to **above** chance (ln 60 = 4.0943), and the
gradient spikes by one to two orders of magnitude. By the next evaluation both
have settled — loss to ~4.10, grad max to ~0.5–0.8 — and the model is at chance.

**So "train loss or grad norm first" cannot be answered from this log:** they move
inside the same 250-step evaluation interval, which is the finest resolution
recorded. Separating them would need per-step logging, which no batch has.

**And the shock does not discriminate.** K = 8's spikes are *larger* than K = 4's
(34–77× against 27×) and K = 8 went on to solve T = 4. Whatever distinguishes the
two, it is not the magnitude of the transition.

## 3. The output collapse is also near-universal

Distinct predicted tokens on the T = 4 split, at each stage boundary:

| cell | 5k | 10k | 25k | **30k (T=4 rung end)** | 90k (end) |
|---|---|---|---|---|---|
| K = 1 s0 | 60 | 60 | 60 | **9** | 1 |
| K = 1 s1 | 60 | 60 | 60 | **9** | 1 |
| K = 1 s2 | 60 | 60 | 60 | **6** | **60** |
| K = 4 s0 | 60 | 60 | 60 | **11** | 1 |
| K = 4 s1 | 60 | 60 | 60 | **4** | 1 |
| K = 4 s2 | 60 | 60 | 60 | **10** | 1 |
| K = 8 s0 | 60 | 60 | 60 | **13** | **60** |
| K = 8 s1 | 60 | 60 | 60 | **2** | **60** |
| K = 8 s2 | 60 | 60 | 60 | 60 | **60** |

**Eight of nine cells collapse to 2–13 tokens immediately after the transition**,
including every K = 8 cell but one. So a narrowed output at step 30,000 does not
predict the final outcome either.

**It also refutes a hypothesis I would otherwise have reached.** A degenerate
output is not an absorbing state: **K = 8 seed 1 fell to 2 distinct tokens — below
the Addendum 03 collapse threshold — and recovered to 60 tokens and 90.4297 %
exact match.** Any account of the K = 4 failure that appeals to collapse being
irreversible is wrong.

## 4. The discriminator: whether a cell ever gets below chance loss

Minimum training loss reached anywhere in the 60,000-step T = 4 target stage,
against chance = **4.0943**:

| cell | min train loss | below chance? | final T = 4 exact match |
|---|---|---|---|
| K = 1 s0 | 4.0952 | **no** | 0.0000 % |
| K = 1 s1 | 4.0952 | **no** | 0.0000 % |
| K = 1 s2 | **3.9089** | yes | 0.0000 % (5.75 % token) |
| K = 4 s0 | 4.0952 | **no** | 0.0000 % |
| K = 4 s1 | 4.0952 | **no** | 0.0000 % |
| K = 4 s2 | 4.0953 | **no** | 0.0000 % |
| K = 8 s0 | **0.0033** | yes | **98.7793 %** |
| K = 8 s1 | **0.0159** | yes | **90.4297 %** |
| K = 8 s2 | **0.0395** | yes | **84.1309 %** |

The split is exact and binary. **Every cell that got below chance went on to learn
something; no cell that failed to ever did — in 60,000 steps.**

**This describes the learned/not-learned split precisely and is not itself a
mechanism.** "Never got below chance" and "never learned" are close to the same
statement about a cross-entropy objective, so the criterion separates the cells
cleanly without explaining anything: the open question is why K = 8 escapes the
plateau and K = 4 does not, and §6 is what would be needed to answer it. The five failing
cells' minimum is 4.0952–4.0953, which is *above* ln 60 by 0.0009: they approach
the uniform baseline from above and stop there.

And the approach is glacially slow. K = 4's train loss over steps 32,000–44,000:

```
s0:  4.0972  4.0968  4.0965  4.0962  4.0958  4.0957  4.0955
```

0.0017 nats in 12,000 steps, with grad max flat at 0.43–0.52 and declining. By
contrast K = 8 seed 0 over the same steps: 4.0968 → 3.9548 → 3.0628 → 2.7159, with
grad max climbing 0.48 → 2.729 as it escapes.

## 5. So the collapse is the symptom and the plateau is the thing

The causal order the data supports:

1. **The depth switch knocks every cell onto the chance plateau** (§2), because a
   model converged on F³ is confidently wrong about F⁴ — its T = 4 validation loss
   at step 25,000 is 33–36 nats, not 4.09.
2. **At the plateau, narrowing the output is loss-optimal.** With no
   input-dependent signal, the best constant prediction is the modal token, worth
   1.9244 % against uniform 1.6667 %. The eight cells that narrow are doing the
   correct thing for a model that has no signal.
3. **Some cells find a below-chance direction and escape** (§4). K = 8 does so in
   3 of 3, within roughly 5,000–15,000 steps of the transition.
4. **Cells that do not remain pinned**, and their output narrows further to the
   single majority token — which is where the 1.9244 % and the 1 distinct token
   in the report's K = 4 rows come from.

**This is the local optimum [the Batch 01 erratum](PHASE2_BATCH_01_REPORT_ERRATUM_01.md)
§3 predicted, observed directly.** That erratum argued from the P1c
partial-credit ceilings that at T = 4 "the loss rewards partial knowledge almost
nothing, so near-uniform output is a local optimum… the T = 4 stage moves away
from the earlier solutions because they are worse under the T = 4 loss than
uniformity is." Batch 06 shows exactly that happening, and shows K = 8 escaping
it while K = 4 does not.

So the report's framing — that the longer T = 3 rung "destroyed" K = 4 — should be
read more precisely: **the rung did not destroy anything; the depth transition
puts every cell on the plateau, and K = 4 could not leave it.** Whether the longer
rung made leaving harder is a separate question this note cannot answer (§6).

## 6. What is still unexplained, and what would be needed

**No mechanism for why K = 4 cannot escape.** The logs contain train loss,
gradient max, validation loss and accuracy. They do not contain anything that
would distinguish the live candidates:

* *insufficient gradient signal* — but K = 4's grad max (~0.45) is comparable to
  K = 8's before K = 8 escapes (~0.47), so the raw magnitude does not separate them;
* *a worse basin* — would need weight-space measurements, e.g. distance from the
  T = 3 solution, or the loss along the path between a K = 4 and a K = 8 solution;
* *gradient direction rather than magnitude* — would need the gradient's alignment
  with a known descent direction, or its cosine similarity across steps;
* *the longer rung making the T = 3 minimum sharper* — would need curvature at the
  rung's end, e.g. a Hessian-vector-product estimate, compared against Batch 04's
  shorter-rung cells.

**A cheap next measurement**, if this is worth pursuing: record gradient norm
*per step* rather than per 250, and the rank or entropy of the output
distribution, through the 2,000 steps either side of the transition. That would
at least order the loss spike against the gradient spike, which §2 cannot.

**Confounds that remain.** The learning rate is 1e-3 everywhere and inherited
([Batch 05 Note 01](PHASE2_BATCH_05_REPORT_NOTE_01.md)); K = 4's inability to
escape a plateau is exactly the kind of thing a rate could change, and no rate has
been searched. And steps are matched while compute is not, so K = 8's advantage
here includes eight times the block executions.

## 7. Limits of this diagnosis

* **250-step resolution** on train loss and gradient, so §2 cannot order the two
  spikes.
* **Distinct tokens at five boundaries only**, so §3's collapse is located to
  within 5,000 steps, not to the step.
* **No per-parameter or weight-space statistics** at all (§6).
* **Three seeds per K**, and the K = 1 row is 2–1 rather than 3–0, so "recovery is
  K-dependent" rests on 9 cells.
* Nothing here is a new measurement; it is a reading of what was already recorded.
