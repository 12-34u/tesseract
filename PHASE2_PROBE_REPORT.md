# Phase 2 — P1c probe report (Probes A and B)

**Date: 2026-09-27.** Classification applied mechanically by
[phase2/probes.py](phase2/probes.py) against the decision rule committed in
[PHASE2_PROBE_PREREGISTRATION.md](PHASE2_PROBE_PREREGISTRATION.md) (commit
`da95a8e`, 2026-09-24 14:45:16 UTC) **before either probe was executed**.

Artifacts: `runs/phase2/probe_a/probe_report.json`,
`runs/phase2/probe_b/probe_report.json`.

Reference values: chance token accuracy **1.6667 %** (= 100/60);
chance loss **ln 60 = 4.09434** nats. LEARNABLE bars: accuracy **≥ 6.6667 %**
*or* loss **≤ 3.99434** nats, in at least 2 of 3 seeds.

## 1. Conditions

| | Probe A | Probe B |
|---|---|---|
| Depth | T = 4 | T = 8 |
| Width | medium (837,436 params) | medium (837,436 params) |
| K | 1, 8 | 1, 8 |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Budget | 20,000 steps | 20,000 steps |
| Start | cold | warm-started from a T=1 solution at the same K |
| Data | frozen validation split only; test never read | same |
| Platform | RTX 3050, torch 2.13.0+cu129, deterministic kernels, TF32 off | same |

## 2. Per-seed results

Accuracy is validation token accuracy (%) at the best-validation checkpoint.
"Bar" is that run's own above-chance threshold: chance + *z*·SE with
*z* = Φ⁻¹(1 − 0.05/M) and M = 80 evaluations, SE computed per-example.

### Probe A — T = 4, cold start

| K | seed | best acc | bar | best loss | final loss | above chance |
|---|---|---|---|---|---|---|
| 1 | 0 | 1.8612 % | 1.8996 % | 4.0955 | 4.0960 | no |
| 1 | 1 | 1.8296 % | 1.8977 % | 4.0967 | 4.0960 | no |
| 1 | 2 | 1.8497 % | 1.8978 % | 4.0959 | 4.0959 | no |
| 8 | 0 | 1.8066 % | 1.8967 % | 4.0996 | 4.0972 | no |
| 8 | 1 | 1.8612 % | 1.8985 % | 4.0979 | 4.0987 | no |
| 8 | 2 | 1.8670 % | 1.9041 % | 4.1002 | 4.0981 | no |

### Probe B — T = 8, warm-started

| K | seed | best acc | bar | best loss | final loss | above chance |
|---|---|---|---|---|---|---|
| 1 | 0 | 1.8813 % | 1.8974 % | 4.0984 | 4.0967 | no |
| 1 | 1 | 1.8411 % | 1.9042 % | 4.1043 | 4.0969 | no |
| 1 | 2 | 1.8555 % | 1.8957 % | 4.1026 | 4.0961 | no |
| 8 | 0 | 1.7923 % | 1.9012 % | 4.1060 | 4.0973 | no |
| 8 | **1** | **1.9273 %** | **1.9069 %** | 4.1018 | 4.0979 | **yes** |
| 8 | 2 | 1.8181 % | 1.8963 % | 4.0978 | 4.0963 | no |

## 3. Mechanical classification

| Probe | K | Classification |
|---|---|---|
| A (T = 4) | 1 | **NOT LEARNABLE** |
| A (T = 4) | 8 | **NOT LEARNABLE** |
| B (T = 8) | 1 | **NOT LEARNABLE** |
| B (T = 8) | 8 | **PARTIAL** |

No cell in either probe met a LEARNABLE bar. No accuracy came within 4.7
percentage points of the 6.6667 % bar, and no loss fell below ln 60 at all —
every final loss sits *above* chance loss. The single PARTIAL arises solely
because one seed crossed its above-chance threshold.

## 4. Interpretation

**The single seed clearing its bar is consistent with noise.** Probe B K=8
seed 1 exceeded its threshold by **0.0204 percentage points** (1.9273 % against
a 1.9069 % bar). Three things make this uninformative:

1. **The expected number of such crossings is close to one.** The above-chance
   bar is set at a per-cell false-positive rate of 0.05, already corrected for
   taking the maximum over M = 80 evaluations. Twelve cells were tested across
   the two probes, so the probability of at least one crossing under the pure
   null is 1 − 0.95¹² ≈ **46 %**, with an expected count of 0.6. Observing
   exactly one is the modal outcome of no effect at all.
2. **The loss arm contradicts it.** That cell's final validation loss is
   4.0979 nats, *above* ln 60 = 4.0943. A model with genuine partial signal
   would be expected to reduce loss below chance, not sit above it.
3. **It is not the deepest or the most favoured cell.** The other two seeds at
   the same K and T scored 1.7923 % and 1.8181 %, both below their bars and
   below several K=1 cells. There is no ordering by K or by seed that the
   crossing fits.

**The result is flat across both axes.** Every one of the twelve cells sits
within roughly 0.13 percentage points of chance, at both T values, at both K
values, cold and warm started. Nothing distinguishes K=8 from K=1, and nothing
distinguishes T=4 from T=8.

**The warm start transferred nothing, which is what makes Probe B
interpretable.** Each stage-1 T=1 solution reached 100 % validation exact match
(stopping early at step 500), and each then scored **1.6918 %** on the T=8
validation split before any T=8 training — chance, to four figures — at a loss
of roughly 15.5 to 16.2 nats. So Probe B's stage-2 numbers are not inherited
T=1 structure; the warm start genuinely began from a solved neighbour and still
went nowhere.

**Residual-stream growth with K is confirmed and remains a confound.** The
pre-registered auxiliary measurement records step-0 validation loss of
**4.43 nats at K=1 and 33.97 nats at K=8**. K=8 therefore starts an order of
magnitude further from chance than K=1, reproducing the Phase 1 finding that
without a final LayerNorm the residual stream grows with the number of block
executions. Any future K comparison must account for this; it is one of the two
open questions Amendment 02 must settle before it is frozen.

## 5. What this does and does not establish

**Establishes:** neither primary depth condition is learnable by a
medium-width model from a cold start within 20,000 steps, and T=8 is not
reachable even when started from a solved T=1 solution at the same K.

**Does not establish:** that the benchmark is infeasible. A result that is
uniformly flat across *both* the depth axis and the recursion axis does not
discriminate between "this architecture cannot express the function at this
scale" and "this task provides no usable gradient from a cold start". The
second is the weaker claim, and it is the one the P1c signal analysis already
points at: predicting a target from all but one input cell gives 26.7 %, from
all but two 13.4 %, from all but three 7.0 % — the signal lives at near-full
interaction order, so there is almost no partial-credit gradient to climb.

Under the plan's fork, that is evidence for the **signal / optimisation** path,
whose pre-registered remedy is Amendment 03's curriculum over T. Infeasibility
is concluded only if that remedy also fails.

**No benchmark, model, metric or decision rule was changed in response to these
results.** Both probe runs and all their artifacts are preserved, including the
two interrupted attempts of the final Probe B cell, which were retried in
sibling directories rather than overwritten.
