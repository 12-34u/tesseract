# Phase 2 — Pre-registration of the P1c probe decision rule

**Date: 2026-09-24. Committed before Probe A or Probe B was executed.**

This document fixes, in advance, how the two P1c probes will be classified.
It exists so the classification cannot be chosen after the numbers are seen.
The thresholds in §2 are transcribed verbatim from the instruction that
commissioned the probes. They are not to be revised in the light of any probe
output. The machine-readable copy is
[configs/phase2/probe_preregistration.yaml](configs/phase2/probe_preregistration.yaml),
and the rule is applied by code
([phase2/probes.py](phase2/probes.py)), not by hand.

## 1. Why these probes exist

P1b established that the A₅ benchmark at T=8 is not learned, and that this is
**not** explained by model capacity (small and medium widths behave alike), by
training budget (20,000 steps, flat throughout), or by instability. The P1c
signal analysis then showed why: the task carries essentially no low-order
signal. Predicting a target token from all but one input cell gives 26.7 %,
from all but two 13.4 %, from all but three 7.0 % — the signal lives at
near-full interaction order, so there is no partial-credit gradient for a model
to climb.

Before committing the full Phase 2 grid to a task that may be unlearnable at
its primary depths, two questions must be answered:

- **Probe A — is T=4 learnable at all?** T=4 is one of the two primary depth
  conditions under Amendment 01. If T=4 is also at the floor, the grid has no
  informative primary condition and the benchmark must be reconsidered.
- **Probe B — is T=8 reachable from a solved neighbour?** Warm-starting T=8
  from the trained T=1 solution tests whether the barrier is optimisation
  (a bad starting point in a flat landscape) or representational (the function
  is out of reach for this architecture at this size).

Probe B is a **diagnostic only**. Warm-starting must not become the grid's
training protocol; doing so would change what the grid measures.

## 2. The decision rule (verbatim, not to be changed)

Outcomes are defined on held-out data, across 3 seeds, within a fixed step
budget:

- **LEARNABLE** — accuracy ≥ chance (1.67 %) + 5 points, **OR**
  loss ≤ ln 60 − 0.1 nats, in at least 2 of 3 seeds.
- **PARTIAL** — above chance on any seed but below the LEARNABLE bar.
- **NOT LEARNABLE** — indistinguishable from chance on all seeds.

Numerically: chance token accuracy is 100/60 = **1.6667 %**, so the LEARNABLE
accuracy bar is **6.6667 %**; ln 60 = **4.09434**, so the LEARNABLE loss bar is
**3.99434 nats**.

## 3. Assessment of these thresholds

The instruction invited an objection before the fact. I do **not** propose
changing the thresholds. Two observations that should shape how a result is
read, neither of which alters the rule:

1. **The OR is asymmetric, and the loss arm is much the weaker of the two.**
   Five accuracy points at vocabulary 60 is a demanding bar; 0.1 nats is not.
   Because gate G1 confirmed the target distribution is uniform to within
   TV ≤ 0.02, a model cannot buy 0.1 nats from marginal statistics alone — so
   the loss arm does require real conditional signal. But it can be satisfied
   by weak, partial-order signal that never produces a correct token. A result
   that is LEARNABLE **by the loss arm only** should therefore be reported as
   weak partial signal, not as acquisition of the task. The rule still
   classifies it as LEARNABLE; this is an instruction about interpretation.
2. **Three seeds gives limited power, and the 2-of-3 requirement is the right
   conservative choice.** A single seed clearing the bar lands in PARTIAL,
   which is where a one-off should land.

Selection bias was checked rather than assumed: with 2,048 validation examples,
the standard error of mean token accuracy (treating each ring as the
independent unit) is ≈ 0.069 points, so the 5-point bar sits ≈ 70 standard
errors above chance. Choosing the best of ~80 evaluations cannot manufacture
that. The 5-point bar is safe against selection; the "above chance" boundary in
§4 is not, and is corrected accordingly.

## 4. Operational details fixed at the same time

The rule above leaves some things unspecified. These are fixed here, before
execution, and are additions to the rule rather than changes to it:

| Item | Decision |
|---|---|
| Held-out data | the frozen 2,048-example **validation** split. The test split is never read by a probe. |
| Primary checkpoint | best-validation checkpoint within the budget (also reported: final checkpoint) |
| Accuracy | token accuracy (%) on validation |
| Loss | mean per-token cross-entropy in nats on validation |
| Fixed step budget | **20,000 steps**, the same budget as P1b, so probes compare directly to it |
| Seeds | 0, 1, 2 |
| Classification | applied mechanically by `phase2/probes.py` |

**Probe B is two-stage.** No medium-width T=1 solution exists yet (P1's T=1 run
was at the small width), so stage 1 trains the same width and the *same K* on
T=1 with the same optimizer, for at most 20,000 steps, stopping early at 99 %
validation exact match. Its best-validation checkpoint is the warm start for
stage 2, which trains on T=8 for the full 20,000-step budget. Warm-starting each
K from a T=1 solution trained at that same K means the only thing changing
between stages is T. **Only stage 2 numbers are classified.**

**"Above chance"**, which separates PARTIAL from NOT LEARNABLE, is defined as:
token accuracy exceeding chance by more than *z*·SE, where SE is the standard
error of mean token accuracy over the validation split using the **example**
as the independent unit (tokens within one ring are correlated, so treating
tokens as independent would understate SE), and *z* = Φ⁻¹(1 − 0.05/M) with M
the number of validation evaluations in the run. The multiplicity correction is
necessary because the primary checkpoint is the maximum over those M
evaluations; an uncorrected bar would be crossed by selection noise alone. A
run also counts as above chance if its **final** checkpoint validation loss is
at most ln 60 − 0.01 nats — loss is not selected on, so it needs no correction.

## 5. Auxiliary measurements — recorded, not part of the classification

1. **Probe B warm-start reference.** Before any T=8 training step, the
   warm-start checkpoint (the trained T=1 solution) is evaluated on the T=8
   validation split and the result recorded. Probe B is classified mechanically
   on its post-training numbers regardless of this value, but without it a
   warm-started run that merely inherited T=1 structure could not be
   distinguished from one that learned something about T=8.
2. **Step-0 validation loss per K.** Phase 1 established that without a final
   LayerNorm the residual stream grows with K, so initial loss rises with K
   (3.24 at K=1 to 28.31 at K=8). A K=8 run that starts further from ln 60 must
   not be read as evidence about depth.

## 6. What each outcome implies

These are the paths the probes distinguish. Choosing between them is not part
of the mechanical rule and remains a decision to be taken after the report.

- **Probe A LEARNABLE** — T=4 is a usable primary condition. The grid can
  proceed at T=4, with T=8 reported as a floor condition.
- **Probe A NOT LEARNABLE, Probe B LEARNABLE** — the barrier is optimisation,
  not representation. The architecture can express the function but cannot find
  it from a cold start. This is the case Amendment 03 (curriculum) is drafted
  for.
- **Both NOT LEARNABLE** — the benchmark is out of reach at this scale, and the
  depth hypothesis cannot be tested on it as specified. The benchmark, not the
  result, is what would then need to change — and only by an explicit,
  recorded amendment, never by tuning it until it yields a preferred answer.
