# Phase 2 — Batch 05 DRAFT: does shared-weight recursion buy depth, or compute?

**Date: 2026-10-01. DRAFT — not pre-registered, not approved, not run.**

This is a design, written to be read and changed. It is **not** a
pre-registration: nothing here is binding, no outcome table is in force, and if
it is adopted it must be committed as `PHASE2_BATCH_05_PREREG.md` before any cell
runs, as every batch before it has been.

## 1. The question this removes a confound from

Batch 03 found that at T = 3 with the residual stream bounded, K = 4 solves the
task exactly (100 % exact match in two of three cells) while K = 1 reaches 43–65 %
token accuracy and ~0 % exact match. Batch 03 §8 records the caveat that makes
that result incomplete:

> **Nothing at matched compute.** Steps are matched; at 35,000 steps K = 8 does
> eight times K = 1's block executions. K ≥ 4 winning at equal steps is partly a
> compute statement.

So the K = 4 result is consistent with two different claims, and the project's
central hypothesis is only one of them:

1. **Depth**: applying one block four times composes something a single
   application cannot, and the parameter count does not need to grow.
2. **Compute**: four times the forward FLOPs is what did it, and any model with
   that much compute would do as well.

**Batch 05 separates them by holding forward compute fixed and varying how it is
spent.**

## 2. The three arms

All at T = 3, all with the non-affine per-step fix on, all with the curriculum and
budget Batch 03 used. The families already exist in `phase2/models.py`; nothing
new is implemented.

| arm | family | parameters | forward FLOPs / sequence | block executions | d_ff |
|---|---|---|---|---|---|
| **A** shared recursion | `tesseract`, K = 4 | **837,436** | 108,068,864 | 4 | 1024 |
| **B** width-scaled | `width_scaled`, matched to K = 4 | **3,214,678** | 108,077,568 | 1 | **5658** |
| **C** unrolled | `unrolled`, L = 4 | **3,206,716** | 108,068,864 | 4 | 1024 |

Measured from the instantiated models at medium width (d_model 256, 8 heads,
n = 17), not computed by hand:

* **Forward compute is matched to within 0.008 %.** Arm C is exact; arm B is
  +8,704 FLOPs on 108.07 M, which is the granularity of the integer `d_ff` search
  (`width_scaled_d_ff` returns 5658).
* **Arm A uses 3.84× fewer parameters than B and 3.83× fewer than C.**

For reference, the K = 1 baseline that Batch 03 measured is 837,436 parameters and
**27,408,896** FLOPs — a quarter of the compute of all three arms here. It is not
an arm of this batch; it is the number that makes the compute matching meaningful.

### 2.1 What each arm isolates

* **A vs B** — the same compute, spent on four applications of a small block
  versus one application of a wide one. This is the depth question. If A ≈ B, the
  Batch 03 result was about compute. If A > B, recursion is doing something width
  cannot, at 3.84× fewer parameters.
* **A vs C** — the same compute *and* the same number of block executions,
  differing only in whether the four blocks share weights. This isolates weight
  sharing from depth. If A ≈ C, sharing costs nothing and the parameter saving is
  free. If C > A, depth needs independent parameters per step and the project's
  parameter-invariance claim buys less than it appears to.
* **B vs C** — near-identical parameter counts (3.21 M vs 3.21 M) at identical
  compute, differing in whether the compute is one wide layer or four narrow
  ones. A control on the other two.

The three comparisons are not independent, and the batch should report all three
rather than selecting one.

## 3. Why `param_matched` is not an arm

`phase2/models.py` also offers `param_matched` (unrolled with d_model and d_ff
shrunk to match the parameter count). At L = 4 it measures 837,516 parameters and
**27,992,064** FLOPs — parameter-matched but **not** compute-matched, at a quarter
of the compute of arms A–C.

It answers a different question and is documented as carrying a known confound
(its attention head dimension is 16 against Tesseract's 32, accepted under
Amendment 02 D3). Including it here would mix a parameter-matched comparison into
a compute-matched batch. It is worth running as its own arm later, and a reader
comparing to it should note the head-dimension difference.

## 4. Configuration (proposed)

| | value |
|---|---|
| Depth | T = 3 |
| Curriculum | T = 1 → 2 → 3, 5,000 steps per rung, then 20,000 at the target |
| Total steps per cell | 35,000 — **identical to Batch 03**, so arm A is directly comparable to Batch 03's K = 4 cells |
| Seeds | 0, 1, 2 |
| Width | medium base for all arms; B and C derive their shapes from it |
| Residual fix | non-affine per-step normalisation |
| Optimiser | AdamW, constant 1e-3, weight decay 0.01 |
| Batch / eval interval | 128 / every 250 steps |
| Primary metric | **exact match**, with token accuracy reported alongside |
| Classification | the standing rule, plus Addendum 03's floor and collapse check, plus the T = 3 shortcut attribution |

### 4.1 BLOCKING: the fix does not apply to the unrolled family, and that family has the confound

Addendum 01 §A1.4 records that per-step normalisation "applies to the
shared-block family, which is the family that has a recursion", and that the
other baselines "need the equivalent check made separately before any
cross-family comparison under the fix". **That check is now made**, on CPU, at
medium width with the T = 3 validation inputs:

| arm | has `reasoner` | `apply_per_step_norm` | z_L RMS growth |
|---|---|---|---|
| **A** `tesseract` K = 4 | yes | applies → `reasoner` | **1.0000×** over 4 steps |
| **B** `width_scaled` | yes | applies → `reasoner` | **1.0000×** over **1** step |
| **C** `unrolled` L = 4 | **no** | **REFUSES** — `ValueError: model has no 'reasoner' to normalise` | **4.9777×** over 4 steps, unbounded |

Three consequences, and the third blocks the batch as drafted.

**Arm B's fix is vacuous, not absent.** It executes one block, so there is no
between-step growth to bound and its 1.0000× is trivially true. That is fine and
should be stated rather than presented as the fix working there.

**Arm A is as Batch 03 measured it.** Nothing new.

**Arm C cannot currently be run with the fix, and it needs it.** Its hidden state
grows **4.98×** across four blocks — smaller than the shared family's 17.70× at
K = 8 but far outside the 1.05 gate, and on exactly the axis arm C is meant to
control for. Running A (bounded) against C (unbounded) would reproduce the
Batch 02 defect in a new form: a cross-arm comparison with the residual confound
present in one arm only. **Batch 05 must not be pre-registered in that shape.**

Four ways out, for the user to choose:

1. **Implement per-step normalisation for the unrolled family** — the honest fix.
   `PerStepNormalisedReasoner` wraps a `reasoner` and reuses its parameters; the
   unrolled family has no such object, so this is new code: two non-affine
   LayerNorms applied after each of the L blocks. It must be non-affine to stay
   at zero parameters, or arm C's parameter count moves and the comparison with B
   changes. Then re-run the bounded-growth gate on all three arms.
2. **Run A vs B only.** That is still the depth-versus-compute question, which is
   the batch's headline, and both arms are bounded today. It drops the
   weight-sharing question (A vs C) to a later batch.
3. **Run all three and record the asymmetry**, with A vs C explicitly not
   claimed. Cheapest, and the least informative — the comparison the asymmetry
   spoils is the one C exists for.
4. **Bound arm C by its own read-out instead**, which Batch 01 §3 already showed
   does not bound the stream (17.70× unchanged at K = 8). Listed only to record
   that it was considered and rejected.

**Recommendation: option 1**, with option 2 as the fallback if the implementation
turns out to need more than the two LayerNorms. Option 1 is a genuine code
addition to a frozen-adjacent area and should be written test-first, with the
growth measured before and after, exactly as Addendum 02 did for the shared
family.

## 5. Compute estimate

Per-step times are **not** measured yet. Arm A is Batch 03's K = 4 cell at
0.04011 s/step, so 23.4 min per cell and **1.17 GPU-h** for three seeds. Arms B
and C are compute-matched in FLOPs, which makes ~0.04 s/step a reasonable first
guess, but FLOPs do not predict wall clock: arm B's single wide matmul should
reach higher GPU utilisation than four small sequential ones, so it may be
*faster* than A, while arm C's four distinct blocks have the same execution
pattern as A and should be similar.

| arm | basis | estimate |
|---|---|---|
| A | measured (Batch 03 K = 4) | 1.17 GPU-h |
| B | guess, needs measuring | ~1.0–1.2 GPU-h |
| C | guess, needs measuring | ~1.2 GPU-h |
| | **estimated total** | **≈ 3.4 GPU-h** |

Comfortably inside the 8 GPU-h threshold on this estimate, but **the estimate is
two-thirds guesswork** and the measurement takes about two minutes
(`scripts/` has the pattern from Batch 03 and 04). Measure before pre-registering.

Disk: 9 cells × 2 checkpoints. Arms B and C are ~3.2 M parameters, so their
checkpoints are ~12.8 MB rather than 3.2 MB: ≈ 6 × 12.8 + 3 × 3.2 ≈ **87 MB**.

## 6. Outcomes to pre-register (proposed, not in force)

Keyed on **exact match** at T = 3, because Batch 03's Note 01 established that
token accuracy admits models with no correct rings.

| pattern | reading |
|---|---|
| **A ≈ C > B** | Depth, not compute. Four block executions beat one wide layer at equal compute, and sharing weights costs nothing — the project's claim, supported, with A using 3.84× fewer parameters than either alternative. |
| **A ≈ B ≈ C** | Compute, not depth. Batch 03's K = 4 result was the four-fold FLOPs, and recursion is one way to spend them rather than a better one. |
| **C > A** | Depth needs unshared parameters. Recursion gets some of the benefit; the parameter invariance costs accuracy, and how much is the number to report. |
| **B > A** | Width beats depth at equal compute, which would be a negative result for the architecture and should be reported as one. |
| any arm DEGENERATE or below the shortcut ceiling | That arm learned nothing; exclude it from the comparison and report why. |

"≈" needs a threshold fixed in advance. Batch 03's K = 4 cells were at 99.90–100 %
exact match with a spread of 0.1 points, while its K = 1 seed spread was 31
points on token accuracy. A candidate: **arms differ only if their best exact
match differs by more than 5 points**, which is far outside K = 4's observed
spread and far inside the A-vs-K=1 gap. This needs argument before adoption, not
just a number.

## 7. What this batch would still not establish

* **Nothing at T = 4 or T = 8.** T = 3 only, and T = 3 fails G2.
* **Nothing about scaling.** One width.
* **No takeoff probability.** Three seeds per arm, and Batch 03 showed cells at
  the same K can be bimodal.
* **Nothing about the fix's effect on the unrolled family**, until §4.1 option 1
  is implemented and its growth measured after training as well as before.
* **Nothing about training compute.** Forward FLOPs per sequence are matched;
  backward cost and optimiser state differ, and arms B and C hold ~3.8× the
  parameters, so they use more memory and more optimiser compute. Matching
  *forward* compute is a choice, and a reader may reasonably want the total
  matched instead.
* **Nothing about inference latency.** Four sequential block calls and one wide
  call have the same FLOPs and different latency; `forward_flops_per_sequence` is
  recorded but wall-clock per arm should be reported too.

## 8. Open questions for the user

1. **§4.1 is blocking and now measured.** The fix refuses to apply to `unrolled`,
   which has 4.98× growth, so the batch cannot be pre-registered as three
   symmetric arms. Choose among options 1–4; I recommend implementing per-step
   normalisation for the unrolled family (option 1), falling back to an A-vs-B
   batch (option 2).
2. **§5** — measure the per-step times before pre-registering. The current total
   is mostly a guess.
3. **§6** — is "differ by more than 5 points of exact match" the right threshold,
   and is forward-FLOP matching the right notion of equal compute, or should
   total training compute be matched instead?
4. Should `param_matched` (§3) run as a fourth arm in a separate batch?

Nothing in this file is in force. It is committed so that the design can be
reviewed before it becomes a pre-registration.
