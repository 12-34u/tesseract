# Proposed Addendum 02 — Arm 2c uses non-affine per-step normalisation

**STATUS: PROPOSAL. NOT PRE-REGISTERED. NOT IN FORCE.**

Drafted 2026-09-28 on the `mac-diagnostics` branch, at the user's request, for a
decision to be taken on the GPU box **before Arm 2c runs**. Nothing in
[PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md) — including Addendum 01 —
has been edited, and no file under `configs/phase2/` has been touched. If this
proposal is accepted it must be appended to the pre-registration there, by the
same rule Addendum 01 followed: appended below the existing text, editing nothing
above it.

If it is rejected, Arm 2c runs as Addendum 01 already specifies, with affine
normalisation. Nothing here blocks that.

---

## 1. What is proposed

> Arm 2c's per-step normalisation of z_L and z_H is **non-affine**: the
> LayerNorms carry no learnable gain and no learnable bias
> (`elementwise_affine=False`).

Everything else in Addendum 01 §A1.2 stands: the normalisation is applied to both
states after every block call, inside the recursion, and the recursion is
otherwise the recorded update rule with no `.detach()`.

## 2. Why — the argument

Addendum 01 §A1.3 requires, of any fix recorded as *the* residual fix, **"a fix
that bounds hidden-state growth, verified by per-step norms"**.

An affine per-step normalisation satisfies that requirement *as measured on an
untrained model*, and only there. Its gain is a free parameter. Nothing in the
training objective prevents the learned gain from growing with depth, and if it
does, the bound that justified the fix is gone — while the fix is still nominally
in place and still reported as satisfying §A1.3. The verification would then have
the same defect as the one Addendum 01 was written to correct: a property
confirmed at the moment of measurement that does not hold where it matters.

A non-affine normalisation closes this. With no gain and no bias, the output of
the normalisation has unit RMS **by construction, at every step of training**,
not merely at initialisation. §A1.3 is satisfied once, structurally, and does not
need to be re-verified at the end of every run.

It is also free. Non-affine normalisation adds **zero parameters** at every K,
which is a stronger position on the project's central claim than the affine
version's 1024: the parameter count stays not just K-invariant but *identical to
the unfixed model*, so no reader can ask whether the fix bought its effect with
the parameters it added, and the existing parameter-matched comparisons need no
adjustment.

## 3. The measurement

**Mac CPU diagnostic, verified 2026-09-28.** Untrained models, medium width
(d_model 256), 256 examples from the T=4 validation split, seed 0 before each
build, torch 2.13.0 on CPU — the conditions of
`scripts/phase2_residual_norms.py`, so the OFF rows are comparable to Addendum
01's table. Produced by `diagnostics/run_per_step_norm_rms.py`.

Per-step z_L RMS:

| condition | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | growth |
|---|---|---|---|---|---|---|---|---|---|
| K=1, OFF | 1.43 | — | — | — | — | — | — | — | 1.00× |
| K=2, OFF | 1.43 | 3.00 | — | — | — | — | — | — | 2.10× |
| K=4, OFF | 1.43 | 3.00 | 4.86 | 7.16 | — | — | — | — | 5.01× |
| K=8, OFF | 1.43 | 3.00 | 4.86 | 7.16 | 10.09 | 13.89 | 18.83 | 25.30 | 17.70× |
| K=1, ON affine | 1.00 | — | — | — | — | — | — | — | 1.00× |
| K=2, ON affine | 1.00 | 1.00 | — | — | — | — | — | — | 1.00× |
| K=4, ON affine | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — | 1.00× |
| K=8, ON affine | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00× |
| K=1, ON **affine=False** | 1.00 | — | — | — | — | — | — | — | 1.00× |
| K=2, ON **affine=False** | 1.00 | 1.00 | — | — | — | — | — | — | 1.00× |
| K=4, ON **affine=False** | 1.00 | 1.00 | 1.00 | 1.00 | — | — | — | — | 1.00× |
| K=8, ON **affine=False** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00× |

Final-step figures at more precision, with the parameter cost:

| K | condition | z_L RMS | z_H RMS | max abs z_L | + parameters |
|---|---|---|---|---|---|
| 1 | OFF | 1.429815 | 0.144418 | 7.00 | 0 |
| 2 | OFF | 3.002617 | 0.429528 | 14.68 | 0 |
| 4 | OFF | 7.163558 | 1.501595 | 34.96 | 0 |
| 8 | OFF | 25.303076 | 7.070640 | 123.40 | 0 |
| 1 | ON affine | 0.999997 | 0.999758 | 4.80 | 1024 |
| 2 | ON affine | 1.000000 | 0.999997 | 4.77 | 1024 |
| 4 | ON affine | 1.000000 | 0.999997 | 4.76 | 1024 |
| 8 | ON affine | 1.000000 | 0.999997 | 4.78 | 1024 |
| 1 | ON affine=False | 0.999997 | 0.999758 | 4.80 | **0** |
| 2 | ON affine=False | 1.000000 | 0.999997 | 4.77 | **0** |
| 4 | ON affine=False | 1.000000 | 0.999997 | 4.76 | **0** |
| 8 | ON affine=False | 1.000000 | 0.999997 | 4.78 | **0** |

Across K ∈ {1, 2, 4, 8}, final-step z_L RMS varies by **0.00 %** under both ON
conditions and by **1669.68 %** under OFF.

### 3.1 The measurement cannot decide this, and that is the point

**The affine and non-affine variants are numerically identical here.** Maximum
absolute difference in per-step z_L RMS across all K: **0.0**. This is expected
and not a defect — at initialisation a LayerNorm's gain is 1 and its bias 0, so
an affine norm *is* a non-affine norm until training changes it.

So no untrained measurement can distinguish the two, and none should be cited as
doing so. The case for `affine=False` rests entirely on what the two permit
*during* training: one can drift, the other cannot. That is an argument about
guarantees, not about measured values, and the decision has to be made on those
terms.

### 3.2 Supporting checks

* **OFF reproduces Addendum 01's recorded table exactly** — 1.43 through 25.30,
  growth 17.70× — so the harness agrees with the GPU box on the shared condition.
* **OFF is bit-identical to the frozen recursion**, asserted at `atol=0` against
  `models/recursive_core.py`; with the flag off nothing is wrapped.
* **The affine implementation agrees exactly with `phase2/batch01.py`'s
  `PerStepNormalisedReasoner`** — maximum absolute per-step z_L RMS difference
  **0.0** — from an independent re-derivation of the recursion, so this is a
  check on both.
* **Gradients still reach step 1** with the flag on, in both variants; no
  `.detach()` was introduced.
* 14 tests over `diagnostics/per_step_norm.py`; full suite 756 passed, 1 skipped.

## 4. The trade-off, stated plainly

**What is given up: a learned per-feature rescaling inside the recursion.**

With `elementwise_affine=True`, the recursion can learn how much to amplify or
suppress each feature after each block call, and can in principle learn a scale
that suits depth — a mild, shared-across-steps form of adaptivity, at 1024
parameters. With `affine=False` that capacity is removed. Every step's output is
forced to unit RMS with no scope to adjust, and any rescaling the model needs must
be absorbed by the shared block's own weights, which are also doing every other
job in the recursion.

This could plausibly hurt. Two ways it might:

1. **It could suppress a real effect.** If depth helps only once the stream can be
   rescaled, a non-affine Arm 2c could return NOT LEARNABLE where an affine one
   would not — and the batch would then record a null result caused by the fix
   rather than by the architecture.
2. **It is a strictly stronger constraint than the confound requires.** The
   confound is that the stream grows *with K*. Bounding the norm to exactly 1.0 at
   every step is sufficient for that but not necessary; a fix that merely kept the
   norm K-independent would also do, with more freedom.

**Why the proposal still favours `affine=False`:** Arm 2c is a diagnostic whose
job is to remove a confound so that the K axis can be read, not to maximise the
chance of a positive result. A fix with a guarantee that holds throughout training
is worth more to that job than 1024 parameters of adaptivity — and the failure
mode of the affine version is the more dangerous one, because it fails *silently*,
by reporting a satisfied §A1.3 that no longer holds.

The risk in (1) is also testable rather than fatal: if non-affine Arm 2c returns
NOT LEARNABLE, an affine re-run distinguishes "the fix suppressed it" from "there
was nothing there", and the two differ by one flag.

## 5. If this is rejected — what Arm 2c must then do

If Arm 2c runs with affine normalisation, Addendum 01 §A1.3 is **not** satisfied
by the untrained measurement above, and the arm must additionally:

* re-measure per-step z_L and z_H RMS **at the end of training**, at every K, on
  the same 256 validation inputs; and
* report the learned LayerNorm gain norms per K, so that any drift is visible;
  and
* state explicitly whether the bound still holds post-training, because only then
  may the fix be recorded in Amendment 02 as **the** residual fix.

Under the non-affine proposal, none of that is needed: the bound is structural.
The post-training measurement is still worth taking as a check, but nothing rests
on it.

## 6. What this proposal does not do

* It does not alter the §10 Stage 3 table, or Addendum 01 §A1.3's wording.
* It does not alter the benchmark, the group, T, K, the metrics, τ, the seeds, the
  step budget or the width.
* It does not change Arm 2b. Arm 2b still tests read-out normalisation and still
  must not be described as removing the confound.
* It does not claim any Arm 2c result. **No Arm 2c run exists.** Everything in §3
  is an untrained structural measurement labelled `mac_cpu_diagnostic`, and it
  must not be entered in the Batch 01 report as an arm outcome.

## 7. Implementation note

The flag exists and is tested in `diagnostics/per_step_norm.py`
(`PerStepNormConfig.elementwise_affine`), which is diagnostic code on this branch
and is **not** what Arm 2c would run. Adopting this proposal means changing
`PerStepNormalisedReasoner` in `phase2/batch01.py` to construct its two
`nn.LayerNorm`s with `elementwise_affine=False`, and updating
`per_step_norm_parameter_cost` there to return 0. That edit is deliberately not
made here: `phase2/` is the GPU box's, and this branch does not touch it.
