# Addendum 02 — Arm 2c uses non-affine per-step normalisation

**Date: 2026-09-29. Committed before the chain reached Arm 2c.**

A separate file by design: [PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md)
(`7dbd7b6`) and its Addendum 01 (`b64305f`) are not edited. Both stand as
committed, including the parts this supersedes.

Adapted from `PROPOSED_ADDENDUM_02_DRAFT.md` on the `mac-diagnostics` branch,
with the pre-LN caveat in §4 added.

## 1. What changes

> Arm 2c's per-step normalisation of z_L and z_H is **non-affine**: the
> LayerNorms carry no learnable gain and no learnable bias
> (`elementwise_affine=False`).

Everything else about Arm 2c is unchanged — the ladder of measurements, the
bounded-growth gate at 1.05, the 20,000-step training run at T = 4, K ∈ {1, 8},
3 seeds, and the §3 success rule.

## 2. Why

Addendum 01 §A1.3 requires, of any fix recorded as *the* residual fix, **"a fix
that bounds hidden-state growth, verified by per-step norms"**.

An affine per-step normalisation satisfies that *as measured on an untrained
model*, and only there. Its gain is a free parameter. Nothing in the training
objective prevents the learned gain from growing with depth, and if it does the
bound that justified the fix is gone — while the fix is still nominally in place
and still reported as satisfying §A1.3.

That is the same defect Addendum 01 was written to correct: a property confirmed
at the moment of measurement that does not hold where it matters. Having caught
it once in the output-side norm, it should not be reintroduced one layer down.

A non-affine normalisation closes it. With no gain and no bias, the output has
unit RMS **by construction at every step of training**, not merely at
initialisation.

It is also free. Non-affine normalisation adds **zero parameters** at every K,
so the parameter count stays not just K-invariant but *identical to the unfixed
model*. No reader can ask whether the fix bought its effect with the parameters
it added, and the existing parameter-matched comparisons need no adjustment —
the `match_parameters` correction recorded in Addendum 01 §A1.4 stays in the
code for the affine case but is not exercised by Arm 2c.

## 3. Measured

Untrained, medium width, 256 validation inputs, seed 0. z_L RMS at the first and
last recursion step.

| K | first | last | growth | parameters added |
|---|---|---|---|---|
| 1 | 0.999997 | 0.999997 | 1.0000× | **0** |
| 2 | 0.999997 | 1.000000 | 1.0000× | **0** |
| 4 | 0.999997 | 1.000000 | 1.0000× | **0** |
| 8 | 0.999997 | 1.000000 | 1.0000× | **0** |

These reproduce the `mac-diagnostics` figures to six decimal places on different
hardware. For comparison, the unfixed model grows 17.70× at K = 8 and the
output-side norm leaves that growth untouched.

## 4. What this does **not** claim

**Per-feature rescaling remains available to the model.** Removing the gain from
the per-step norms bounds the *norm* of the state; it does not remove all
learned rescaling from the recursion. The shared `TransformerBlock` is pre-LN
and its own LayerNorms keep their learnable gains, so the block can still
rescale individual features however training finds useful, including in ways
that vary across the feature dimension.

What the non-affine per-step norm guarantees is narrower and is exactly what
§A1.3 asks for: the RMS of z_L and z_H entering each execution is unity at every
step of training, so the *scale* of the residual stream cannot grow with K. It
does not make the recursion scale-free, and no claim to that effect may be made
from it.

## 5. Post-training verification

Because the bound is structural, §A1.3 no longer needs re-verifying per run.
Arm 2c nonetheless records the per-step RMS of the trained best checkpoint at
K ∈ {1, 8} in its report (`post_training_per_step_rms`,
`post_training_growth_ratio`). If that ever disagrees with the structural
argument, the argument is wrong and the report must say so.

## 6. Isolation

The change reaches Arm 2c and nothing else. Arms 2b, 3 and 4 — running or queued
when it landed — resolve to byte-identical settings before and after, asserted
in `tests/phase2/test_batch01.py`:

| arm | fix | curriculum | steps | seeds |
|---|---|---|---|---|
| 2b | `output` | no | 20,000 | 3 |
| 3 | `none` | yes | 35,000 | 3 |
| 4 | `none` | no | 40,000 | 3 |
| **2c** | **`per_step`** | no | 20,000 | 3 |

Arm 2c is the only arm resolving to `per_step`, also asserted. The affine
variant is kept available behind `elementwise_affine=True` so Addendum 01's
specification remains reproducible; it is no longer the default.
