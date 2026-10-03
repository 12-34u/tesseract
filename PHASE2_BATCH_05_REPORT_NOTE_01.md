# Note 01 — the learning rate was tuned for arm A, and that confounds "slower"

**Date: 2026-10-03.** A dated note on
[PHASE2_BATCH_05_REPORT.md](PHASE2_BATCH_05_REPORT.md) (`9891003`). That file is
not edited and stands as committed. **No classification and no number changes:**
this records a confound the report did not name.

## 1. The confound

All three arms trained with **AdamW, learning rate 1e-3 constant, weight decay
0.01, batch 128** — identical settings, hardcoded in
`scripts/phase2_batch05.py` exactly as in `phase2_batch01.py` and
`phase2_batch03.py`.

**Those settings were not chosen for arms B and C.** The 1e-3 traces back to
[PHASE2_BATCH_01_PREREG.md](PHASE2_BATCH_01_PREREG.md), which records the peak
learning rate as "**unchanged from REF**" — and REF is Probe A, a `tesseract`
shared-block model at 837,436 parameters. So the optimiser settings were
inherited, through five batches, from a configuration tuned on **arm A's
family at arm A's scale.**

Arms B and C hold **3.8× more parameters** (3,214,678 and 3,206,716 against
837,436). Larger models commonly prefer smaller learning rates, and nothing in
this project has ever searched a learning rate for either family.

**So the report's central framing is partly confounded.** §4 says the shared
recursion "finds the composable solution far faster than either alternative". The
alternative explanation the report does not consider is that arms B and C are
slower *at a learning rate selected for arm A*, and would be faster at one
selected for them.

## 2. Why the §4 result is unlikely to be learning rate alone

Three reasons, all from data already in the report. None of them is a test.

**The alternatives were at chance, not merely slower.** At the end of the
5,000-step T = 3 rung, arms B and C sat at 1.5453–1.7233 % against uniform chance
of 1.6667 % — all six cells within **0.08 points of chance**. A suboptimal but
reasonable learning rate normally slows convergence; it does not usually produce
a result indistinguishable from no learning at all.

**The same learning rate demonstrably works for those models at lower depths.**
Every cell of every arm solved its T = 1 and T = 2 rungs at 1e-3:

| arm | T = 1 rung-end exact match | T = 2 rung-end exact match |
|---|---|---|
| A | 100.00 / 100.00 / 100.00 % | 100.00 / 100.00 / 100.00 % |
| **B** | **100.00 / 100.00 / 100.00 %** | **100.00 / 99.95 / 100.00 %** |
| **C** | **100.00 / 100.00 / 100.00 %** | **99.95 / 100.00 / 100.00 %** |

So 1e-3 is not globally mis-set for the 3.2M-parameter models: at the same rate,
in the same run, they learn depth-1 and depth-2 composition perfectly. What
changes at T = 3 is the task, not the optimiser.

**Arms B and C did eventually learn at this rate**, in the target stage — arm B
to 70.4 % token accuracy and arm C to 82.8 %, both still rising. A learning rate
too large to make progress would not produce that; a learning rate merely
suboptimal is consistent with it.

## 3. What is nevertheless untested

**The optimal learning rate may be depth-dependent.** T = 3 is a harder
objective than T = 2 — the P1c partial-credit analysis shows the gradient toward
partial knowledge thinning with depth — so a rate that suffices at T = 2 is not
guaranteed to be near-optimal at T = 3, and §2's second argument is weaker than
it looks for exactly that reason.

**No learning rate has ever been searched, for any family.** The project has one
rate, inherited. It is entirely possible that arm B or arm C at 3e-4 crosses the
shortcut ceiling inside its T = 3 rung, which would change §4's conclusion from
"the shared recursion finds it faster" to "the shared recursion finds it faster
*at this learning rate*".

**So §4's claim should be read with the qualifier attached**: at matched forward
compute **and at a learning rate inherited from arm A's family**, the shared
recursion finds the composable solution far faster than either alternative. The
per-arm results in §1 and §2 of the report are unaffected — they are per-arm, not
comparative — and so is arm A's 100 % exact match.

[PHASE2_BATCH_07_DRAFT.md](PHASE2_BATCH_07_DRAFT.md) is designed to settle it.

## 4. A provenance gap, found while checking this

**The optimiser configuration is not recorded in any batch report.** The
`config` block of `t3_family_report.json` holds the arms, curriculum, stages,
budget, seeds, width and fix — but not the optimiser, learning rate, weight decay
or batch size. Those exist only in the runner source.

That is the same class of defect as Batch 01's recorded deviation (a measurement
living in console output rather than a report field): a reader cannot check from
the artifact what learning rate produced it, and I could only establish it for
this note by reading the code and tracing the value back through
pre-registrations.

It affects Batches 01 through 06, whose reports are written or whose runner is
fixed, so it is recorded rather than retrofitted. **Any successor batch should
write the resolved optimiser configuration into its report**, and
[PHASE2_BATCH_07_DRAFT.md](PHASE2_BATCH_07_DRAFT.md) §6 includes it — which it
must, since it varies the learning rate across cells and the artifact would
otherwise not say which cell got which.
