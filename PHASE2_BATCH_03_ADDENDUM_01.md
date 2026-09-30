# Addendum 01 — Batch 03 was interrupted by a reboot and is resumed

**Date: 2026-09-30. Committed before the resumed run starts.**

A separate dated file, as the other addenda are.
[PHASE2_BATCH_03_PREREG.md](PHASE2_BATCH_03_PREREG.md) (`db39663`) is not edited
and stands as committed.

This records a **deviation in provenance, not in design**. No cell's
configuration, budget, seed or classification rule changes. What changes is where
some recorded numbers come from, and that has to be visible in the report rather
than inferred.

## 1. What happened

Batch 03 was launched at 13:37 and the machine rebooted at **21:36** and again at
**21:56** (`last -x reboot`). The service was killed; the chain log holds a
`START` with no `END`, and no `t3_k_sweep_report.json` was written, because the
report is only written after all twelve cells complete.

State at the interruption:

| | |
|---|---|
| Cells finished, with both checkpoints on disk | **9** — K = 1, 2 and 4, all three seeds each |
| Bounded-growth gate artifact | written and intact, all four K at 1.0000× |
| K = 8 seed 0 | killed ~35 min into a ~46 min cell; **no checkpoint written** |
| K = 8 seeds 1 and 2 | never started |
| Report | not written |

## 2. Why a resume rather than a clean rerun

A clean rerun of all twelve cells costs ≈ 4.80 GPU-h and would make every cell's
provenance uniform. A resume costs ≈ 2.3 GPU-h and leaves nine cells with one
field group sourced differently. The user chose the resume, and §4 states the
cost precisely so the choice is auditable.

The alternative was not free of risk either: rerunning the nine finished cells
would produce *different* numbers from the ones already reported in this
conversation, since seeds fix initialisation but the interrupted run's results
are already on disk and in the log. Re-deriving them from the surviving
checkpoints keeps the record consistent with what was measured.

## 3. What is recomputed, and what is salvaged

For each of the nine finished cells, everything measured **on the final model**
is recomputed from its `best_checkpoint.pt`, so those fields are artifact-backed
exactly as a freshly trained cell's are:

`best_val` (accuracy and loss), `best_val_distinct_predicted_tokens`,
`best_val_shortcut_agreement_percent`, `best_checkpoint_validation_by_t`,
`post_training_per_step_rms`, `post_training_growth_ratio`,
`post_training_bound_held`.

What cannot be recomputed is anything measured **during** training, because it
was taken at steps 5,000 / 10,000 / 15,000 / 35,000 and the report that would
have held it was never written. Those come from the console log:

`curve`, `stage_boundary_validation`, `steps_run`, `final_val.loss`.

Every recovered cell carries `source: "recovered_from_checkpoint"` plus explicit
`recomputed_from_checkpoint` and `recovered_from_console_log` field lists, so a
reader does not have to infer which is which.

### 3.1 The integrity check that makes this admissible

A console log is not an artifact: no checksum, no run id, and nothing ties it to
the checkpoints beyond the directory it sat beside. The one thing that can be
checked is whether the two agree, and it is checked per cell:
the recomputed best-validation accuracy is compared against the accuracy the log
printed, and the delta is recorded in
`console_vs_recomputed_accuracy_delta`.

Verified before committing this file, on two cells:

| cell | recomputed | console log | delta |
|---|---|---|---|
| K = 1 seed 0 | 64.5393 % | 64.5393 % | **0.0000** |
| K = 4 seed 0 | 100.0000 % | 100.0000 % | **0.0000** |

A non-zero delta would mean the checkpoint and the log are not from the same
run, which is the one failure a resume must not paper over. It is printed with
`*** MISMATCH ***` above 0.01 points.

## 4. What is permanently lost

The salvaged stage-boundary rows are a strict **subset** of the recorded field.
The log printed, per depth, token accuracy, loss and shortcut agreement. It did
not print:

* `exact_match`
* `distinct_predicted_tokens`
* `majority_class_percent`

These are gone for the nine recovered cells and are named in each cell's
`stage_boundary_missing_fields`. The three K = 8 cells will have them, so the
report will have an asymmetry across the K axis in exactly these three fields.

**None of them enters any classification.** The Addendum 03 collapse check uses
`best_val_distinct_predicted_tokens`, which is recomputed from the checkpoint
and present for every cell; the majority-class floor is a property of the split,
not of a cell; and exact match is not in any bar. So the loss is to the
descriptive stage-by-stage picture, not to any verdict.

This is the same defect Batch 01's recorded deviation was about — a measurement
living in console output instead of a report field — arriving a second time by a
different route. §5 closes it.

## 5. What is fixed so this cannot recur

`scripts/phase2_batch03.py` now writes `t3_k_sweep_partial.json` after **every
cell**, holding every record and detail entry accumulated so far. The cost is
milliseconds per cell. Had it existed, this addendum would not have been needed:
nine cells of stage-boundary data would have been on disk as an artifact when the
machine rebooted.

The same gap exists in `scripts/phase2_batch01.py` and
`scripts/phase2_batch02.py`, which also write only at the end. They are not
edited here — Batch 01 and Batch 02 have reported and their artifacts are
complete, so changing them now would alter committed code for no benefit — but
any future batch should write incrementally.

## 6. What does not change

The configuration, the 35,000-step budget, the curriculum, the seeds, the K
range, the bars, the majority-class floor, the collapse threshold, the shortcut
attribution thresholds, the §7 trend reading and the §8 outcome table are all
untouched. The bounded-growth gate is re-run on resume and must pass again
before any cell trains.

Nothing here makes a positive result easier to obtain.
