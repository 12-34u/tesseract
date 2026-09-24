# Erratum — the Phase 1 frozen-file count

**Date: 2026-09-24.** This erratum corrects a figure reported in earlier Phase 2
progress reports. It adds a record alongside the frozen files; it does not alter
any of them.

## The claim being corrected

Earlier Phase 2 reports stated that **"135 of 136 Phase 1 files are
byte-identical"**, the single exception being `PHASE2_BENCHMARK_DESIGN.md`,
which was being extended with each amendment. That sentence was repeated in
several progress reports, including the project summary.

## Why it cannot be reproduced

The manifest behind that figure was a `sha256sum` file written to the session's
temporary scratchpad directory. That directory is session-scoped and was
deleted when the session ended. **The exact list of files it covered is not
recoverable**, so the number 135 can no longer be checked against anything.

The list was assembled by walking the working tree at a moment *during* Phase 2
and excluding a handful of directories (`.git`, `.venv`, `__pycache__`,
`node_modules`, `runs`). It therefore included the dashboard and whichever
Phase 2 files existed at that moment, which is why it counted more files than
the Phase 1 set actually contains. Reconstructing it today gives a different
total again, because Phase 2 has added files since — which is precisely the
defect: the set was defined by *when the snapshot was taken* rather than by
provenance.

## What replaces it

[phase2/phase1_freeze.json](phase2/phase1_freeze.json) — a committed,
content-addressed manifest of **104 files**, defined by provenance rather than
by timing: every file tracked at the Phase 1 completion commit **`a69054f`**
("Made changes and verified Phase 1 of TESSERACT before starting phase 2"),
excluding Phase 2's own work product.

It is verified by `python scripts/phase1_freeze_manifest.py`, and by check 1 of
`python scripts/phase2_integrity.py`. Rebuilding it with `--build` is **refused**
if any covered file has already drifted, so the manifest cannot be quietly
re-baselined onto a change.

## Where the count difference comes from

| | files |
|---|---|
| Tracked at `a69054f` | 165 |
| − Phase 2 work product (`phase2/`, `tests/phase2/`, `configs/phase2/`, `scripts/phase2_*`, `PHASE2_*`) | 32 |
| − `app/` dashboard | 29 |
| **= Phase 1 frozen set** | **104** |

Both exclusions are deliberate. The Phase 2 files were Phase 2's own work that
simply had not been separated into their own directories when `a69054f` was
made. The dashboard is a read-only viewer that Phase 2 extended with new panels
under explicit approval; **10 of its 29 files have legitimately changed** since
`a69054f`, so including it would make a "nothing changed" claim false for
reasons that have nothing to do with Phase 1's science.

The 104 files are all of Phase 1's model, training loop, evaluation metrics,
experiments, configs and tests, together with its **44 recorded result
artifacts**.

## What is and is not affected

**Not affected: the substantive claim.** No Phase 1 file has changed. Measured
against the stricter, provenance-defined set, **104 of 104 files are
byte-identical** to the Phase 1 completion commit, and `git diff a69054f HEAD`
over those paths is empty. The scientific guarantee that Phase 2 has not
disturbed Phase 1 holds, and now holds under a definition that can be re-checked
by anyone from the repository alone.

**Affected: the number, and the durability of the check.** "135 of 136" should
not be cited again; the correct figure is 104 of 104. No committed document ever
contained the old figure — it appeared only in session progress reports — so no
document required correction, and this erratum exists so the discrepancy is on
the record rather than silently replaced.
