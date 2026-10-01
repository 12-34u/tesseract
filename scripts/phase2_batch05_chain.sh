#!/bin/bash
# Batch 05 driver: the T = 3 learnability probe (PHASE2_BATCH_05_PREREG.md).
#
# One sweep rather than a chain of arms, but kept in the same shape as the
# Batch 01 and 02 drivers so the launch procedure is the one already used.
#
# The Python runner refuses to train if the bounded-growth gate fails, so an
# unbounded K cannot silently produce a null result here.
#
# Arguments are passed through to the Python runner, so a resume after an
# interruption is:
#
#   systemd-run --user --unit=tesseract-batch05 --same-dir \
#       scripts/phase2_batch05_chain.sh --resume
#
# The console log is appended to, not truncated, because a resume salvages the
# curves and stage boundaries of already-finished cells out of it.
#
# Resumable by design. A probe whose report already exists is skipped, so an
# interrupted run picks up where it stopped instead of redoing completed work.
# A partially written directory is never overwritten: the Python runner sends
# the retry to a sibling __attemptN directory.
#
# Run it as a systemd user SERVICE, not a scope — a scope lives in the calling
# process tree and dies with the shell that started it:
#
#   systemd-run --user --unit=tesseract-batch05 --same-dir \
#       scripts/phase2_batch05_chain.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/parth/tesseract-cuda-venv/bin/python
LOG=runs/phase2/batch05_chain.log
mkdir -p runs/phase2

run () {   # $1 = label, $2 = report glob relative to runs/phase2/batch05, $3.. = command
  local label="$1" glob="$2"; shift 2
  if compgen -G "runs/phase2/batch05/$glob" > /dev/null; then
    echo "SKIP $label (report exists) $(date -u +%H:%M:%S)" >> "$LOG"
    return 0
  fi
  echo "START $label $(date -u +%H:%M:%S)" >> "$LOG"
  "$@" >> "runs/phase2/batch05_$label.console.log" 2>&1
  echo "END $label rc=$? $(date -u +%H:%M:%S)" >> "$LOG"
}

run t3_family "t3_family*/t3_family_report.json" \
    $PY -u scripts/phase2_batch05.py "$@"
echo "CHAIN COMPLETE $(date -u +%H:%M:%S)" >> "$LOG"
