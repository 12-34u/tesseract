#!/bin/bash
# Batch 02 driver: the T = 3 learnability probe (PHASE2_BATCH_02_PREREG.md).
#
# One probe rather than a chain of arms, but kept in the same shape as the
# Batch 01 driver so the launch procedure is the same one already used.
#
# Resumable by design. A probe whose report already exists is skipped, so an
# interrupted run picks up where it stopped instead of redoing completed work.
# A partially written directory is never overwritten: the Python runner sends
# the retry to a sibling __attemptN directory.
#
# Run it as a systemd user SERVICE, not a scope — a scope lives in the calling
# process tree and dies with the shell that started it:
#
#   systemd-run --user --unit=tesseract-batch02 --same-dir \
#       scripts/phase2_batch02_chain.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/parth/tesseract-cuda-venv/bin/python
LOG=runs/phase2/batch02_chain.log
mkdir -p runs/phase2

run () {   # $1 = label, $2 = report glob relative to runs/phase2/batch02, $3.. = command
  local label="$1" glob="$2"; shift 2
  if compgen -G "runs/phase2/batch02/$glob" > /dev/null; then
    echo "SKIP $label (report exists) $(date -u +%H:%M:%S)" >> "$LOG"
    return 0
  fi
  echo "START $label $(date -u +%H:%M:%S)" >> "$LOG"
  "$@" > "runs/phase2/batch02_$label.console.log" 2>&1
  echo "END $label rc=$? $(date -u +%H:%M:%S)" >> "$LOG"
}

run t3_curriculum "t3_curriculum*/t3_curriculum_report.json" \
    $PY -u scripts/phase2_batch02.py
echo "CHAIN COMPLETE $(date -u +%H:%M:%S)" >> "$LOG"
