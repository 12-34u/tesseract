#!/bin/bash
# Batch 01 driver: runs the arms in the pre-registered order.
#
# Resumable by design. An arm whose report already exists is skipped, so an
# interrupted chain picks up where it stopped instead of redoing completed
# work. A partially written arm directory is never overwritten: the Python
# runner sends the retry to a sibling __attemptN directory.
#
# Run it as a systemd user SERVICE, not a scope — a scope lives in the calling
# process tree and dies with the shell that started it:
#
#   systemd-run --user --unit=tesseract-batch01 --same-dir \
#       scripts/phase2_batch01_chain.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/parth/tesseract-cuda-venv/bin/python
LOG=runs/phase2/batch01_chain.log

run () {   # $1 = arm flag, $2 = report glob relative to runs/phase2/batch01
  if compgen -G "runs/phase2/batch01/$2" > /dev/null; then
    echo "SKIP arm $1 (report exists) $(date -u +%H:%M:%S)" >> "$LOG"
    return 0
  fi
  echo "START arm $1 $(date -u +%H:%M:%S)" >> "$LOG"
  $PY -u scripts/phase2_batch01.py --arm "$1" > "runs/phase2/batch01_arm$1.console.log" 2>&1
  echo "END arm $1 rc=$? $(date -u +%H:%M:%S)" >> "$LOG"
}

run 1  "arm1_fixed_set*/arm1_report.json"
run 2b "arm2b_residual_fix*/arm2b_residual_fix_report.json"
run 3  "arm3_curriculum*/arm3_curriculum_report.json"
run 4  "arm4_warmup*/arm4_warmup_report.json"
run 2c "arm2c_per_step_norm_train*/arm2c_per_step_norm_train_report.json"
echo "CHAIN COMPLETE $(date -u +%H:%M:%S)" >> "$LOG"
