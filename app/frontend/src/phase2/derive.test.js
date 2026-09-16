import { describe, expect, it } from 'vitest';
import {
  badgeClass,
  d5Summary,
  gateBadgeClass,
  isAvailable,
  isError,
  isPending,
  p1bMatrix,
  pendingMessage,
  readiness,
  roleLabel,
  roleRows,
  stageBadgeClass,
} from './derive';

// The payloads below mirror the shapes produced by app/backend/services/phase2_loader.py.
// None of them carries a real experimental value: the learnability numbers in the
// P1b fixture are synthetic and exist only to exercise the mapping.

const ROLES = { status: 'available', anchor: [1], diagnostic: [2], primary_depth: [4, 8] };

describe('block states', () => {
  it('separates available, pending and error', () => {
    expect(isAvailable({ status: 'available' })).toBe(true);
    expect(isPending({ status: 'missing' })).toBe(true);
    expect(isPending({ status: 'pending' })).toBe(true);
    expect(isError({ status: 'error' })).toBe(true);
    expect(isAvailable({ status: 'missing' })).toBe(false);
    expect(isAvailable(undefined)).toBe(false);
  });

  it('always yields a message for a block with no data', () => {
    expect(pendingMessage({ status: 'missing', message: 'P1b results not available yet.' }))
      .toBe('P1b results not available yet.');
    expect(pendingMessage({ status: 'missing' }, 'fallback')).toBe('fallback');
    expect(pendingMessage(undefined)).toBeTruthy();
  });
});

describe('T roles', () => {
  it('labels each T from the Amendment 01 partition', () => {
    expect(roleLabel(1, ROLES)).toBe('Anchor');
    expect(roleLabel(2, ROLES)).toBe('Diagnostic');
    expect(roleLabel(4, ROLES)).toBe('Primary inference');
    expect(roleLabel(8, ROLES)).toBe('Primary inference');
    expect(roleLabel(3, ROLES)).toBeNull();
  });

  it('builds ordered rows and nothing at all without the payload', () => {
    expect(roleRows(ROLES).map((r) => [r.t, r.role])).toEqual([
      [1, 'Anchor'],
      [2, 'Diagnostic'],
      [4, 'Primary inference'],
      [8, 'Primary inference'],
    ]);
    expect(roleRows({ status: 'missing' })).toEqual([]);
  });
});

describe('gate badges', () => {
  it('never turns a raw G2 failure into a pass', () => {
    const diagnostic = { g2_applicable: true, g2_passed: false, amendment_01_diagnostic: true };
    // Amber rather than red, because the failure is documented — but not green.
    expect(gateBadgeClass(diagnostic)).toBe('badge-warn');
    expect(gateBadgeClass(diagnostic)).not.toBe('badge-pass');
  });

  it('marks an undocumented failure as a hard failure', () => {
    expect(gateBadgeClass({ g2_applicable: true, g2_passed: false, amendment_01_diagnostic: false }))
      .toBe('badge-fail');
  });

  it('passes and not-applicable are distinct', () => {
    expect(gateBadgeClass({ g2_applicable: true, g2_passed: true })).toBe('badge-pass');
    expect(gateBadgeClass({ g2_applicable: false })).toBe('badge-warn');
  });
});

describe('readiness', () => {
  it('is BLOCKED while any pre-registration step is outstanding', () => {
    const plan = {
      status: 'available',
      blocked_reasons: ['model width is PENDING_P1B', 'P1b is PENDING'],
      total_planned_runs: 138,
      total_completed_runs: 0,
    };
    const state = readiness(plan);
    expect(state.state).toBe('BLOCKED');
    expect(state.reasons).toHaveLength(2);
  });

  it('is READY only when nothing blocks and nothing has run', () => {
    expect(readiness({ status: 'available', blocked_reasons: [], total_planned_runs: 138, total_completed_runs: 0 }).state)
      .toBe('READY');
  });

  it('tracks RUNNING and COMPLETE from the counts', () => {
    const base = { status: 'available', blocked_reasons: [], total_planned_runs: 138 };
    expect(readiness({ ...base, total_completed_runs: 12 }).state).toBe('RUNNING');
    expect(readiness({ ...base, total_completed_runs: 138 }).state).toBe('COMPLETE');
  });

  it('is UNKNOWN rather than READY when the plan is missing', () => {
    expect(readiness({ status: 'missing' }).state).toBe('UNKNOWN');
  });
});

describe('stage and tone badges', () => {
  it('treats BLOCKED as a warning, not a failure', () => {
    expect(stageBadgeClass('BLOCKED')).toBe('badge-warn');
    expect(stageBadgeClass('COMPLETE')).toBe('badge-pass');
    expect(stageBadgeClass('READY')).toBe('badge-pass');
  });

  it('maps backend tones', () => {
    expect(badgeClass('ok')).toBe('badge-pass');
    expect(badgeClass('fail')).toBe('badge-fail');
    expect(badgeClass('warn')).toBe('badge-warn');
    expect(badgeClass(undefined)).toBe('badge-warn');
  });
});

describe('P1b matrix', () => {
  const p1b = {
    status: 'available',
    cells: [
      { width: 'small', k: 1, final_val_chance_normalised: 0.01, at_floor_final: true },
      { width: 'small', k: 8, final_val_chance_normalised: 0.5, at_floor_final: false },
      { width: 'medium', k: 1, final_val_chance_normalised: 0.4, at_floor_final: false },
      { width: 'medium', k: 8, final_val_chance_normalised: 0.6, at_floor_final: false },
    ],
  };

  it('groups the four cells by width and K', () => {
    const { widths, kValues, cell } = p1bMatrix(p1b);
    expect(widths).toEqual(['small', 'medium']);
    expect(kValues).toEqual([1, 8]);
    expect(cell('small', 1).at_floor_final).toBe(true);
    expect(cell('medium', 8).final_val_chance_normalised).toBe(0.6);
    expect(cell('small', 4)).toBeNull();
  });

  it('is empty, not fabricated, when P1b has not run', () => {
    const { widths, kValues, cell } = p1bMatrix({ status: 'missing' });
    expect(widths).toEqual([]);
    expect(kValues).toEqual([]);
    expect(cell('small', 1)).toBeNull();
  });
});

describe('D5 summary', () => {
  it('is null when P1b has produced no decision', () => {
    expect(d5Summary({ status: 'missing' })).toBeNull();
    expect(d5Summary({ status: 'available', d5: { status: 'error', decision: null } })).toBeNull();
  });

  it('surfaces the width and budget outcome including the not-converged flag', () => {
    const summary = d5Summary({
      status: 'available',
      d5: {
        decision: {
          width_decision: { outcome: 'SELECT', selected_width: 'small', k_low: 1, k_low_at_floor: true },
          budget_decision: { outcome: 'SELECT', recommended_max_steps: 20000, not_converged: true, k_budgets_agree: false },
        },
      },
    });
    expect(summary.selectedWidth).toBe('small');
    expect(summary.kLowAtFloor).toBe(true);
    expect(summary.recommendedMaxSteps).toBe(20000);
    expect(summary.notConverged).toBe(true);
    expect(summary.budgetsAgree).toBe(false);
  });

  it('reports DO_NOT_PROCEED without a width or budget', () => {
    const summary = d5Summary({
      status: 'available',
      d5: {
        decision: {
          width_decision: { outcome: 'DO_NOT_PROCEED', selected_width: null },
          budget_decision: { outcome: 'DO_NOT_PROCEED', recommended_max_steps: null },
        },
      },
    });
    expect(summary.widthOutcome).toBe('DO_NOT_PROCEED');
    expect(summary.selectedWidth).toBeNull();
    expect(summary.recommendedMaxSteps).toBeNull();
  });
});

describe('no hardcoded experimental values', () => {
  it('the derivation module contains no accuracy, loss or step-count literals', async () => {
    const fs = await import('node:fs');
    const source = fs.readFileSync(new URL('./derive.js', import.meta.url), 'utf8');
    // Any bare decimal or large integer here would be an experimental value
    // baked into the UI rather than read from an artifact.
    expect(source).not.toMatch(/\b0\.\d+\b/);
    expect(source).not.toMatch(/\b\d{4,}\b/);
  });
});
