import { describe, expect, it } from 'vitest';
import {
  badgeClass,
  capacityGrid,
  d5CriteriaRows,
  hasCellMetrics,
  widthSelection,
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

describe('model family selector', () => {
  // Shapes mirror /api/phase2/status.model_families. Parameter counts are
  // architecture sizes computed by the backend from model configs, not results.
  const families = {
    status: 'available',
    selection_pending: true,
    selected_base: null,
    vocab_size: 60,
    families: [
      { name: 'small', base: 'prototype_small', label: 'Small Prototype (222K)', d_model: 128, num_heads: 4, head_dim: 32, d_ff: 512, parameter_count: 222140, selected: false },
      { name: 'medium', base: 'phase2/prototype_medium', label: 'Medium Prototype (837K)', d_model: 256, num_heads: 8, head_dim: 32, d_ff: 1024, parameter_count: 837436, selected: false },
      { name: 'large', base: 'phase2/prototype_large', label: 'Large Research Model (~7M)', d_model: 768, num_heads: 24, head_dim: 32, d_ff: 3072, parameter_count: 7230780, selected: false },
    ],
  };

  it('lists all three widths with the required labels', () => {
    expect(isAvailable(families)).toBe(true);
    expect(families.families.map((f) => f.label)).toEqual([
      'Small Prototype (222K)',
      'Medium Prototype (837K)',
      'Large Research Model (~7M)',
    ]);
  });

  it('marks nothing as selected while the width is pending', () => {
    expect(families.selection_pending).toBe(true);
    expect(families.selected_base).toBeNull();
    expect(families.families.some((f) => f.selected)).toBe(false);
  });

  it('keeps head dimension constant across the family', () => {
    expect(families.families.map((f) => f.head_dim)).toEqual([32, 32, 32]);
    expect(families.families.map((f) => f.d_ff / f.d_model)).toEqual([4, 4, 4]);
  });

  it('renders as pending, not empty, when the backend has no family data', () => {
    const missing = { status: 'missing', message: 'configs not found' };
    expect(isAvailable(missing)).toBe(false);
    expect(pendingMessage(missing)).toBe('configs not found');
  });
});

// ---------------------------------------------------------------------------
// Capacity diagnostics: P1b / P1b-2 / combined D5
// ---------------------------------------------------------------------------

const PENDING_CAPACITY = {
  status: 'available',
  note: 'Large is an eligible candidate, not a selection. D5 recommends the smallest learnable width.',
  p1b: { label: 'P1b — Original Capacity Diagnostic', widths: ['small', 'medium'], status: 'missing',
         state: 'PENDING', tone: 'warn', message: 'P1b results not available yet.', cells: [] },
  p1b2: { label: 'P1b-2 — Large Capacity Diagnostic', widths: ['large'], status: 'missing',
          state: 'NOT STARTED', tone: 'warn',
          message: 'P1b-2 has not started. Large is a candidate width awaiting its capacity diagnostic.',
          cells: [], provenance_note: 'P1b-2 was introduced after the Large model family was added…' },
  combined: { label: 'Combined D5 — Width Selection', status: 'missing', state: 'WAITING FOR P1b + P1b-2',
              tone: 'warn', message: 'Width selection waiting for P1b + P1b-2.',
              awaiting: ['P1b', 'P1b-2'], cells: [], d5: null },
  candidate_widths: [
    { name: 'small', label: 'Small Prototype (222K)', status: 'Candidate', diagnostic: 'P1b' },
    { name: 'medium', label: 'Medium Prototype (837K)', status: 'Candidate', diagnostic: 'P1b' },
    { name: 'large', label: 'Large Research Model (~7M)', status: 'Candidate — P1b-2 pending', diagnostic: 'P1b-2' },
  ],
};

describe('capacity diagnostics — pending state', () => {
  it('renders all six rows with no artifacts, none of them complete', () => {
    const { rows } = capacityGrid(PENDING_CAPACITY);
    expect(rows.map((r) => `${r.width}/K=${r.k}`)).toEqual([
      'small/K=1', 'small/K=8', 'medium/K=1', 'medium/K=8', 'large/K=1', 'large/K=8',
    ]);
    expect(rows.every((r) => r.state === 'pending')).toBe(true);
    expect(rows.every((r) => r.cell === null)).toBe(true);
    expect(rows.every((r) => !hasCellMetrics(r.cell))).toBe(true);
  });

  it('places each width in the column of the diagnostic that owns it', () => {
    const { rows } = capacityGrid(PENDING_CAPACITY);
    const owner = Object.fromEntries(rows.map((r) => [r.width, r.owner]));
    expect(owner).toEqual({ small: 'p1b', medium: 'p1b', large: 'p1b2' });
  });

  it('P1b and P1b-2 sections carry their own distinct states', () => {
    expect(PENDING_CAPACITY.p1b.state).toBe('PENDING');
    expect(PENDING_CAPACITY.p1b2.state).toBe('NOT STARTED');
  });

  it('shows Large as a candidate, never as selected', () => {
    const large = PENDING_CAPACITY.candidate_widths.find((w) => w.name === 'large');
    expect(large.status).toBe('Candidate — P1b-2 pending');
    expect(PENDING_CAPACITY.candidate_widths.some((w) => /selected/i.test(w.status))).toBe(false);
  });

  it('width and budget stay pending while D5 has no artifact', () => {
    const s = widthSelection(PENDING_CAPACITY);
    expect(s.state).toBe('WAITING FOR P1b + P1b-2');
    expect(s.selectedWidth).toBeNull();
    expect(s.recommendedMaxSteps).toBeNull();
    expect(s.awaiting).toEqual(['P1b', 'P1b-2']);
    expect(d5CriteriaRows(PENDING_CAPACITY)).toEqual([]);
  });

  it('is still pending when only P1b has landed', () => {
    const halfway = { ...PENDING_CAPACITY,
      p1b: { ...PENDING_CAPACITY.p1b, status: 'available', state: 'COMPLETE' },
      combined: { ...PENDING_CAPACITY.combined, awaiting: ['P1b-2'] } };
    expect(widthSelection(halfway).selectedWidth).toBeNull();
    expect(widthSelection(halfway).awaiting).toEqual(['P1b-2']);
  });
});

describe('capacity diagnostics — completed state from fixture artifacts', () => {
  const completeCell = (width, k, score, loss) => ({
    width, k, source: width === 'large' ? 'P1b-2' : 'P1b',
    final_val_chance_normalised: score, final_val_loss: loss,
    final_val_token_accuracy: 50, final_val_exact_match: 1,
    best_step: 20000, steps_run: 20000, at_floor_final: score < 0.02,
  });

  const COMPLETE = {
    status: 'available',
    note: PENDING_CAPACITY.note,
    p1b: { ...PENDING_CAPACITY.p1b, status: 'available', state: 'COMPLETE',
           cells: [completeCell('small', 1, 0.001, 4.6), completeCell('small', 8, 0.001, 4.6),
                   completeCell('medium', 1, 0.4, 3.6), completeCell('medium', 8, 0.5, 3.6)] },
    p1b2: { ...PENDING_CAPACITY.p1b2, status: 'available', state: 'COMPLETE',
            cells: [completeCell('large', 1, 0.6, 3.5), completeCell('large', 8, 0.7, 3.5)] },
    combined: {
      label: 'Combined D5 — Width Selection', status: 'available', state: 'COMBINED', tone: 'ok',
      awaiting: [], sources: [{ label: 'P1b', widths: ['small', 'medium'] }, { label: 'P1b-2', widths: ['large'] }],
      single_seed_note: 'one seed per cell',
      d5: {
        status: 'available', evaluated_once_on_combined_report: true,
        decision: {
          width_decision: { outcome: 'SELECT', selected_width: 'medium', k_low: 1, k_low_at_floor: false },
          budget_decision: { outcome: 'SELECT', recommended_max_steps: 15000, not_converged: false },
          widths: [
            { width: 'small', parameter_count: 222140, learnable: false,
              per_k: { 1: { criteria: { a: false, b: false, c: false }, learnable: false },
                       8: { criteria: { a: false, b: false, c: false }, learnable: false } } },
            { width: 'medium', parameter_count: 837436, learnable: true,
              per_k: { 1: { criteria: { a: true, b: true, c: true }, learnable: true },
                       8: { criteria: { a: true, b: true, c: true }, learnable: true } } },
            { width: 'large', parameter_count: 7230780, learnable: true,
              per_k: { 1: { criteria: { a: true, b: true, c: true }, learnable: true },
                       8: { criteria: { a: true, b: true, c: true }, learnable: true } } },
          ],
        },
      },
    },
    candidate_widths: PENDING_CAPACITY.candidate_widths,
  };

  it('marks completed cells complete and pending cells pending', () => {
    const { rows } = capacityGrid(COMPLETE);
    expect(rows).toHaveLength(6);
    expect(rows.every((r) => hasCellMetrics(r.cell))).toBe(true);
  });

  it('reads the width and budget from the D5 artifact', () => {
    const s = widthSelection(COMPLETE);
    expect(s.selectedWidth).toBe('medium');
    expect(s.recommendedMaxSteps).toBe(15000);
    expect(s.notConverged).toBe(false);
    expect(s.kLowAtFloor).toBe(false);
    expect(s.evaluatedOnce).toBe(true);
  });

  it('does not select Large even though Large is learnable', () => {
    expect(widthSelection(COMPLETE).selectedWidth).not.toBe('large');
  });

  it('exposes per-width, per-criterion rows straight from the decision', () => {
    const rows = d5CriteriaRows(COMPLETE);
    expect(rows).toHaveLength(6);
    expect(rows.map((r) => `${r.width}/${r.k}`)).toEqual([
      'small/1', 'small/8', 'medium/1', 'medium/8', 'large/1', 'large/8',
    ]);
    expect(rows.filter((r) => r.learnable)).toHaveLength(4);
  });

  it('reports no override until one is recorded', () => {
    expect(widthSelection(COMPLETE).override).toBeNull();
  });

  it('surfaces an override when the artifact records one', () => {
    const withOverride = { ...COMPLETE,
      combined: { ...COMPLETE.combined, override: { approved_width: 'large', reason: 'documented' } } };
    expect(widthSelection(withOverride).override.approved_width).toBe('large');
    expect(widthSelection(withOverride).recommendedWidth).toBe('medium');
  });
});

describe('capacity helpers fabricate nothing', () => {
  it('returns empty structures rather than defaults for absent input', () => {
    expect(capacityGrid(undefined).rows).toEqual([]);
    expect(hasCellMetrics(null)).toBe(false);
    expect(hasCellMetrics({})).toBe(false);
    expect(d5CriteriaRows(undefined)).toEqual([]);
    expect(widthSelection(undefined).selectedWidth).toBeNull();
  });
});
