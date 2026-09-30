/**
 * Derived state for the probes-and-batches section.
 *
 * The assertions that matter here are the ones protecting a research claim:
 * that an amended verdict never silently replaces the pre-registered one, that
 * the two unclassified Batch 01 entries stay visible so "four arms were
 * classified" can be checked, that the shortcut context survives the probe not
 * having reported, and that the K comparison is built from recorded curves
 * rather than invented when they are absent.
 */

import { describe, expect, it } from 'vitest';

import {
  CHANCE_PERCENT,
  LEARNABLE_BAR_PERCENT,
  SHORTCUT_CEILING_PERCENT,
  arm5Summary,
  armRows,
  batch02State,
  cellRows,
  diagnosticsState,
  kComparison,
  probeRows,
  recordRows,
  stageBoundaryRows,
} from './diagnostics.js';

const kRow = (k, overrides = {}) => ({
  k,
  classification: 'PARTIAL',
  classification_tone: 'warn',
  classification_amended: 'DEGENERATE',
  classification_amended_tone: 'fail',
  changed: true,
  seeds_above_chance: [2],
  seeds_above_chance_amended: [],
  seeds_meeting_learnable_bar: [],
  degenerate_seeds: [0, 1, 2],
  n_seeds: 3,
  per_seed: [],
  ...overrides,
});

const DIAGNOSTICS = {
  status: 'available',
  probes: {
    status: 'available',
    probes: [
      { id: 'probe_a', label: 'Probe A', t: 4, note: 'cold', status: 'available',
        warm_start: false, fixed_step_budget: 20000,
        by_k: [kRow(1, { classification: 'NOT_LEARNABLE', classification_amended: 'NOT_LEARNABLE', changed: false })] },
      { id: 'probe_b', label: 'Probe B', t: 8, note: 'warm', status: 'missing', message: 'not found.' },
    ],
    majority_class_percent_by_t: { 4: 1.9244, 8: 1.8354 },
  },
  batch01: {
    status: 'available',
    classified_arms: 4,
    arms_that_moved: 0,
    arm5_threshold: 2,
    arm5_approved: false,
    stage3_row: 'all arms NOT LEARNABLE',
    arms: [
      { id: 'arm1', label: 'Arm 1', lever: 'Fixed-set fit (gate)', question: 'q', status: 'available',
        is_gate: true, gate_passed: true, by_k: [] },
      { id: 'arm2a', label: 'Arm 2a', lever: 'Step-0 loss', question: 'q', status: 'available',
        is_gate: false, by_k: [] },
      { id: 'arm2b', label: 'Arm 2b', lever: 'Output-side LayerNorm', question: 'q', status: 'available',
        is_gate: false, moves_under_arm5_criteria: false, by_k: [kRow(1), kRow(8)] },
      { id: 'arm5', label: 'Arm 5', lever: 'Union of movers', question: 'q', status: 'missing',
        message: 'no report matching arm5*.' },
    ],
  },
  batch02: {
    status: 'missing',
    message: "Batch 02's T=3 probe has not reported.",
    t: 3,
    question: 'is T=3 learnable?',
    g1_passed: true,
    g2_passed: false,
    shortcut_candidate: 'window_product_forward',
    shortcut_condition: '(s[i+1]*s[i+2])^2 = e',
    shortcut_token_agreement: 0.2666,
    shortcut_agreement_iff_condition: true,
    preregistration: 'PHASE2_BATCH_02_PREREG.md',
    addendum: 'PHASE2_BATCH_02_ADDENDUM_01.md',
  },
  record: {
    status: 'available',
    documents: [
      { name: 'PHASE2_BATCH_02_PREREG.md', label: 'prereg', kind: 'prereg', present: true, sha256: 'a'.repeat(64) },
      { name: 'MISSING.md', label: 'gone', kind: 'erratum', present: false, sha256: null },
    ],
  },
};

const STATUS = { diagnostics: DIAGNOSTICS };

describe('bars', () => {
  it('uses the pre-registered bars, not rounded stand-ins', () => {
    expect(CHANCE_PERCENT).toBeCloseTo(1.6667, 4);
    expect(LEARNABLE_BAR_PERCENT).toBeCloseTo(6.6667, 4);
    expect(SHORTCUT_CEILING_PERCENT[3]).toBeCloseTo(26.6257, 4);
  });

  it('places the T=3 shortcut ceiling above the LEARNABLE bar', () => {
    // This inequality is the whole reason the attribution check exists.
    expect(SHORTCUT_CEILING_PERCENT[3]).toBeGreaterThan(LEARNABLE_BAR_PERCENT);
  });
});

describe('diagnosticsState', () => {
  it('reports available when the backend did', () => {
    expect(diagnosticsState(STATUS).available).toBe(true);
  });

  it('surfaces the backend wording when the block is pending', () => {
    const state = diagnosticsState({ diagnostics: { status: 'missing', message: 'nothing yet.' } });
    expect(state.available).toBe(false);
    expect(state.message).toBe('nothing yet.');
  });

  it('does not invent a payload when the block is absent', () => {
    expect(diagnosticsState({}).available).toBe(false);
    expect(diagnosticsState(undefined).available).toBe(false);
  });
});

describe('probeRows', () => {
  it('keeps both verdicts per K', () => {
    const [a] = probeRows(STATUS);
    expect(a.byK[0].classification).toBe('NOT_LEARNABLE');
    expect(a.byK[0].classification_amended).toBe('NOT_LEARNABLE');
  });

  it('marks a probe with no artifact as unavailable and carries its message', () => {
    const b = probeRows(STATUS).find((p) => p.id === 'probe_b');
    expect(b.available).toBe(false);
    expect(b.message).toBe('not found.');
  });

  it('returns nothing rather than throwing when probes did not load', () => {
    expect(probeRows({ diagnostics: {} })).toEqual([]);
  });
});

describe('armRows', () => {
  it('keeps the gate and the measurement visible, marked unclassified', () => {
    const rows = armRows(STATUS);
    const gate = rows.find((r) => r.id === 'arm1');
    const measurement = rows.find((r) => r.id === 'arm2a');
    expect(gate.classified).toBe(false);
    expect(gate.isGate).toBe(true);
    expect(gate.gatePassed).toBe(true);
    expect(measurement.classified).toBe(false);
    expect(measurement.isGate).toBe(false);
  });

  it('carries a classified arm both verdicts and whether it moved', () => {
    const arm = armRows(STATUS).find((r) => r.id === 'arm2b');
    expect(arm.classified).toBe(true);
    expect(arm.moved).toBe(false);
    expect(arm.byK.map((r) => [r.classification, r.classification_amended])).toEqual([
      ['PARTIAL', 'DEGENERATE'],
      ['PARTIAL', 'DEGENERATE'],
    ]);
  });

  it('keeps an arm that never ran, with its message', () => {
    const arm = armRows(STATUS).find((r) => r.id === 'arm5');
    expect(arm.available).toBe(false);
    expect(arm.message).toContain('no report');
  });
});

describe('arm5Summary', () => {
  it('reports the trigger as the criteria file defines it', () => {
    const summary = arm5Summary(STATUS);
    expect(summary.moved).toBe(0);
    expect(summary.threshold).toBe(2);
    expect(summary.approved).toBe(false);
    expect(summary.sentence).toContain('0 of 4');
    expect(summary.stage3Row).toBe('all arms NOT LEARNABLE');
  });

  it('approves only at or above the threshold', () => {
    const moved = { diagnostics: { ...DIAGNOSTICS, batch01: { ...DIAGNOSTICS.batch01, arms_that_moved: 2, arm5_approved: true } } };
    expect(arm5Summary(moved).approved).toBe(true);
  });
});

describe('batch02State', () => {
  it('keeps the shortcut context even though the probe has not reported', () => {
    // The G2 failure is a property of the task, measured before the probe ran,
    // so a reader must see it while the probe is still running.
    const state = batch02State(STATUS);
    expect(state.available).toBe(false);
    expect(state.context.g2Passed).toBe(false);
    expect(state.context.shortcutCandidate).toBe('window_product_forward');
    expect(state.context.shortcutCeiling).toBeCloseTo(26.6257, 4);
    expect(state.context.learnableBar).toBeCloseTo(6.6667, 4);
  });

  it('flags a smoke run so it cannot be read as a result', () => {
    const state = batch02State({
      diagnostics: { ...DIAGNOSTICS, batch02: { ...DIAGNOSTICS.batch02, status: 'available', smoke: true, by_k: [], cells: [] } },
    });
    expect(state.available).toBe(true);
    expect(state.smoke).toBe(true);
  });

  it('returns unavailable rather than throwing with no payload', () => {
    expect(batch02State({}).available).toBe(false);
  });
});

describe('stageBoundaryRows', () => {
  const cell = {
    stage_boundary_validation: [
      {
        step: 5000,
        stage_t: 1,
        validation_by_t: [
          { t: 1, token_accuracy: 100, loss: 0.0001, distinct_predicted_tokens: 60, majority_class_percent: 1.85 },
          { t: 3, token_accuracy: 1.64, loss: 22.7, distinct_predicted_tokens: 59,
            majority_class_percent: 1.81, shortcut_candidate: 'window_product_forward',
            shortcut_agreement_percent: 1.61 },
        ],
      },
    ],
  };

  it('flattens one row per boundary and depth', () => {
    const rows = stageBoundaryRows(cell);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ step: 5000, stageT: 1, t: 1, tokenAccuracy: 100, isStageDepth: true });
    expect(rows[1]).toMatchObject({ t: 3, isStageDepth: false, shortcutAgreement: 1.61 });
  });

  it('uses null, not zero, where a depth has no shortcut candidate', () => {
    expect(stageBoundaryRows(cell)[0].shortcutAgreement).toBeNull();
  });

  it('returns nothing for a cell with no recorded boundaries', () => {
    expect(stageBoundaryRows({})).toEqual([]);
    expect(stageBoundaryRows(undefined)).toEqual([]);
  });
});

describe('cellRows and kComparison', () => {
  const state = {
    cells: [
      // Field names are the backend's, which renames the report's
      // best_val_* cell fields; a fixture using the raw report names would
      // pass against code that reads neither.
      { k: 1, seed: 0, steps_run: 35000, distinct_predicted_tokens: 60, shortcut_agreement_percent: 18.11 },
      { k: 8, seed: 0, steps_run: 35000, distinct_predicted_tokens: 60, shortcut_agreement_percent: 20.0 },
    ],
    curves: {
      per_cell: [
        { k: 1, seed: 0, best_val_token_accuracy: 64.04, final_val_token_accuracy: 64.04,
          crossed_learnable_bar: { step: 16000 }, crossed_shortcut_ceiling: { step: 18500 },
          final_slope: { slope_points_per_1000_steps: 0.9 }, still_rising: true },
      ],
    },
  };

  it('joins each cell to its curve analysis by k and seed', () => {
    const rows = cellRows(state);
    expect(rows[0]).toMatchObject({
      k: 1, bestAccuracy: 64.04, crossedCeilingAt: 18500, crossedBarAt: 16000,
      stillRising: true, shortcutAgreement: 18.11, distinctTokens: 60,
    });
  });

  it('leaves curve fields null for a cell with no curve row, rather than guessing', () => {
    const row = cellRows(state).find((r) => r.k === 8);
    expect(row.bestAccuracy).toBeNull();
    expect(row.crossedCeilingAt).toBeNull();
    expect(row.finalSlope).toBeNull();
  });

  it('aggregates per K and ignores missing values in the means', () => {
    const groups = kComparison(state);
    expect(groups.map((g) => g.k)).toEqual([1, 8]);
    expect(groups[0].meanBestAccuracy).toBeCloseTo(64.04, 2);
    expect(groups[0].stillRising).toBe(1);
    expect(groups[0].meanShortcutAgreement).toBeCloseTo(18.11, 2);
    expect(groups[1].meanBestAccuracy).toBeNull();
    expect(groups[1].meanShortcutAgreement).toBeCloseTo(20.0, 2);
  });

  it('produces nothing without cells', () => {
    expect(kComparison({})).toEqual([]);
    expect(cellRows(undefined)).toEqual([]);
  });
});

describe('recordRows', () => {
  it('shortens the checksum for display but keeps the full one', () => {
    const [first] = recordRows(STATUS);
    expect(first.shortSha).toHaveLength(12);
    expect(first.sha256).toHaveLength(64);
  });

  it('keeps a missing document in the list, marked absent', () => {
    const missing = recordRows(STATUS).find((d) => d.name === 'MISSING.md');
    expect(missing.present).toBe(false);
    expect(missing.shortSha).toBeNull();
  });
});
