/**
 * Derived state for the probes-and-batches section.
 *
 * What actually ran in Phase 2 so far is not the grid: it is two P1c probes,
 * Batch 01's four diagnostic arms plus two measurements, and Batch 02's T=3
 * probe. This module turns the backend's `diagnostics` payload into rows, and
 * decides nothing the backend already decided — classifications, tones and the
 * Arm 5 verdict all arrive computed, because they come from pre-registered
 * rules that the research code owns.
 *
 * Two things it does enforce, because they are presentation rules rather than
 * research rules:
 *
 * - An accuracy at T=2 or T=3 is never returned without the shortcut ceiling
 *   that belongs beside it. G2 fails at both depths, so a bare accuracy there
 *   invites exactly the misreading the pre-registration exists to prevent.
 * - Original and amended classifications always travel together. Addendum 03
 *   is explicit that neither replaces the other.
 */

import { isAvailable, isPending, pendingMessage } from './derive.js';

/** Token accuracy a shallow shortcut reaches, by depth. Recorded in the prereg. */
export const SHORTCUT_CEILING_PERCENT = { 2: 26.6, 3: 26.6257 };

/** The pre-registered LEARNABLE bar: chance (100/60) + 5 points. */
export const LEARNABLE_BAR_PERCENT = 100 / 60 + 5;

/** Uniform chance over the 60-element vocabulary. */
export const CHANCE_PERCENT = 100 / 60;

/** Whether the diagnostics block loaded at all. */
export function diagnosticsState(status) {
  const block = status?.diagnostics;
  if (isAvailable(block)) return { available: true, block };
  return {
    available: false,
    message: isPending(block)
      ? pendingMessage(block, 'No diagnostics payload yet.')
      : block?.message || 'Diagnostics did not load.',
  };
}

/** Probes A and B, one row each, with both classifications per K. */
export function probeRows(status) {
  const block = status?.diagnostics?.probes;
  if (!isAvailable(block)) return [];
  return (block.probes || []).map((probe) => ({
    id: probe.id,
    label: probe.label,
    t: probe.t,
    note: probe.note,
    available: probe.status === 'available',
    message: probe.message,
    warmStart: probe.warm_start,
    steps: probe.fixed_step_budget,
    byK: probe.by_k || [],
  }));
}

/**
 * Batch 01's arms.
 *
 * Arm 1 is a gate and Arm 2a is a measurement, so neither carries a
 * classification. They are kept in the list with `classified: false` rather
 * than dropped, because "four arms were classified" is only checkable if the
 * two that were not are visible.
 */
export function armRows(status) {
  const block = status?.diagnostics?.batch01;
  if (!isAvailable(block)) return [];
  return (block.arms || []).map((arm) => ({
    id: arm.id,
    label: arm.label,
    lever: arm.lever,
    question: arm.question,
    available: arm.status === 'available',
    message: arm.message,
    isGate: Boolean(arm.is_gate),
    gatePassed: arm.gate_passed,
    classified: Boolean((arm.by_k || []).length),
    moved: arm.moves_under_arm5_criteria,
    byK: arm.by_k || [],
  }));
}

/** The Arm 5 trigger, as the criteria file defines it. */
export function arm5Summary(status) {
  const block = status?.diagnostics?.batch01;
  if (!isAvailable(block)) return null;
  const moved = block.arms_that_moved;
  const threshold = block.arm5_threshold;
  return {
    moved,
    threshold,
    classified: block.classified_arms,
    approved: Boolean(block.arm5_approved),
    tone: block.arm5_approved ? 'ok' : 'idle',
    stage3Row: block.stage3_row,
    sentence:
      moved == null
        ? null
        : `${moved} of ${block.classified_arms} classified arms moved; Arm 5 needs ${threshold}.`,
  };
}

/**
 * Batch 02's T=3 probe.
 *
 * Returns the shortcut context whether or not the probe has reported, because
 * the G2 failure at T=3 is a property of the task that was measured before the
 * probe ran. A reader looking at this section while the probe is still running
 * should still see why a T=3 accuracy will need an attribution.
 */
export function batch02State(status) {
  const block = status?.diagnostics?.batch02;
  if (!block) return { available: false, message: 'No Batch 02 payload.' };
  const context = {
    t: block.t,
    question: block.question,
    g1Passed: block.g1_passed,
    g2Passed: block.g2_passed,
    shortcutCandidate: block.shortcut_candidate,
    shortcutCondition: block.shortcut_condition,
    shortcutAgreement: block.shortcut_token_agreement,
    shortcutExact: block.shortcut_agreement_iff_condition,
    shortcutCeiling: SHORTCUT_CEILING_PERCENT[block.t] ?? null,
    learnableBar: LEARNABLE_BAR_PERCENT,
    preregistration: block.preregistration,
    addendum: block.addendum,
  };
  if (block.status !== 'available') {
    return { available: false, message: block.message, context, smoke: false };
  }
  return {
    available: true,
    context,
    smoke: Boolean(block.smoke),
    reportPath: block.report_path,
    majorityClass: block.majority_class_percent,
    byK: block.by_k || [],
    cells: block.cells || [],
    curves: block.curve_analysis || null,
  };
}

/**
 * One cell's stage-boundary measurements, flattened for a table.
 *
 * This is the field that fixes Batch 01's recorded deviation, so it is shown
 * as recorded data rather than summarised away: one row per (boundary, depth),
 * carrying the shortcut agreement wherever the depth has a candidate.
 */
export function stageBoundaryRows(cell) {
  const rows = [];
  for (const boundary of cell?.stage_boundary_validation || []) {
    for (const entry of boundary.validation_by_t || []) {
      rows.push({
        step: boundary.step,
        stageT: boundary.stage_t,
        t: entry.t,
        tokenAccuracy: entry.token_accuracy,
        loss: entry.loss,
        distinctTokens: entry.distinct_predicted_tokens,
        majorityClass: entry.majority_class_percent,
        shortcutCandidate: entry.shortcut_candidate ?? null,
        shortcutAgreement: entry.shortcut_agreement_percent ?? null,
        // The depth being trained at this boundary, which is the depth whose
        // accuracy is expected to be high; the others show retention.
        isStageDepth: entry.t === boundary.stage_t,
      });
    }
  }
  return rows;
}

/** Per-cell rows for the K comparison, joined to the curve analysis when present. */
export function cellRows(state) {
  const byCell = new Map();
  for (const row of state?.curves?.per_cell || []) byCell.set(`${row.k}:${row.seed}`, row);
  return (state?.cells || []).map((cell) => {
    const curve = byCell.get(`${cell.k}:${cell.seed}`) || {};
    return {
      k: cell.k,
      seed: cell.seed,
      stepsRun: cell.steps_run,
      distinctTokens: cell.distinct_predicted_tokens,
      shortcutAgreement: cell.shortcut_agreement_percent,
      bestAccuracy: curve.best_val_token_accuracy ?? null,
      finalAccuracy: curve.final_val_token_accuracy ?? null,
      crossedBarAt: curve.crossed_learnable_bar?.step ?? null,
      crossedCeilingAt: curve.crossed_shortcut_ceiling?.step ?? null,
      finalSlope: curve.final_slope?.slope_points_per_1000_steps ?? null,
      stillRising: curve.still_rising ?? null,
    };
  });
}

/**
 * The K comparison, aggregated per K.
 *
 * Deliberately labelled descriptive. A curriculum probe is not a grid cell,
 * six cells cannot separate a K effect from seed variance, and the classification
 * is what governs — so this returns the numbers and a caveat together rather
 * than letting the table imply more than it can support.
 */
export function kComparison(state) {
  const groups = new Map();
  for (const row of cellRows(state)) {
    if (!groups.has(row.k)) groups.set(row.k, []);
    groups.get(row.k).push(row);
  }
  const mean = (values) => {
    const usable = values.filter((v) => Number.isFinite(v));
    return usable.length ? usable.reduce((a, b) => a + b, 0) / usable.length : null;
  };
  return [...groups.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([k, cells]) => ({
      k,
      cells: cells.length,
      meanBestAccuracy: mean(cells.map((c) => c.bestAccuracy)),
      meanShortcutAgreement: mean(cells.map((c) => c.shortcutAgreement)),
      meanTakeoffStep: mean(cells.map((c) => c.crossedCeilingAt)),
      meanFinalSlope: mean(cells.map((c) => c.finalSlope)),
      stillRising: cells.filter((c) => c.stillRising === true).length,
      perSeed: cells.sort((a, b) => a.seed - b.seed),
    }));
}

/** The pre-registration and correction record. */
export function recordRows(status) {
  const block = status?.diagnostics?.record;
  if (!isAvailable(block)) return [];
  return (block.documents || []).map((doc) => ({
    name: doc.name,
    label: doc.label,
    kind: doc.kind,
    present: doc.present,
    sha256: doc.sha256,
    shortSha: doc.sha256 ? doc.sha256.slice(0, 12) : null,
  }));
}
