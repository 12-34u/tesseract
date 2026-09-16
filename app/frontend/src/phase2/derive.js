/**
 * Pure helpers for the Phase 2 status view.
 *
 * Every function here maps a backend payload onto display state. None of them
 * supplies an experimental value: when the backend reports a block as missing
 * or pending, these return the pending state, never a substitute number. They
 * are separated from the components so the mapping can be unit tested without
 * a DOM.
 */

export const PENDING_STATES = new Set(['missing', 'pending']);

/** A block is renderable only when the backend says it is available. */
export const isAvailable = (block) => block?.status === 'available';

/** A block the backend could not parse. Rendered as an error, never as data. */
export const isError = (block) => block?.status === 'error';

/**
 * Pending means "no artifact yet", which is a legitimate state to display.
 * It is deliberately distinct from an error.
 */
export const isPending = (block) => PENDING_STATES.has(block?.status);

/**
 * The message to show for a block that has no data. The backend supplies the
 * wording; this only guarantees something is shown rather than an empty panel.
 */
export const pendingMessage = (block, fallback = 'Not available yet.') =>
  block?.message || fallback;

/** Map a backend tone onto one of the dashboard's badge classes. */
export function badgeClass(tone) {
  if (tone === 'ok') return 'badge-pass';
  if (tone === 'fail') return 'badge-fail';
  return 'badge-warn';
}

/** Map a plan stage state onto a badge class. BLOCKED is a warning, not a failure. */
export function stageBadgeClass(state) {
  if (state === 'COMPLETE') return 'badge-pass';
  if (state === 'RUNNING') return 'badge-pass';
  if (state === 'READY') return 'badge-pass';
  return 'badge-warn';
}

/**
 * Badge class for a raw gate result.
 *
 * A G2 failure at a diagnostic T is still a FAIL. Amendment 01 annotates it; it
 * is never recoloured to look like a pass. The only softening is that the badge
 * for a documented diagnostic failure is amber rather than red, and the label
 * still reads FAIL.
 */
export function gateBadgeClass(row) {
  if (row?.g2_applicable === false) return 'badge-warn';
  if (row?.g2_passed === true) return 'badge-pass';
  if (row?.amendment_01_diagnostic) return 'badge-warn';
  return 'badge-fail';
}

/** Human label for a T value given the Amendment 01 role partition. */
export function roleLabel(t, roles) {
  if (!isAvailable(roles)) return null;
  if ((roles.anchor || []).includes(t)) return 'Anchor';
  if ((roles.diagnostic || []).includes(t)) return 'Diagnostic';
  if ((roles.primary_depth || []).includes(t)) return 'Primary inference';
  return null;
}

/** Rows for the T-role table, derived from the roles payload alone. */
export function roleRows(roles) {
  if (!isAvailable(roles)) return [];
  const rows = [];
  (roles.anchor || []).forEach((t) => rows.push({ t, role: 'Anchor', note: 'Reported; not a depth condition' }));
  (roles.diagnostic || []).forEach((t) =>
    rows.push({ t, role: 'Diagnostic', note: 'Reported with the shortcut baseline; excluded from primary inference' }),
  );
  (roles.primary_depth || []).forEach((t) =>
    rows.push({ t, role: 'Primary inference', note: 'Carries the depth claim' }),
  );
  return rows.sort((a, b) => a.t - b.t);
}

/**
 * Overall readiness of the Phase 2 experiment.
 *
 * Derived only from the blockers the backend reports, so it cannot claim the
 * experiment is ready while any pre-registration step is outstanding.
 */
export function readiness(plan) {
  if (!isAvailable(plan)) return { state: 'UNKNOWN', tone: 'warn', reasons: [] };
  const reasons = plan.blocked_reasons || [];
  if (reasons.length > 0) return { state: 'BLOCKED', tone: 'warn', reasons };
  const done = plan.total_completed_runs || 0;
  const planned = plan.total_planned_runs || 0;
  if (planned > 0 && done >= planned) return { state: 'COMPLETE', tone: 'ok', reasons: [] };
  if (done > 0) return { state: 'RUNNING', tone: 'ok', reasons: [] };
  return { state: 'READY', tone: 'ok', reasons: [] };
}

/** Grouped P1b cells: one row per width, one column per K. */
export function p1bMatrix(p1b) {
  if (!isAvailable(p1b) || !Array.isArray(p1b.cells)) return { widths: [], kValues: [], cell: () => null };
  const widths = [...new Set(p1b.cells.map((c) => c.width))];
  const kValues = [...new Set(p1b.cells.map((c) => c.k))].sort((a, b) => a - b);
  const index = new Map(p1b.cells.map((c) => [`${c.width}/${c.k}`, c]));
  return { widths, kValues, cell: (width, k) => index.get(`${width}/${k}`) ?? null };
}

/** The D5 outcome, or null when P1b has not produced a report. */
export function d5Summary(p1b) {
  const decision = p1b?.d5?.decision;
  if (!decision) return null;
  const width = decision.width_decision || {};
  const budget = decision.budget_decision || {};
  return {
    widthOutcome: width.outcome ?? null,
    selectedWidth: width.selected_width ?? null,
    kLowAtFloor: width.k_low_at_floor ?? null,
    kLow: width.k_low ?? null,
    budgetOutcome: budget.outcome ?? null,
    recommendedMaxSteps: budget.recommended_max_steps ?? null,
    notConverged: budget.not_converged ?? null,
    perK: budget.per_k ?? null,
    budgetsAgree: budget.k_budgets_agree ?? null,
  };
}
