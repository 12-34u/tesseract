/**
 * Presentation helpers for the Phase 1 cellular-automaton T x K matrix.
 *
 * Presentation only: nothing here reads, derives or alters a recorded value.
 * The Phase 1 results are historical and fixed — in particular the 0.0 %
 * validation exact match is a genuine, documented result, not a fault, and is
 * never styled as an error.
 */

/**
 * Chance level for a metric, as a percentage.
 *
 * Rule 90 is binary, so a single token is right half the time by chance. A
 * whole sequence is not: exact match has no comparable floor, so it is treated
 * as zero.
 */
export const CHANCE_PERCENT = { token_acc: 50, exact_match: 0 };

/**
 * Colour a matrix cell by how far it sits above chance, not by "low is red".
 *
 * Shading is proportional to the margin above the metric's chance level, which
 * is what actually distinguishes the cells. A cell at or below chance gets no
 * fill rather than a red one.
 */
export function cellStyle(value, metricMode) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return { background: 'transparent', color: 'var(--text-muted)' };
  }
  const chance = CHANCE_PERCENT[metricMode] ?? 0;
  const margin = Math.max(0, (value - chance) / Math.max(1, 100 - chance));
  return {
    background: margin === 0 ? 'transparent' : `rgba(53, 198, 244, ${0.06 + margin * 0.34})`,
    color: margin === 0 ? 'var(--text-secondary)' : 'var(--text-primary)',
  };
}
