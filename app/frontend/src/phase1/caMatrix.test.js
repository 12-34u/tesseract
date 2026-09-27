import { describe, expect, it } from 'vitest';
import { CHANCE_PERCENT, cellStyle } from './caMatrix';

describe('CA matrix cell shading', () => {
  it('uses 50 % chance for binary Rule 90 token accuracy', () => {
    expect(CHANCE_PERCENT.token_acc).toBe(50);
    expect(CHANCE_PERCENT.exact_match).toBe(0);
  });

  it('gives a recorded 0 % no fill and no error colour', () => {
    // Validation exact match is 0.0 % in every Phase 1 cell. That is a genuine
    // documented result, so it must never be styled as a failure.
    const style = cellStyle(0, 'exact_match');
    expect(style.background).toBe('transparent');
    expect(style.color).not.toContain('rose');
    expect(style.color).toBe('var(--text-secondary)');
  });

  it('treats token accuracy at chance as no margin', () => {
    expect(cellStyle(50, 'token_acc').background).toBe('transparent');
  });

  it('shades in proportion to the margin above chance', () => {
    const low = cellStyle(60, 'token_acc');
    const high = cellStyle(90, 'token_acc');
    const alpha = (s) => Number(s.background.match(/([\d.]+)\)$/)[1]);
    expect(alpha(low)).toBeGreaterThan(0.06);
    expect(alpha(high)).toBeGreaterThan(alpha(low));
    expect(alpha(high)).toBeLessThanOrEqual(0.4);
  });

  it('does not shade below chance', () => {
    expect(cellStyle(20, 'token_acc').background).toBe('transparent');
  });

  it('renders a missing value as muted rather than as zero', () => {
    for (const missing of [null, undefined, NaN]) {
      const style = cellStyle(missing, 'token_acc');
      expect(style.background).toBe('transparent');
      expect(style.color).toBe('var(--text-muted)');
    }
  });

  it('scales exact match against a zero floor', () => {
    const alpha = (s) => Number(s.background.match(/([\d.]+)\)$/)[1]);
    expect(alpha(cellStyle(100, 'exact_match'))).toBeCloseTo(0.4);
    expect(alpha(cellStyle(100, 'token_acc'))).toBeCloseTo(0.4);
    // The same raw value means different things on the two scales: 75 % token
    // accuracy is only halfway above its 50 % chance level, while 75 % exact
    // match is three quarters of the way above a zero floor. The token-accuracy
    // cell must therefore shade LESS.
    expect(alpha(cellStyle(75, 'token_acc'))).toBeLessThan(alpha(cellStyle(75, 'exact_match')));
    expect(alpha(cellStyle(75, 'token_acc'))).toBeCloseTo(0.23);
    expect(alpha(cellStyle(75, 'exact_match'))).toBeCloseTo(0.315);
  });
});
