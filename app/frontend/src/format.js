/** Display helpers. Missing values render as an em dash and are never replaced by a default. */

export const MISSING = '—';

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export const fmtNum = (v, digits = 1, suffix = '') => (isNum(v) ? `${v.toFixed(digits)}${suffix}` : MISSING);

export const fmtInt = (v) => (isNum(v) ? v.toLocaleString() : MISSING);

export const uniqueSorted = (values) => [...new Set(values.filter(isNum))].sort((a, b) => a - b);

export const verdictStatus = (verdict) => {
  if (verdict === 'PASS') return 'pass';
  if (verdict === 'FAIL') return 'fail';
  return 'warn';
};
