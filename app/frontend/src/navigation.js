/**
 * View <-> URL hash mapping for the dashboard.
 *
 * The selected view lives in the URL so a view can be linked to directly. That
 * only works in both directions if the app also *reads* the hash when it
 * changes: writing it from state alone leaves the browser's Back and Forward
 * buttons changing the URL while the page keeps showing the previous view.
 */

export const VIEWS = ['overview', 'crucible', 'k_scaling', 'cellular_automaton', 'phase2'];

export const DEFAULT_VIEW = 'overview';

/** The view a hash refers to, or the default when it names nothing valid. */
export function viewFromHash(hash, views = VIEWS, fallback = DEFAULT_VIEW) {
  const name = String(hash ?? '').replace(/^#/, '');
  return views.includes(name) ? name : fallback;
}

/**
 * Subscribe to browser navigation, calling `onChange` with the view the URL now
 * names. Returns an unsubscribe function.
 *
 * `hashchange` covers Back, Forward, and a hand-edited address bar alike.
 */
export function subscribeToHash(onChange, windowRef = typeof window === 'undefined' ? null : window) {
  if (!windowRef) return () => {};
  const handler = () => onChange(viewFromHash(windowRef.location.hash));
  windowRef.addEventListener('hashchange', handler);
  return () => windowRef.removeEventListener('hashchange', handler);
}
