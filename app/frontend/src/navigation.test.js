import { describe, expect, it, vi } from 'vitest';
import { DEFAULT_VIEW, VIEWS, subscribeToHash, viewFromHash } from './navigation';

describe('view from hash', () => {
  it('resolves every real view, with or without the leading hash', () => {
    for (const view of VIEWS) {
      expect(viewFromHash(`#${view}`)).toBe(view);
      expect(viewFromHash(view)).toBe(view);
    }
  });

  it('falls back to the default for anything it does not recognise', () => {
    for (const bad of ['', '#', '#nope', '#PHASE2', undefined, null, '#phase2extra']) {
      expect(viewFromHash(bad)).toBe(DEFAULT_VIEW);
    }
  });
});

describe('browser navigation', () => {
  /** A minimal window stand-in, so this needs no DOM. */
  function fakeWindow(hash = '#overview') {
    const listeners = {};
    return {
      location: { hash },
      addEventListener: (type, fn) => { (listeners[type] ||= []).push(fn); },
      removeEventListener: (type, fn) => { listeners[type] = (listeners[type] || []).filter((f) => f !== fn); },
      fire: (type) => (listeners[type] || []).forEach((fn) => fn()),
      count: (type) => (listeners[type] || []).length,
    };
  }

  it('reports the view the URL names when the browser navigates', () => {
    // The regression: Back and Forward changed the URL while the page kept
    // showing the previous view, because nothing read the hash back.
    const win = fakeWindow('#crucible');
    const seen = [];
    subscribeToHash((v) => seen.push(v), win);

    win.location.hash = '#overview';
    win.fire('hashchange');
    win.location.hash = '#phase2';
    win.fire('hashchange');

    expect(seen).toEqual(['overview', 'phase2']);
  });

  it('falls back rather than reporting a view that does not exist', () => {
    const win = fakeWindow();
    const seen = [];
    subscribeToHash((v) => seen.push(v), win);
    win.location.hash = '#garbage';
    win.fire('hashchange');
    expect(seen).toEqual([DEFAULT_VIEW]);
  });

  it('unsubscribes cleanly', () => {
    const win = fakeWindow();
    const onChange = vi.fn();
    const unsubscribe = subscribeToHash(onChange, win);
    expect(win.count('hashchange')).toBe(1);
    unsubscribe();
    expect(win.count('hashchange')).toBe(0);
    win.fire('hashchange');
    expect(onChange).not.toHaveBeenCalled();
  });

  it('is a no-op without a window, so server rendering does not throw', () => {
    expect(() => subscribeToHash(() => {}, null)()).not.toThrow();
  });
});
