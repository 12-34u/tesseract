/**
 * Building blocks for the Phase 2 research console.
 *
 * These carry the visual language and nothing else: no component here reads a
 * payload or decides a state. Anything derived lives in src/phase2/pipeline.js
 * so it can be tested without a DOM.
 */

import React, { useId, useState } from 'react';
import { ChevronRight } from 'lucide-react';

/** A status dot. `tone` is one of ok | active | idle | note | fail. */
export function Dot({ tone = 'idle', title }) {
  return <span className="rc-dot" data-tone={tone} title={title} aria-hidden="true" />;
}

/** A state written in monospace with a leading dot. Preferred over a filled badge. */
export function State({ tone = 'idle', children }) {
  return (
    <span className="rc-state-text" data-tone={tone}>
      <Dot tone={tone} />
      {children}
    </span>
  );
}

/** A bordered label. Used sparingly — a state usually reads better. */
export function Tag({ tone = 'idle', children, title }) {
  return (
    <span className="rc-tag" data-tone={tone} title={title}>
      {children}
    </span>
  );
}

/** A numbered research section. */
export function Section({ id, index, title, note, children }) {
  return (
    <section className="rc-section" id={id}>
      <div className="rc-section-head">
        <div>
          {index ? <p className="rc-eyebrow">{index}</p> : null}
          <h2 className="rc-section-title">{title}</h2>
        </div>
        {note ? <p className="rc-section-note">{note}</p> : null}
      </div>
      {children}
    </section>
  );
}

/** A bordered panel with an optional title row. */
export function Panel({ title, subtitle, aside, quiet = false, children, style }) {
  return (
    <div className={`rc-panel${quiet ? ' rc-panel-quiet' : ''}`} style={style}>
      {(title || aside) && (
        <div className="rc-panel-head">
          <div style={{ minWidth: 0 }}>
            {title ? <h3 className="rc-panel-title">{title}</h3> : null}
            {subtitle ? <p className="rc-panel-sub">{subtitle}</p> : null}
          </div>
          {aside}
        </div>
      )}
      {children}
    </div>
  );
}

/** A small uppercase rule heading, for sub-blocks inside a panel. */
export function RuleHeading({ children }) {
  return <p className="rc-rule-heading">{children}</p>;
}

/**
 * A collapsible region.
 *
 * Mathematical and implementation detail lives behind these, closed by default.
 * The button carries `aria-expanded` so the state is announced.
 */
export function Disclosure({ label, hint, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  return (
    <div className="rc-disclosure">
      <button
        type="button"
        className="rc-disclosure-btn"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="rc-disclosure-label">
          <ChevronRight size={13} className="rc-caret" data-open={open} />
          {label}
        </span>
        {hint ? <span className="rc-disclosure-hint">{hint}</span> : null}
      </button>
      {open ? (
        <div className="rc-disclosure-body" id={bodyId}>
          {children}
        </div>
      ) : null}
    </div>
  );
}

/** A definition list. Values are monospace when they are figures or paths. */
export function KeyValues({ fields }) {
  if (!fields?.length) return null;
  return (
    <dl className="rc-kv">
      {fields.map((f) => (
        <React.Fragment key={f.label}>
          <dt>{f.label}</dt>
          <dd className={f.mono ? 'rc-mono' : undefined}>{f.value}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

/** A displayed equation or formal definition. */
export function Equation({ children }) {
  return <div className="rc-equation">{children}</div>;
}

/**
 * The empty state for a section with no artifact.
 *
 * Waiting for an experiment is the normal condition here, not a fault, so this
 * never uses the danger styling.
 */
export function EmptyState({ title, children }) {
  return (
    <div className="rc-empty">
      <p className="rc-empty-title">{title}</p>
      {children ? <p className="rc-empty-body">{children}</p> : null}
    </div>
  );
}

/** Reserved space for a chart that has no data yet. Never a fake chart. */
export function ChartSlot({ label }) {
  return <div className="rc-placeholder">{label}</div>;
}

/** A backend failure. Distinct from a pending state, and always visible. */
export function ErrorPanel({ children }) {
  return <div className="rc-error">{children}</div>;
}

/** A pending block: shows the backend's own wording, never a substitute value. */
export function PendingNote({ children }) {
  return <p className="rc-prose" style={{ margin: 0 }}>{children}</p>;
}
