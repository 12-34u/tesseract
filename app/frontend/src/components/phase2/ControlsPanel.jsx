/**
 * 09 — Controls and baselines.
 *
 * Collapsed by default: essential to the argument, but not to the story. Arms
 * are read from the plan, so a newly added arm appears here without a code
 * change, and each carries the purpose it serves in the design.
 */

import React from 'react';
import { Disclosure, State } from './primitives.jsx';
import { controlSummary } from '../../phase2/pipeline.js';

export function ControlsPanel({ status }) {
  const controls = controlSummary(status);
  if (!controls.available) return null;

  return (
    <Disclosure
      label="Controls & baselines"
      hint={`${controls.rows.length} arms · ${controls.plannedRuns} planned runs`}
    >
      <p className="rc-prose" style={{ marginTop: 0, marginBottom: '1rem', maxWidth: '72ch' }}>
        Controls test whether an observed K dependence is specific to recursive sequential computation, rather than
        to capacity, span, or unrolling.
      </p>
      <table className="rc-table">
        <thead>
          <tr>
            <th>Control</th>
            <th>Purpose</th>
            <th>Planned</th>
            <th>State</th>
          </tr>
        </thead>
        <tbody>
          {controls.rows.map((row) => (
            <tr key={row.key}>
              <td style={{ fontWeight: 500 }}>{row.name}</td>
              <td style={{ color: 'var(--rc-text-dim)' }}>{row.purpose}</td>
              <td className="rc-mono">{row.plannedRuns}</td>
              <td>
                <State tone={row.state === 'BLOCKED' ? 'note' : 'ok'}>{row.state}</State>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Disclosure>
  );
}
