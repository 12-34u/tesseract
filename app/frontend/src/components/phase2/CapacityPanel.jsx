/**
 * 06 — Capacity diagnostic.
 *
 * A comparison table rather than three large cards. The designated research
 * width is marked with a thin cyan rule and a small label; it is never called a
 * winner, the best, or selected because it performs better — the designation is
 * pre-registered in the P1b-2 configuration, and the D5 selection rule has not
 * run.
 *
 * Recorded metrics are shown for cells that have them, through the existing
 * `capacityGrid` / `hasCellMetrics` reading. A cell with no artifact shows an
 * em dash and never a substitute number.
 */

import React from 'react';
import { Disclosure, KeyValues, Panel, PendingNote, RuleHeading, Section, State } from './primitives.jsx';
import { capacityGrid, hasCellMetrics } from '../../phase2/derive.js';
import { widthCards } from '../../phase2/pipeline.js';
import { MISSING } from '../../format.js';

function diagnosticTone(state) {
  if (state === 'COMPLETE') return 'ok';
  if (state === 'RUNNING') return 'active';
  return 'idle';
}

export function CapacityPanel({ status, showInternals = true }) {
  const capacity = status?.capacity_diagnostics;
  const widths = widthCards(status);
  const { rows } = capacityGrid(capacity);

  if (!widths.length) {
    return (
      <Section id="diagnostics" index="06" title="Capacity diagnostic">
        <Panel>
          <PendingNote>{capacity?.message || 'No capacity diagnostic configured.'}</PendingNote>
        </Panel>
      </Section>
    );
  }

  const kValues = [...new Set(rows.map((r) => r.k))].sort((a, b) => a - b);
  const byWidth = new Map();
  rows.forEach((row) => {
    if (!byWidth.has(row.width)) byWidth.set(row.width, new Map());
    byWidth.get(row.width).set(row.k, row);
  });
  const blocks = [capacity?.p1b, capacity?.p1b2].filter(Boolean);

  return (
    <Section
      id="diagnostics"
      index="06"
      title="Capacity diagnostic"
      note="Determines whether a candidate width can learn T = 8 before the primary matrix is run."
    >
      <Panel>
        <table className="rc-table">
          <thead>
            <tr>
              <th>Width</th>
              <th>Parameters</th>
              <th>Role</th>
              <th>Diagnostic</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {widths.map((w) => (
              <tr key={w.name} className={w.researchScale ? 'rc-row-accent' : undefined}>
                <td style={{ textTransform: 'capitalize', fontWeight: 500 }}>
                  {w.name}
                  {w.researchScale && (
                    <>
                      <br />
                      <span className="rc-designate">Designated research width</span>
                    </>
                  )}
                </td>
                <td className="rc-mono rc-figure">
                  {w.parameterCount != null ? w.parameterCount.toLocaleString() : MISSING}
                </td>
                <td style={{ color: 'var(--rc-text-dim)' }}>{w.researchScale ? 'Research width' : 'Reference'}</td>
                <td className="rc-mono">{w.diagnostic}</td>
                <td>
                  <State tone={diagnosticTone(w.diagnosticState)}>{String(w.diagnosticState).toUpperCase()}</State>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{ marginTop: '1.5rem' }}>
          <RuleHeading>Final validation, chance-normalised</RuleHeading>
          <table className="rc-table rc-table-center">
            <thead>
              <tr>
                <th>Width</th>
                {kValues.map((k) => (
                  <th key={k} className="rc-mono">
                    K = {k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...byWidth.entries()].map(([width, cells]) => (
                <tr key={width}>
                  <td style={{ textTransform: 'capitalize' }}>{width}</td>
                  {kValues.map((k) => {
                    const cell = cells.get(k)?.cell;
                    if (!hasCellMetrics(cell)) {
                      return (
                        <td key={k} className="rc-mono" style={{ color: 'var(--rc-text-faint)' }}>
                          {MISSING}
                        </td>
                      );
                    }
                    return (
                      <td key={k} className="rc-mono">
                        <div>{cell.final_val_chance_normalised?.toFixed(4)}</div>
                        <div style={{ fontSize: '0.68rem', color: 'var(--rc-text-faint)' }}>
                          loss {cell.final_val_loss?.toFixed(3)}
                          {cell.at_floor_final ? ' · at floor' : ''}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="rc-consequence" style={{ marginTop: '0.6rem' }}>
            A width with no recorded diagnostic shows {MISSING}.
          </p>
        </div>

        {showInternals && (
          <div style={{ marginTop: '1.25rem' }}>
            <Disclosure label="Diagnostic scope and settings" hint="what this does and does not establish">
              {capacity?.note ? <div className="rc-decision">{capacity.note}</div> : null}
              {blocks.map((block) => (
                <div key={block.label} style={{ marginTop: '1rem' }}>
                  <RuleHeading>{block.label}</RuleHeading>
                  <KeyValues
                    fields={[
                      { label: 'Widths', value: (block.widths || []).join(', '), mono: true },
                      { label: 'Scope', value: block.scope },
                      { label: 'Seeds', value: block.single_seed_note },
                      { label: 'Report', value: block.report_path, mono: true },
                      { label: 'Max steps', value: block.settings?.max_steps?.toLocaleString(), mono: true },
                      { label: 'Floor threshold', value: block.settings?.floor_threshold, mono: true },
                      { label: 'Provenance', value: block.provenance_note },
                    ].filter((f) => f.value != null && f.value !== '')}
                  />
                </div>
              ))}
            </Disclosure>
          </div>
        )}
      </Panel>
    </Section>
  );
}
