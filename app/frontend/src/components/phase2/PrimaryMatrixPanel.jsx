/**
 * 08 — Primary experiment matrix.
 *
 * A scientific experiment table. The grid, seeds and run counts all come from
 * the recorded plan, so the figures cannot drift from
 * configs/phase2/phase2_experiment.yaml. Every cell reads LOCKED until a result
 * artifact exists; no cell ever shows a number that was not recorded.
 */

import React from 'react';
import { KeyValues, Panel, PendingNote, Section, State } from './primitives.jsx';
import { primaryMatrix } from '../../phase2/pipeline.js';
import { MISSING } from '../../format.js';

export function PrimaryMatrixPanel({ status }) {
  const matrix = primaryMatrix(status);
  if (!matrix.available) {
    return (
      <Section id="experiments" index="08" title="Primary experiment matrix">
        <Panel>
          <PendingNote>{status?.plan?.message || 'No experiment plan recorded.'}</PendingNote>
        </Panel>
      </Section>
    );
  }

  const families = status?.model_families?.families || [];
  const designated = families.find((f) => f.name === 'large') || families[families.length - 1] || null;
  const cellLabel = matrix.locked ? 'LOCKED' : 'PLANNED';

  return (
    <Section
      id="experiments"
      index="08"
      title="Primary experiment matrix"
      note={`${designated ? `${designated.display || designated.name} model · ` : ''}A₅ benchmark · primary depth conditions`}
    >
      <Panel>
        <div className="rc-grid rc-cols-3" style={{ marginBottom: '1.25rem' }}>
          <KeyValues
            fields={[{
              label: 'Model',
              value: designated
                ? `${designated.display || designated.name} · ${designated.parameter_count?.toLocaleString()}`
                : MISSING,
              mono: true,
            }]}
          />
          <KeyValues fields={[{ label: 'Behaviour', value: 'P constant across K' }]} />
          <KeyValues
            fields={[{ label: 'Design', value: `${matrix.plannedRuns} runs · ${matrix.seeds.length} seeds`, mono: true }]}
          />
        </div>

        <table className="rc-table rc-table-center">
          <thead>
            <tr>
              <th />
              {matrix.kValues.map((k) => (
                <th key={k} className="rc-mono">K = {k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.tValues.map((t) => (
              <tr key={t}>
                <td className="rc-mono" style={{ fontWeight: 600 }}>T = {t}</td>
                {matrix.kValues.map((k) => {
                  const cell = matrix.cell(t, k);
                  return (
                    <td key={k}>
                      <span className="rc-cell" data-state={cell.state} title={`T=${t}, K=${k}`}>
                        {cellLabel}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <p className="rc-consequence" style={{ marginTop: '0.9rem' }}>
          Experiments unlock automatically once the research-width and budget gates pass.
          {' '}
          {matrix.completedRuns} of {matrix.plannedRuns} primary runs recorded.
        </p>

        {matrix.stages.length > 0 && (
          <table className="rc-table" style={{ marginTop: '1.25rem' }}>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Depth</th>
                <th>Planned</th>
                <th>Recorded</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {matrix.stages.map((s) => (
                <tr key={`${s.label}-${s.stage}`}>
                  <td>{s.label}</td>
                  <td className="rc-mono">{s.stage.replace('t', 'T = ')}</td>
                  <td className="rc-mono">{s.plannedRuns}</td>
                  <td className="rc-mono">{s.completedRuns}</td>
                  <td>
                    <State tone={s.state === 'BLOCKED' ? 'note' : 'ok'}>{s.state}</State>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </Section>
  );
}
