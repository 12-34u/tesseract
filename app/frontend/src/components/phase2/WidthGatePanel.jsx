/**
 * 07 — Research width gate.
 *
 * Two columns for two different decisions:
 *
 *   D5 — LEARNABILITY. The test. Does T = 8 become learnable, against the
 *        recorded criteria? The verdict is read from the D5 evaluator artifact
 *        through `widthSelection`; nothing is recomputed in the browser, and
 *        while the combined report is missing no width is shown as selected.
 *
 *   D6 — WIDTH POLICY. The designation. Which width the research runs at,
 *        pre-registered in the P1b-2 configuration rather than chosen on
 *        performance. Amendment 02 records D1 to D5, so there is no separate D6
 *        entry; if one is added it appears here with no code change, and the
 *        recorded D5 selection rule stays visible either way.
 *
 * The flow beneath makes the dependency explicit, and the awaiting state is
 * rendered as a normal condition rather than as an error.
 */

import React from 'react';
import { Disclosure, KeyValues, Panel, RuleHeading, Section, State, Tag } from './primitives.jsx';
import { d5CriteriaRows } from '../../phase2/derive.js';
import { d5Layer, d6Layer } from '../../phase2/pipeline.js';
import { MISSING } from '../../format.js';

const D5_CRITERIA = [
  'CN accuracy ≥ 0.02',
  'Final-20 mean ≥ 0.02',
  'Final loss < ln(60)',
];

export function WidthGatePanel({ status, showInternals = true }) {
  const d5 = d5Layer(status);
  const d6 = d6Layer(status);
  const criteria = d5CriteriaRows(status?.capacity_diagnostics);
  const designated = d6.designatedWidth;

  return (
    <Section
      id="widthgate"
      index="07"
      title="Research width gate"
      note="Two separate decisions: whether a width can learn the task, and which width the research runs at."
    >
      <div className="rc-grid rc-cols-2">
        <Panel
          title="D5 — Learnability"
          subtitle="Does T = 8 become learnable?"
          aside={<State tone={d5.evaluated ? 'ok' : 'idle'}>{d5.evaluated ? 'EVALUATED' : 'AWAITING'}</State>}
        >
          <ul className="rc-list">
            {D5_CRITERIA.map((c) => (
              <li className="rc-list-item" key={c}>
                <span className="rc-dot" aria-hidden="true" />
                <span className="rc-criterion">{c}</span>
              </li>
            ))}
          </ul>
          <div style={{ marginTop: '1.1rem' }}>
            <KeyValues
              fields={[
                { label: 'State', value: d5.state || MISSING, mono: true },
                { label: 'Awaiting', value: d5.awaiting.length ? d5.awaiting.join(' + ') : null, mono: true },
                { label: 'Selected width', value: d5.selectedWidth ?? MISSING, mono: true },
                {
                  label: 'Budget',
                  value: d5.recommendedMaxSteps != null ? `${d5.recommendedMaxSteps.toLocaleString()} steps` : MISSING,
                  mono: true,
                },
              ].filter((f) => f.value != null)}
            />
          </div>
        </Panel>

        <Panel
          title="D6 — Width policy"
          subtitle="Which width does the research run at?"
          aside={<Tag tone="active">PRE-REGISTERED</Tag>}
        >
          {designated ? (
            <>
              <span className="rc-designate">Designated research width</span>
              <div style={{ margin: '0.35rem 0 0.15rem', textTransform: 'capitalize', fontSize: '1rem', fontWeight: 600 }}>
                {designated.name}
              </div>
              <div className="rc-mono rc-figure" style={{ color: 'var(--rc-accent)' }}>
                {designated.parameterCount != null ? designated.parameterCount.toLocaleString() : MISSING}
              </div>
              <p className="rc-consequence">trainable parameters, constant across K</p>
              <p className="rc-consequence" style={{ marginTop: '0.75rem' }}>
                Designated in advance by the {d6.designationSource}, not selected on performance.
              </p>
            </>
          ) : (
            <p className="rc-prose" style={{ margin: 0 }}>
              No research-scale width is configured.
            </p>
          )}

          {d6.recordedPolicy && (
            <div className="rc-decision rc-decision-note" style={{ marginTop: '1rem' }}>
              {d6.recordedPolicy}
            </div>
          )}
        </Panel>
      </div>

      <Panel style={{ marginTop: 'var(--rc-gutter)' }}>
        <div className="rc-gateflow">
          <span className="rc-gatenode" data-emphasis={String(d5.evaluated)}>
            D5 LEARNABILITY
          </span>
          <span className="rc-gatearrow">→</span>
          <span className="rc-gatenode" data-emphasis="true">
            {designated ? designated.name.toUpperCase() : MISSING}
          </span>
          <span className="rc-gatearrow">→</span>
          <span className="rc-gatenode">PRIMARY MATRIX</span>
          <span style={{ marginLeft: 'auto' }}>
            <State tone={d5.evaluated ? 'ok' : 'idle'}>
              {d5.awaiting.length ? `AWAITING ${d5.awaiting.join(' + ')}` : d5.state || 'AWAITING'}
            </State>
          </span>
        </div>
        {!d5.evaluated && (
          <p className="rc-consequence" style={{ marginTop: '0.85rem' }}>
            If the designated width does not clear D5, the primary matrix stays blocked. The reference widths are
            diagnostics and cannot substitute for it.
          </p>
        )}
      </Panel>

      {showInternals && criteria.length > 0 && (
        <div style={{ marginTop: '0.5rem' }}>
          <Disclosure label="D5 criteria per width and K" hint="from the recorded decision">
            <RuleHeading>Recorded evaluation</RuleHeading>
            <table className="rc-table">
              <thead>
                <tr>
                  <th>Width</th>
                  <th>K</th>
                  <th>Final CN</th>
                  <th>Tail mean</th>
                  <th>Final loss</th>
                  <th>Learnable</th>
                </tr>
              </thead>
              <tbody>
                {criteria.map((row) => (
                  <tr key={`${row.width}/${row.k}`}>
                    <td style={{ textTransform: 'capitalize' }}>{row.width}</td>
                    <td className="rc-mono">{row.k}</td>
                    <td className="rc-mono">{row.finalScore != null ? row.finalScore.toFixed(4) : MISSING}</td>
                    <td className="rc-mono">{row.tailMean != null ? row.tailMean.toFixed(4) : MISSING}</td>
                    <td className="rc-mono">{row.finalLoss != null ? row.finalLoss.toFixed(4) : MISSING}</td>
                    <td>
                      <State tone={row.learnable ? 'ok' : 'note'}>{row.learnable ? 'YES' : 'NO'}</State>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Disclosure>
        </div>
      )}
    </Section>
  );
}
