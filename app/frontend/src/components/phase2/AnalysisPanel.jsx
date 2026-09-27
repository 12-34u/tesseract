/**
 * 11 — Analysis.
 *
 * The primary question, the three headline outputs, and the recorded decision
 * rule beside them. Thresholds are read from the amendment rather than restated
 * here, so the page cannot disagree with the decision it is describing.
 */

import React from 'react';
import { KeyValues, Panel, RuleHeading, Section, State } from './primitives.jsx';
import { analysisPlan, headlineAnalyses } from '../../phase2/pipeline.js';
import { MISSING } from '../../format.js';

export function AnalysisPanel({ status, showInternals = true }) {
  const plan = analysisPlan(status);
  const headline = headlineAnalyses();

  return (
    <Section id="analysis" index="11" title="Analysis" note={plan.primaryQuestion}>
      <Panel aside={<State tone={plan.pending ? 'idle' : 'ok'}>{plan.pending ? 'PENDING EXPERIMENTS' : 'AVAILABLE'}</State>}>
        <div className="rc-grid rc-cols-3">
          {headline.map((item) => (
            <div key={item.id}>
              <div className="rc-mono rc-figure" style={{ color: 'var(--rc-accent)' }}>{item.symbol}</div>
              <p className="rc-consequence" style={{ marginTop: '0.35rem' }}>{item.label}</p>
            </div>
          ))}
        </div>

        <div className="rc-grid rc-cols-wide" style={{ marginTop: '1.75rem' }}>
          <div>
            <RuleHeading>Secondary analyses</RuleHeading>
            <ul className="rc-list">
              {plan.items
                .filter((i) => !headline.some((h) => h.id === i.id))
                .map((item) => (
                  <li className="rc-list-item" key={item.id}>
                    <span className="rc-dot" aria-hidden="true" />
                    <span>
                      <span className="rc-item-label">{item.label}</span>
                      <br />
                      <span className="rc-item-detail">{item.detail}</span>
                    </span>
                  </li>
                ))}
            </ul>
          </div>

          {showInternals && (
            <div>
              <RuleHeading>Recorded decision rule</RuleHeading>
              <KeyValues
                fields={[
                  { label: 'Primary T', value: plan.primaryT.length ? plan.primaryT.join(', ') : MISSING, mono: true },
                  {
                    label: 'K low / high',
                    value: plan.kLow != null && plan.kHigh != null ? `${plan.kLow} / ${plan.kHigh}` : MISSING,
                    mono: true,
                  },
                  { label: 'tau', value: plan.tau ?? MISSING, mono: true },
                  { label: 'Floor', value: plan.floorThreshold ?? MISSING, mono: true },
                  { label: 'Score', value: plan.score ?? MISSING },
                ]}
              />
            </div>
          )}
        </div>
      </Panel>
    </Section>
  );
}
