/**
 * 10 — Primary results.
 *
 * Until an analysis artifact exists this is an empty state plus reserved space
 * for the figures that will fill it. The slots are labelled outlines, never
 * cards pretending to be charts, and no placeholder number is invented.
 */

import React from 'react';
import { ChartSlot, EmptyState, Panel, RuleHeading, Section, State } from './primitives.jsx';
import { resultsState } from '../../phase2/pipeline.js';

const FIGURES = [
  'K × T HEATMAP',
  'EXACT MATCH',
  'TOKEN ACCURACY',
  'CHANCE-NORMALISED ACCURACY',
  'Δ(T)',
  'K*(T)',
  'COMPUTE VS PERFORMANCE',
];

export function ResultsPanel({ status, showMetricNames = true }) {
  const results = resultsState(status);

  if (results.available) {
    return (
      <Section id="results" index="10" title="Primary results">
        <Panel aside={<State tone="ok">RECORDED</State>}>
          <pre className="rc-equation" style={{ whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(results.analysis, null, 2)}
          </pre>
        </Panel>
      </Section>
    );
  }

  return (
    <Section id="results" index="10" title="Primary results">
      <Panel>
        <EmptyState title="Experiments have not started">
          Results will appear here after the diagnostic and execution gates are satisfied.
        </EmptyState>

        <div style={{ marginTop: '1.5rem' }}>
          <RuleHeading>Reserved figures</RuleHeading>
          <div className="rc-grid rc-cols-3">
            {FIGURES.map((figure) => (
              <ChartSlot label={figure} key={figure} />
            ))}
          </div>
        </div>

        {results.blockers.length > 0 && (
          <div style={{ marginTop: '1.5rem' }}>
            <RuleHeading>Outstanding preconditions</RuleHeading>
            <ul className="rc-list">
              {results.blockers.map((reason) => (
                <li className="rc-list-item" key={reason}>
                  <span className="rc-dot" data-tone="note" aria-hidden="true" />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {showMetricNames && results.plannedMetrics.length > 0 && (
          <div style={{ marginTop: '1.25rem' }}>
            <RuleHeading>Metrics the runner will record</RuleHeading>
            <p className="rc-source" style={{ margin: 0, lineHeight: 1.8 }}>
              {results.plannedMetrics.join(' · ')}
            </p>
          </div>
        )}
      </Panel>
    </Section>
  );
}
