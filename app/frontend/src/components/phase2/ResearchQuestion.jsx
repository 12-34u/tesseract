/**
 * 02 — Research question.
 *
 * The statement runs full width, as it would in a paper, with the relationship
 * between the three variables set beneath it as a plain technical diagram:
 * P held flat by a rule, K and T as ladders, each with its consequence stated.
 * The K and T ranges come from the benchmark payload, so the diagram cannot
 * drift from the configuration.
 */

import React from 'react';
import { Panel, Section } from './primitives.jsx';
import { isAvailable } from '../../phase2/pipeline.js';
import { MISSING } from '../../format.js';

export function ResearchQuestion({ status }) {
  const benchmark = status?.benchmark;
  const available = isAvailable(benchmark);
  const kValues = available && Array.isArray(benchmark.k_values) ? benchmark.k_values : [];
  const tValues = available && Array.isArray(benchmark.t_values) ? benchmark.t_values : [];
  const ladder = (values) => (values.length ? values.join('  →  ') : MISSING);

  return (
    <Section id="question" index="02" title="Research question">
      <Panel>
        <p className="rc-statement">
          When a task requires T sequential transformations, does increasing recursive depth K improve
          performance?
        </p>

        <div className="rc-grid rc-cols-3" style={{ marginTop: '1.75rem' }}>
          <div>
            <span className="rc-relation-key">
              Parameters <b>P</b>
            </span>
            <div className="rc-constant-bar" style={{ marginTop: '0.55rem' }}>
              <span>P</span>
              <span />
              <span>CONSTANT</span>
            </div>
            <p className="rc-consequence">One shared block, reused K times.</p>
          </div>

          <div>
            <span className="rc-relation-key">
              Recursion <b>K</b>
            </span>
            <div className="rc-ladder" style={{ marginTop: '0.55rem' }}>{ladder(kValues)}</div>
            <p className="rc-consequence">More computation, same parameters.</p>
          </div>

          <div>
            <span className="rc-relation-key">
              Task depth <b>T</b>
            </span>
            <div className="rc-ladder" style={{ marginTop: '0.55rem' }}>{ladder(tValues)}</div>
            <p className="rc-consequence">More sequential transformations required.</p>
          </div>
        </div>
      </Panel>
    </Section>
  );
}
