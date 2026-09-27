/**
 * 01 — Research overview.
 *
 * A compact research header: eyebrow, title, abstract, and two status blocks.
 * No hero fill, no full-width colour. Beneath it the experimental contract —
 * the fixed terms of the experiment — as a single bordered strip of small
 * technical cells rather than a row of pills.
 *
 * Both the status blocks and the contract are derived from payloads, so this
 * renders what the artifacts say and asserts nothing of its own.
 */

import React from 'react';
import { experimentalContract, overallState, runProgress } from '../../phase2/pipeline.js';
import { MISSING } from '../../format.js';

export function ResearchOverview({ status }) {
  const state = overallState(status);
  const progress = runProgress(status);
  const contract = experimentalContract(status);

  return (
    <div className="rc-section" id="overview">
      <div className="rc-lede">
        <p className="rc-eyebrow">Phase 2 · Research</p>
        <h1 className="rc-title">Recursive Depth vs. Sequential Transformation Depth</h1>
        <p className="rc-abstract">
          Does increasing recursive computation K improve performance on tasks requiring T sequential
          transformations, while keeping trainable parameters constant?
        </p>

        <div className="rc-lede-status">
          <span className="rc-status" data-tone={state.tone}>
            <span className="rc-status-label">{state.label}</span>
          </span>
          <span className="rc-status">
            <span className="rc-status-label rc-mono">
              {progress.planned > 0 ? `${progress.completed} / ${progress.planned} RUNS` : `${MISSING} RUNS`}
            </span>
          </span>
        </div>
        <p className="rc-consequence" style={{ marginTop: '0.7rem' }}>{state.detail}</p>
      </div>

      {contract.length > 0 && (
        <div className="rc-contract">
          {contract.map((term) => (
            <div className="rc-term" key={term.label}>
              <span className="rc-term-label">{term.label}</span>
              <span className={`rc-term-value${term.mono ? ' rc-mono' : ''}`}>{term.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
