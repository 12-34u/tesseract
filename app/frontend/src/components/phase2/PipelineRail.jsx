/**
 * 03 — Experiment pipeline.
 *
 * Seven compact nodes, each with its number, title, a short descriptor and a
 * state. The point of the section is to explain *why* the primary matrix is
 * locked, so the outstanding preconditions are listed beneath the rail rather
 * than buried inside one node.
 */

import React from 'react';
import { Section } from './primitives.jsx';
import { pipelineStages, stageTone } from '../../phase2/pipeline.js';

const STATE_LABEL = {
  DONE: 'VERIFIED',
  ACTIVE: 'IN PROGRESS',
  WAITING: 'WAITING',
  LOCKED: 'LOCKED',
  ATTENTION: 'NEEDS REVIEW',
};

/** A short descriptor per stage: what the stage is about, not its status. */
const DESCRIPTOR = {
  benchmark: 'A₅ dynamics',
  gates: 'G1 / G2',
  capacity: 'P1b / P1b-2',
  width: 'D5 / D6',
  budget: 'Step budget',
  probes: 'Batch 01 / 02',
  matrix: 'K × T',
  analysis: 'Δ(T), K*(T)',
};

export function PipelineRail({ status }) {
  const stages = pipelineStages(status);
  const locked = stages.find((s) => s.id === 'matrix');
  const blockers = locked?.blockers || [];

  return (
    <Section
      id="pipeline"
      index="03"
      title="Experiment pipeline"
      note="Each stage depends on the one before it. A stage that has not started is waiting, not failing."
    >
      <div className="rc-pipeline">
        {stages.map((stage) => (
          <div className="rc-stage" data-state={stage.state} key={stage.id}>
            <span className="rc-stage-index">{stage.index}</span>
            <span className="rc-stage-title">{stage.title}</span>
            <span className="rc-stage-desc">{DESCRIPTOR[stage.id]}</span>
            <span className="rc-stage-state">
              <span className="rc-dot" data-tone={stageTone(stage.state)} aria-hidden="true" />
              {STATE_LABEL[stage.state] ?? stage.state}
            </span>
          </div>
        ))}
      </div>

      {blockers.length > 0 && (
        <div className="rc-why">
          The primary matrix is locked because {blockers.length} precondition
          {blockers.length === 1 ? '' : 's'} {blockers.length === 1 ? 'is' : 'are'} outstanding:
          <ul>
            {blockers.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </div>
      )}
    </Section>
  );
}
