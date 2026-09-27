/**
 * Phase 2 research console.
 *
 * Same props contract as before ({ status, error }), plus an optional
 * `presentationMode`, so the Dashboard wiring stays minimal. The data contract
 * is unchanged: every panel renders what the backend reported, and a block the
 * backend calls pending renders as pending rather than as a number.
 *
 * The page is a research paper turned into a console, read top to bottom:
 *
 *   01 overview -> 02 question -> 03 pipeline -> 04 benchmark -> 05 gates ->
 *   06 capacity -> 07 width gate -> 08 matrix -> 09 controls -> 10 results ->
 *   11 analysis -> implementation reference
 *
 * The navigation scrolls between them rather than hiding them, so the page
 * still reads as one continuous argument.
 *
 * Presentation mode is a different layout, not a different dataset: it drops
 * the diagnostics section and every implementation detail (commit hashes, run
 * directories, checksums, raw metric names) and keeps the research. No figure
 * changes between modes.
 *
 * Derived state lives in src/phase2/pipeline.js and src/phase2/derive.js, both
 * unit tested. Nothing is derived inline here.
 */

import React, { useCallback, useState } from 'react';
import '../phase2/console.css';

import { ErrorPanel } from './phase2/primitives.jsx';
import { Masthead } from './phase2/Masthead.jsx';
import { ResearchOverview } from './phase2/ResearchOverview.jsx';
import { ResearchQuestion } from './phase2/ResearchQuestion.jsx';
import { PipelineRail } from './phase2/PipelineRail.jsx';
import { BenchmarkPanel } from './phase2/BenchmarkPanel.jsx';
import { GatesPanel } from './phase2/GatesPanel.jsx';
import { CapacityPanel } from './phase2/CapacityPanel.jsx';
import { WidthGatePanel } from './phase2/WidthGatePanel.jsx';
import { PrimaryMatrixPanel } from './phase2/PrimaryMatrixPanel.jsx';
import { ControlsPanel } from './phase2/ControlsPanel.jsx';
import { ResultsPanel } from './phase2/ResultsPanel.jsx';
import { AnalysisPanel } from './phase2/AnalysisPanel.jsx';
import { MethodologyDrawer } from './phase2/MethodologyDrawer.jsx';
import { presentationFilter } from '../phase2/pipeline.js';

/** Which section each nav entry scrolls to. */
const TARGET = {
  overview: 'overview',
  methodology: 'methodology',
  diagnostics: 'diagnostics',
  experiments: 'experiments',
  results: 'results',
  analysis: 'analysis',
};

export function Phase2Section({ status, error, presentationMode = false }) {
  const [active, setActive] = useState('overview');

  const select = useCallback((id) => {
    setActive(id);
    const node = document.getElementById(TARGET[id] ?? id);
    if (node) node.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  if (error) {
    return (
      <div className="rc">
        <div className="rc-shell">
          <ErrorPanel>
            <strong>Phase 2 status failed to load.</strong>
            <div style={{ marginTop: '0.4rem' }}>{error}</div>
          </ErrorPanel>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="rc">
        <div className="rc-shell">
          <ErrorPanel>No Phase 2 status payload was returned by the backend.</ErrorPanel>
        </div>
      </div>
    );
  }

  const show = presentationFilter(presentationMode);
  // Presentation mode hides the diagnostics section, so the nav must not stay
  // marked on it. Derived during render rather than synced through an effect.
  const activeSection = presentationMode && active === 'diagnostics' ? 'overview' : active;

  return (
    <div className="rc">
      <div className="rc-shell">
        <Masthead active={activeSection} onSelect={select} presentationMode={presentationMode} />

        <ResearchOverview status={status} />
        <ResearchQuestion status={status} />
        <PipelineRail status={status} />
        <BenchmarkPanel status={status} showSources={show.showSourcePaths} />
        <GatesPanel status={status} showInternals={show.showDiagnosticInternals} />

        {!presentationMode && <CapacityPanel status={status} showInternals={show.showDiagnosticInternals} />}

        <WidthGatePanel status={status} showInternals={show.showDiagnosticInternals} />
        <PrimaryMatrixPanel status={status} />

        <div className="rc-section">
          <ControlsPanel status={status} />
        </div>

        <ResultsPanel status={status} showMetricNames={show.showPlannedMetricNames} />
        <AnalysisPanel status={status} showInternals={show.showDiagnosticInternals} />

        {show.showMethodologyDrawer && (
          <div className="rc-section">
            <MethodologyDrawer status={status} showSources={show.showSourcePaths} />
          </div>
        )}
      </div>
    </div>
  );
}
