import React, { useState, useEffect } from 'react';
import { api, API_BASE_URL } from '../services/api';
import { subscribeToHash, viewFromHash } from '../navigation';
import { fmtInt, fmtNum, verdictStatus } from '../format';
import { Header } from '../components/Header';
import { MetricCard } from '../components/MetricCard';
import { ArchitectureCard } from '../components/ArchitectureCard';
import { KScalingSection } from '../components/KScalingSection';
import { CrucibleSection } from '../components/CrucibleSection';
import { CASection } from '../components/CASection';
import { KeyFindings } from '../components/KeyFindings';
import { StatusPanel } from '../components/StatusPanel';
import { RawDataModal } from '../components/RawDataModal';
import { Phase2Section } from '../components/Phase2Section';
import {
  batch07ReportExists,
  completedMilestones,
  keyStats,
  limitations,
  parameterGroups,
  projectMilestones,
} from '../data/projectStatus';

function sourceHref(source) {
  return `${API_BASE_URL}/api/source/${source}`;
}

function SourceLink({ source }) {
  return (
    <a className="dashboard-source" href={sourceHref(source)} target="_blank" rel="noreferrer">
      {source}
    </a>
  );
}

function ProjectStatusPanel() {
  return (
    <section className="dashboard-panel card">
      <div className="card-title">
        Project Status
        <span className="status-count">{completedMilestones} of {projectMilestones.length} milestones done</span>
      </div>
      <div className="milestone-list">
        {projectMilestones.map((milestone) => (
          <div className="milestone-row" key={milestone.label}>
            <span className={`milestone-status milestone-${milestone.status.toLowerCase().replace(' ', '-')}`}>{milestone.status}</span>
            <span className="milestone-label">{milestone.label}</span>
            <SourceLink source={milestone.source} />
          </div>
        ))}
      </div>
    </section>
  );
}

function KeyStatsPanel() {
  return (
    <section className="dashboard-panel card">
      <div className="card-title">Key Stats</div>
      <div className="stat-list">
        {keyStats.map(([value, label, source]) => (
          <div className="stat-row" key={`${value}-${label}`}>
            <strong>{value}</strong>
            <span>{label}</span>
            <SourceLink source={source} />
          </div>
        ))}
        <div className="stat-row stat-pending">
          <strong>{batch07ReportExists ? 'reported' : 'pending'}</strong>
          <span>Batch 07 results</span>
          <SourceLink source="PHASE2_BATCH_07_PREREG.md" />
        </div>
      </div>
    </section>
  );
}

function ParametersPanel() {
  return (
    <section className="dashboard-panel card">
      <div className="card-title">Parameters</div>
      {parameterGroups.map((group) => (
        <div className="parameter-group" key={group.title}>
          <h3>{group.title}</h3>
          {group.rows.map(([name, value, source]) => (
            <div className="parameter-row" key={`${group.title}-${name}`}>
              <span>{name}</span>
              <strong>{value}</strong>
              <SourceLink source={source} />
            </div>
          ))}
        </div>
      ))}
    </section>
  );
}

function LimitationsPanel() {
  return (
    <section className="dashboard-panel card">
      <div className="card-title">Limitations</div>
      <ul className="limitations-list">
        {limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
      </ul>
    </section>
  );
}

const ENDPOINTS = [
  ['summary', api.getSummary],
  ['crucible', api.getCrucible],
  ['kScaling', api.getKScaling],
  ['ca', api.getCellularAutomaton],
  ['raw', api.getRawResults],
  ['phase2', api.getPhase2Status],
];

/** Fetch every endpoint; one failing endpoint does not hide the others. */
async function fetchAll() {
  const settled = await Promise.allSettled(ENDPOINTS.map(([, fetcher]) => fetcher()));
  const data = {};
  const errors = {};
  settled.forEach((result, i) => {
    const key = ENDPOINTS[i][0];
    if (result.status === 'fulfilled') data[key] = result.value;
    else errors[key] = result.reason?.message || String(result.reason);
  });
  return { data, errors };
}

/** The selected view is kept in the URL hash so a view can be linked to directly. */
function initialView() {
  return viewFromHash(typeof window === 'undefined' ? '' : window.location.hash);
}

export function Dashboard() {
  const [selectedExperiment, setSelectedExperiment] = useState(initialView);
  const [presentationMode, setPresentationMode] = useState(false);
  const [isRawDataOpen, setIsRawDataOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState({});
  const [data, setData] = useState({});

  const applyResult = (result) => {
    setData(result.data);
    setErrors(result.errors);
    setLoading(false);
  };

  const loadAllData = () => {
    setLoading(true);
    fetchAll().then(applyResult);
  };

  useEffect(() => {
    fetchAll().then(applyResult);
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') window.location.hash = selectedExperiment;
  }, [selectedExperiment]);

  // Read the hash back when the browser changes it. Without this, Back and
  // Forward move the URL while the page keeps showing the previous view.
  useEffect(() => subscribeToHash(setSelectedExperiment), []);

  const { summary, crucible, kScaling, ca, raw, phase2 } = data;
  const counts = summary?.parameter_counts ?? [];
  const crucibleSummary = crucible?.summary;
  const errorList = Object.entries(errors);
  const isPhase2 = selectedExperiment === 'phase2';
  const show = (id) => !isPhase2 && (selectedExperiment === 'overview' || selectedExperiment === id);

  return (
    <div className={`dashboard-container ${presentationMode ? 'presentation-mode' : ''}`}>
      <Header
        selectedExperiment={selectedExperiment}
        setSelectedExperiment={setSelectedExperiment}
        presentationMode={presentationMode}
        setPresentationMode={setPresentationMode}
        onRefresh={loadAllData}
        onOpenRawData={() => setIsRawDataOpen(true)}
      />

      {loading && (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading experiment artifacts…
        </div>
      )}

      {errorList.length > 0 && (
        <div
          style={{
            padding: '1.25rem',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid var(--accent-rose)',
            borderRadius: '0.5rem',
            color: 'var(--accent-rose)',
            marginBottom: '1.5rem',
          }}
        >
          <strong>Backend error</strong> ({API_BASE_URL}):
          <ul style={{ marginTop: '0.5rem', paddingLeft: '1.25rem' }}>
            {errorList.map(([key, message]) => (
              <li key={key}>{message}</li>
            ))}
          </ul>
        </div>
      )}

      {!loading && isPhase2 && (
        <Phase2Section status={phase2} error={errors.phase2} presentationMode={presentationMode} />
      )}

      {!loading && !isPhase2 && (
        <>
          <div className="grid grid-cols-4" style={{ marginBottom: '1.5rem' }}>
            <MetricCard
              title="Trainable Parameters"
              value={fmtInt(summary?.parameter_count)}
              subtext={
                counts.length > 1
                  ? `Differs across K: ${counts.map((c) => c.toLocaleString()).join(', ')}`
                  : 'Measured from each model (K-scaling results)'
              }
              badge={counts.length === 1 ? 'CONSTANT ACROSS K' : counts.length > 1 ? 'VARIES' : 'NO DATA'}
              status={counts.length === 1 ? 'pass' : counts.length > 1 ? 'fail' : 'warn'}
            />
            <MetricCard
              title="K Tested"
              value={summary?.k_tested?.length ? summary.k_tested.join(' / ') : '—'}
              subtext="Recursive depths in K-scaling results"
            />
            <MetricCard
              title="Crucible Exact Match"
              value={fmtNum(crucibleSummary?.final_exact_match_accuracy, 1, '%')}
              subtext={
                crucibleSummary
                  ? `Loss ${fmtNum(crucibleSummary.final_loss, 6)} after ${crucibleSummary.training_steps} steps`
                  : 'No Crucible summary.json'
              }
              badge={crucibleSummary?.verdict ?? 'NO VERDICT'}
              status={verdictStatus(crucibleSummary?.verdict)}
            />
            <MetricCard
              title="Device"
              value={summary?.device ? summary.device.toUpperCase() : '—'}
              subtext="Recorded in K-scaling run metadata"
            />
          </div>

          {show('k_scaling') && <KScalingSection data={kScaling} error={errors.kScaling} />}
          {show('crucible') && <CrucibleSection data={crucible} error={errors.crucible} />}
          {show('cellular_automaton') && <CASection data={ca} error={errors.ca} />}

          {selectedExperiment === 'overview' && (
            <>
              <div className="dashboard-overview-grid">
                <ProjectStatusPanel />
                <KeyStatsPanel />
                <ParametersPanel />
                <LimitationsPanel />
              </div>
              <ArchitectureCard summary={summary} crucible={crucible} />
              <KeyFindings findings={summary?.findings} />
              <StatusPanel experiments={summary?.experiments} />
            </>
          )}
        </>
      )}

      <RawDataModal isOpen={isRawDataOpen} onClose={() => setIsRawDataOpen(false)} rawData={raw} />
    </div>
  );
}
