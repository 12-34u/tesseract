import React, { useState, useEffect } from 'react';
import { api, API_BASE_URL } from '../services/api';
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

const ENDPOINTS = [
  ['summary', api.getSummary],
  ['crucible', api.getCrucible],
  ['kScaling', api.getKScaling],
  ['ca', api.getCellularAutomaton],
  ['raw', api.getRawResults],
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

export function Dashboard() {
  const [selectedExperiment, setSelectedExperiment] = useState('overview');
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

  const { summary, crucible, kScaling, ca, raw } = data;
  const counts = summary?.parameter_counts ?? [];
  const crucibleSummary = crucible?.summary;
  const errorList = Object.entries(errors);
  const show = (id) => selectedExperiment === 'overview' || selectedExperiment === id;

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

      {!loading && (
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
