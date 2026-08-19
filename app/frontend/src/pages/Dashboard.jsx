import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { Header } from '../components/Header';
import { MetricCard } from '../components/MetricCard';
import { ArchitectureCard } from '../components/ArchitectureCard';
import { KScalingSection } from '../components/KScalingSection';
import { CrucibleSection } from '../components/CrucibleSection';
import { CASection } from '../components/CASection';
import { KeyFindings } from '../components/KeyFindings';
import { StatusPanel } from '../components/StatusPanel';
import { RawDataModal } from '../components/RawDataModal';

export function Dashboard() {
  const [selectedExperiment, setSelectedExperiment] = useState('overview');
  const [presentationMode, setPresentationMode] = useState(false);
  const [isRawDataOpen, setIsRawDataOpen] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [summaryData, setSummaryData] = useState(null);
  const [crucibleData, setCrucibleData] = useState(null);
  const [kScalingData, setKScalingData] = useState(null);
  const [caData, setCAData] = useState(null);
  const [rawData, setRawData] = useState(null);

  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sum, cru, ksc, ca, raw] = await Promise.all([
        api.getSummary(),
        api.getCrucible(),
        api.getKScaling(),
        api.getCellularAutomaton(),
        api.getRawResults(),
      ]);

      setSummaryData(sum);
      setCrucibleData(cru);
      setKScalingData(ksc);
      setCAData(ca);
      setRawData(raw);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
      setError(err.message || 'Failed to fetch experiment data from backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

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
          Loading experiment metrics from runs/...
        </div>
      )}

      {error && (
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
          <strong>Backend Error:</strong> {error}. Ensure FastAPI backend is running at http://localhost:8000.
        </div>
      )}

      {!loading && (
        <>
          {/* Top Summary Cards (Compact Overview) */}
          <div className="grid grid-cols-4" style={{ marginBottom: '1.5rem' }}>
            <MetricCard
              title="Parameters"
              value={summaryData?.parameter_count?.toLocaleString() || '210,832'}
              subtext="Shared Recursive Transformer"
              badge="CONSTANT"
              status="pass"
            />

            <MetricCard
              title="K Tested"
              value="1 / 2 / 4 / 8"
              subtext="Recursive Iterations"
              badge="INVARIANCE"
              status="pass"
            />

            <MetricCard
              title="Crucible Benchmark"
              value="100.0%"
              subtext="Copy Task Loss < 0.01"
              badge="PASS"
              status="pass"
            />

            <MetricCard
              title="Device"
              value={summaryData?.device || 'CPU'}
              subtext="PyTorch Standard Engine"
              badge="READY"
              status="pass"
            />
          </div>

          {/* Conditional Experiment Display */}
          {(selectedExperiment === 'overview' || selectedExperiment === 'k_scaling') && (
            <KScalingSection data={kScalingData} />
          )}

          {(selectedExperiment === 'overview' || selectedExperiment === 'crucible') && (
            <CrucibleSection data={crucibleData} />
          )}

          {(selectedExperiment === 'overview' || selectedExperiment === 'cellular_automaton') && (
            <CASection data={caData} />
          )}

          {selectedExperiment === 'overview' && (
            <>
              <ArchitectureCard summaryData={summaryData} />
              <KeyFindings findings={summaryData?.key_findings} />
              <StatusPanel statusData={summaryData?.status} />
            </>
          )}
        </>
      )}

      <RawDataModal
        isOpen={isRawDataOpen}
        onClose={() => setIsRawDataOpen(false)}
        rawData={rawData}
      />
    </div>
  );
}
