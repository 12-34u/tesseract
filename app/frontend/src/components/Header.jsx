import React from 'react';
import { RefreshCw, Monitor, FileText, Activity } from 'lucide-react';

export function Header({
  selectedExperiment,
  setSelectedExperiment,
  presentationMode,
  setPresentationMode,
  onRefresh,
  onOpenRawData,
}) {
  return (
    <header className="header">
      <div className="header-title">
        <h1>
          <Activity size={24} style={{ color: 'var(--accent-cyan)' }} />
          TESSERACT
        </h1>
        <p>Transformer Engineered for Sequential State Evaluation and Recursive Action — Evaluation Dashboard</p>
        <p className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
          Phase 1 — completed / historical &nbsp;·&nbsp; Phase 2 — current research, pre-registration only
        </p>
      </div>

      <div className="header-controls">
        <label className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Experiment:
        </label>
        <select
          className="select-input mono"
          value={selectedExperiment}
          onChange={(e) => setSelectedExperiment(e.target.value)}
        >
          <option value="overview">Phase 1 · Overview</option>
          <option value="crucible">Phase 1 · Crucible (Copy Task)</option>
          <option value="k_scaling">Phase 1 · K-Scaling</option>
          <option value="cellular_automaton">Phase 1 · Cellular Automaton</option>
          <option value="phase2">Phase 2 · Status &amp; Pre-registration</option>
        </select>

        <button className="btn" onClick={onRefresh} title="Re-read experiment files from runs/">
          <RefreshCw size={16} />
          <span>Refresh</span>
        </button>

        <button className="btn" onClick={onOpenRawData} title="View Raw Experiment CSVs">
          <FileText size={16} />
          <span>Raw Data</span>
        </button>

        <button
          className={`btn ${presentationMode ? 'btn-active' : 'btn-primary'}`}
          onClick={() => setPresentationMode(!presentationMode)}
          title="Toggle Presentation Mode"
        >
          <Monitor size={16} />
          <span>{presentationMode ? 'Exit Presentation' : 'Presentation Mode'}</span>
        </button>
      </div>
    </header>
  );
}
