import React, { useState } from 'react';
import { X, Download } from 'lucide-react';

const EXPERIMENT_LABELS = {
  crucible: 'Crucible',
  k_scaling: 'K-Scaling',
  cellular_automaton: 'Cellular Automaton',
};

/** Shows artifact files exactly as written by the experiments (no re-serialisation). */
export function RawDataModal({ isOpen, onClose, rawData }) {
  const [experiment, setExperiment] = useState('k_scaling');
  const [fileName, setFileName] = useState(null);

  if (!isOpen) return null;

  const files = rawData?.[experiment] ?? {};
  const names = Object.keys(files);
  const activeFile = fileName && names.includes(fileName) ? fileName : names[0];
  const content = activeFile ? files[activeFile] : null;

  const download = () => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `${experiment}_${activeFile}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>Raw Experiment Artifacts</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button className="btn btn-primary" onClick={download} disabled={typeof content !== 'string'}>
              <Download size={14} />
              <span>Download</span>
            </button>
            <button className="btn" onClick={onClose}>
              <X size={16} />
            </button>
          </div>
        </div>

        <div style={{ padding: '0.75rem 1.25rem', borderBottom: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div className="tab-group">
            {Object.entries(EXPERIMENT_LABELS).map(([id, label]) => (
              <button key={id} className={`tab-btn ${experiment === id ? 'active' : ''}`} onClick={() => { setExperiment(id); setFileName(null); }}>
                {label}
              </button>
            ))}
          </div>
          <div className="tab-group">
            {names.map((name) => (
              <button key={name} className={`tab-btn mono ${activeFile === name ? 'active' : ''}`} onClick={() => setFileName(name)}>
                {name}
              </button>
            ))}
          </div>
        </div>

        <div className="modal-body">
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {rawData ? (typeof content === 'string' ? content : `(${activeFile ?? 'file'} not present in this run)`) : '(raw data not loaded)'}
          </pre>
        </div>
      </div>
    </div>
  );
}
