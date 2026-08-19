import React, { useState } from 'react';
import { X, Download } from 'lucide-react';

export function RawDataModal({ isOpen, onClose, rawData }) {
  const [activeTab, setActiveTab] = useState('k_scaling');

  if (!isOpen) return null;

  const exportCsv = (filename, content) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getCsvContent = () => {
    if (activeTab === 'final_results') {
      return rawData?.final_results_csv || '';
    }
    const dataset = rawData?.[activeTab];
    if (!dataset || !dataset.results) return JSON.stringify(dataset, null, 2);
    
    const results = dataset.results;
    if (!results.length) return '';
    const headers = Object.keys(results[0]).join(',');
    const rows = results.map((r) => Object.values(r).join(',')).join('\n');
    return `${headers}\n${rows}`;
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>
            Raw Experiment Results Artifacts (runs/)
          </h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              className="btn btn-primary"
              onClick={() => exportCsv(`${activeTab}_export.csv`, getCsvContent())}
            >
              <Download size={14} />
              <span>Export CSV</span>
            </button>
            <button className="btn" onClick={onClose}>
              <X size={16} />
            </button>
          </div>
        </div>

        <div style={{ padding: '0.75rem 1.25rem', borderBottom: '1px solid var(--border-color)' }}>
          <div className="tab-group">
            <button
              className={`tab-btn ${activeTab === 'k_scaling' ? 'active' : ''}`}
              onClick={() => setActiveTab('k_scaling')}
            >
              K-Scaling CSV
            </button>
            <button
              className={`tab-btn ${activeTab === 'crucible' ? 'active' : ''}`}
              onClick={() => setActiveTab('crucible')}
            >
              Crucible CSV
            </button>
            <button
              className={`tab-btn ${activeTab === 'cellular_automaton' ? 'active' : ''}`}
              onClick={() => setActiveTab('cellular_automaton')}
            >
              Cellular Automaton CSV
            </button>
            <button
              className={`tab-btn ${activeTab === 'final_results' ? 'active' : ''}`}
              onClick={() => setActiveTab('final_results')}
            >
              Final Results Matrix CSV
            </button>
          </div>
        </div>

        <div className="modal-body">
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
            {getCsvContent()}
          </pre>
        </div>
      </div>
    </div>
  );
}
