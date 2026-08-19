import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';

export function CASection({ data }) {
  const [metricMode, setMetricMode] = useState('exact_match'); // 'exact_match' or 'token_acc'
  const [splitMode, setSplitMode] = useState('train'); // 'train' or 'val'
  const [selectedT, setSelectedT] = useState(1);
  const [selectedK, setSelectedK] = useState(4);

  if (!data || !data.available || !data.results) {
    return (
      <div className="card">
        <div className="card-title">Cellular Automaton Experiment</div>
        <p style={{ color: 'var(--text-muted)' }}>Cellular Automaton results are not available.</p>
      </div>
    );
  }

  const results = data.results;
  const tValues = [1, 2, 4, 8];
  const kValues = [1, 2, 4, 8];

  // Helper to extract matrix value
  const getCellValue = (t, k, split, metric) => {
    const item = results.find((r) => r.T === t && r.K === k);
    if (!item) return 0;
    if (split === 'train') {
      return metric === 'exact_match' ? item.train_exact_match : item.train_token_accuracy;
    } else {
      return metric === 'exact_match' ? item.val_exact_match : item.val_token_accuracy;
    }
  };

  // Prepare line chart data (Accuracy vs K for each T)
  const lineChartData = kValues.map((k) => {
    const obj = { K: `K=${k}` };
    tValues.forEach((t) => {
      obj[`T=${t}`] = getCellValue(t, k, splitMode, metricMode);
    });
    return obj;
  });

  // Prepare Train vs Val grouped bar chart for selected T and K
  const selectedItem = results.find((r) => r.T === selectedT && r.K === selectedK) || {};
  const groupedBarData = [
    {
      name: `T=${selectedT}, K=${selectedK}`,
      'Train Exact Match': selectedItem.train_exact_match || 0,
      'Val Exact Match': selectedItem.val_exact_match || 0,
      'Train Token Acc': selectedItem.train_token_accuracy || 0,
      'Val Token Acc': selectedItem.val_token_accuracy || 0,
    },
  ];

  return (
    <div className="section">
      <div className="section-header">
        <h2 className="section-title">
          Iterative Transformation Evaluation (Cellular Automaton Rule 90)
        </h2>
        <span className="badge badge-warn">POST-AUG-21 — Overfits Train (Val 0.0%)</span>
      </div>

      {/* Task Parameters Top Cards */}
      <div className="grid grid-cols-4" style={{ marginBottom: '1.25rem' }}>
        <div className="card">
          <div className="card-title">Automaton Rule</div>
          <div className="card-value" style={{ color: 'var(--accent-amber)' }}>Rule 90</div>
          <div className="card-subtext">Binary 1D Sequence</div>
        </div>

        <div className="card">
          <div className="card-title">Sequence Length</div>
          <div className="card-value">N = 32</div>
          <div className="card-subtext">32 Spatial Tokens</div>
        </div>

        <div className="card">
          <div className="card-title">Train / Val Split</div>
          <div className="card-value">256 / 64</div>
          <div className="card-subtext">Small Sample Regime</div>
        </div>

        <div className="card">
          <div className="card-title">Trainable Parameters</div>
          <div className="card-value" style={{ color: 'var(--accent-cyan)' }}>207,234</div>
          <div className="card-subtext">Constant Across K</div>
        </div>
      </div>

      {/* Interactive Matrix / Heatmap */}
      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div className="card-title">
          <span>Transformation Depth (T) vs Recursive Depth (K) Matrix</span>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <div className="tab-group">
              <button
                className={`tab-btn ${splitMode === 'train' ? 'active' : ''}`}
                onClick={() => setSplitMode('train')}
              >
                Train Accuracy
              </button>
              <button
                className={`tab-btn ${splitMode === 'val' ? 'active' : ''}`}
                onClick={() => setSplitMode('val')}
              >
                Validation Accuracy
              </button>
            </div>

            <div className="tab-group">
              <button
                className={`tab-btn ${metricMode === 'exact_match' ? 'active' : ''}`}
                onClick={() => setMetricMode('exact_match')}
              >
                Exact Match (%)
              </button>
              <button
                className={`tab-btn ${metricMode === 'token_acc' ? 'active' : ''}`}
                onClick={() => setMetricMode('token_acc')}
              >
                Token Accuracy (%)
              </button>
            </div>
          </div>
        </div>

        <table className="matrix-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Required T \ Recursive K</th>
              {kValues.map((k) => (
                <th key={k}>K = {k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tValues.map((t) => (
              <tr key={t}>
                <td style={{ fontWeight: 'bold', color: 'var(--accent-cyan)', textAlign: 'left' }}>
                  T = {t} (Steps)
                </td>
                {kValues.map((k) => {
                  const val = getCellValue(t, k, splitMode, metricMode);
                  const isZero = val === 0;
                  const intensity = val / 100;
                  const bg = isZero
                    ? 'rgba(244, 63, 94, 0.08)'
                    : `rgba(6, 182, 212, ${0.1 + intensity * 0.4})`;

                  return (
                    <td
                      key={k}
                      className="matrix-cell"
                      style={{
                        backgroundColor: bg,
                        color: isZero ? 'var(--accent-rose)' : '#ffffff',
                      }}
                    >
                      {val.toFixed(1)}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <div className="card-subtext" style={{ marginTop: '0.75rem' }}>
          Currently showing: <strong>{splitMode.toUpperCase()}</strong>{' '}
          {metricMode === 'exact_match' ? 'Exact Match Accuracy' : 'Token Accuracy'}. Note: Out-of-distribution validation exact-match remains 0.0% across all entries, demonstrating a significant training/validation generalization gap.
        </div>
      </div>

      {/* Graphs */}
      <div className="grid grid-cols-2">
        {/* Performance Graph: Accuracy vs K for T=1,2,4,8 */}
        <div className="card">
          <div className="card-title">
            <span>
              {splitMode === 'train' ? 'Train' : 'Val'}{' '}
              {metricMode === 'exact_match' ? 'Exact Match' : 'Token Acc'} vs K
            </span>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineChartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="K" stroke="#64748b" />
                <YAxis stroke="#64748b" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                  formatter={(val) => [`${val.toFixed(1)}%`, 'Accuracy']}
                />
                <Legend />
                <Line type="monotone" dataKey="T=1" stroke="#06b6d4" strokeWidth={2} />
                <Line type="monotone" dataKey="T=2" stroke="#10b981" strokeWidth={2} />
                <Line type="monotone" dataKey="T=4" stroke="#f59e0b" strokeWidth={2} />
                <Line type="monotone" dataKey="T=8" stroke="#8b5cf6" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">
            Performance comparison across recursive depth K for transformation depth T=1, 2, 4, 8.
          </div>
        </div>

        {/* Grouped Bar Chart: Train vs Validation Gap */}
        <div className="card">
          <div className="card-title">
            <span>Train vs Validation Gap</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <select
                className="select-input mono"
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                value={selectedT}
                onChange={(e) => setSelectedT(Number(e.target.value))}
              >
                {tValues.map((t) => (
                  <option key={t} value={t}>T = {t}</option>
                ))}
              </select>

              <select
                className="select-input mono"
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                value={selectedK}
                onChange={(e) => setSelectedK(Number(e.target.value))}
              >
                {kValues.map((k) => (
                  <option key={k} value={k}>K = {k}</option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={groupedBarData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#64748b" />
                <YAxis stroke="#64748b" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                  formatter={(val) => [`${val.toFixed(1)}%`, 'Accuracy']}
                />
                <Legend />
                <Bar dataKey="Train Exact Match" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Val Exact Match" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">
            Highlights overfitting gap: high training exact match accuracy vs 0.0% validation exact match.
          </div>
        </div>
      </div>
    </div>
  );
}
