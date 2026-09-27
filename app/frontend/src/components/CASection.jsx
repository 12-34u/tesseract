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
import { fmtInt, fmtNum, isNum, MISSING, uniqueSorted } from '../format';
import { Provenance, Unavailable } from './Provenance';
import { cellStyle } from '../phase1/caMatrix';

const TITLE = 'Phase 1 · Cellular Automaton (Rule 90, Transformation Depth T × Recursive Depth K)';
const LINE_COLORS = ['#35c6f4', '#4fae7e', '#c9963f', '#8b7cf6', '#c9646f', '#5b9ef5'];
const METRIC_KEYS = {
  train: { exact_match: 'train_exact_match', token_acc: 'train_token_accuracy' },
  val: { exact_match: 'val_exact_match', token_acc: 'val_token_accuracy' },
};
const pct = (val) => [fmtNum(val, 1, '%'), 'Accuracy'];

export function CASection({ data, error }) {
  // Token accuracy is the default view: validation exact match is 0.0 % in
  // every cell, which is a genuine recorded result but tells the reader
  // nothing about how the cells differ. Exact match stays one click away.
  const [metricMode, setMetricMode] = useState('token_acc');
  const [splitMode, setSplitMode] = useState('val');
  const [selectedT, setSelectedT] = useState(null);
  const [selectedK, setSelectedK] = useState(null);

  if (error || !data || data.status !== 'available') {
    return <Unavailable title={TITLE} payload={data} error={error} />;
  }

  const results = data.results;
  const task = data.config?.data ?? data.config?.task ?? {}; // new schema: data, legacy: task
  const tValues = uniqueSorted(results.map((r) => r.T));
  const kValues = uniqueSorted(results.map((r) => r.K));
  const params = uniqueSorted(results.map((r) => r.parameter_count));
  const t = selectedT ?? tValues[0];
  const k = selectedK ?? kValues[0];

  const find = (tv, kv) => results.find((r) => r.T === tv && r.K === kv);
  const cell = (tv, kv, split, metric) => {
    const value = find(tv, kv)?.[METRIC_KEYS[split][metric]];
    return isNum(value) ? value : null;
  };

  const valEm = results.map((r) => r.val_exact_match).filter(isNum);
  const valTok = results.map((r) => r.val_token_accuracy).filter(isNum);
  const allValEmZero = valEm.length > 0 && Math.max(...valEm) === 0;

  const lineData = kValues.map((kv) => {
    const point = { K: `K=${kv}` };
    tValues.forEach((tv) => {
      point[`T=${tv}`] = cell(tv, kv, splitMode, metricMode);
    });
    return point;
  });

  const selected = find(t, k);
  const barData = [
    {
      name: `T=${t}, K=${k}`,
      'Train Exact Match': selected?.train_exact_match ?? null,
      'Val Exact Match': selected?.val_exact_match ?? null,
      'Train Token Acc': selected?.train_token_accuracy ?? null,
      'Val Token Acc': selected?.val_token_accuracy ?? null,
    },
  ];

  return (
    <div className="section">
      <div className="section-header">
        <h2 className="section-title">{TITLE}</h2>
        <span className={`badge ${allValEmZero ? 'badge-warn' : 'badge-pass'}`}>
          {valEm.length ? `Val exact match: ${fmtNum(Math.min(...valEm))}–${fmtNum(Math.max(...valEm), 1, '%')}` : 'No validation data'}
        </span>
      </div>

      <div className="grid grid-cols-4" style={{ marginBottom: '1.25rem' }}>
        <div className="card">
          <div className="card-title">Automaton Rule</div>
          <div className="card-value">Rule {task.rule_number ?? MISSING}</div>
          <div className="card-subtext">Binary 1D, periodic boundary</div>
        </div>
        <div className="card">
          <div className="card-title">Sequence Length</div>
          <div className="card-value">N = {task.seq_len ?? MISSING}</div>
          <div className="card-subtext">Exact match needs all N tokens right</div>
        </div>
        <div className="card">
          <div className="card-title">Train / Val</div>
          <div className="card-value">{task.num_train ?? MISSING} / {task.num_val ?? MISSING}</div>
          <div className="card-subtext">Disjoint initial states</div>
        </div>
        <div className="card">
          <div className="card-title">Trainable Parameters</div>
          <div className="card-value">{params.length === 1 ? fmtInt(params[0]) : params.map(fmtInt).join(' / ') || MISSING}</div>
          <div className="card-subtext">{params.length === 1 ? 'Constant across K' : 'Not constant across K'}</div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div className="card-title">
          <span>T × K Matrix</span>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <div className="tab-group">
              {['train', 'val'].map((split) => (
                <button key={split} className={`tab-btn ${splitMode === split ? 'active' : ''}`} onClick={() => setSplitMode(split)}>
                  {split === 'train' ? 'Train' : 'Validation'}
                </button>
              ))}
            </div>
            <div className="tab-group">
              {[['exact_match', 'Exact Match (%)'], ['token_acc', 'Token Accuracy (%)']].map(([mode, label]) => (
                <button key={mode} className={`tab-btn ${metricMode === mode ? 'active' : ''}`} onClick={() => setMetricMode(mode)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <table className="matrix-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>T \ K</th>
              {kValues.map((kv) => (
                <th key={kv}>K = {kv}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tValues.map((tv) => (
              <tr key={tv}>
                <td style={{ fontWeight: 'bold', color: 'var(--accent-cyan)', textAlign: 'left' }}>T = {tv}</td>
                {kValues.map((kv) => {
                  const val = cell(tv, kv, splitMode, metricMode);
                  const { background, color } = cellStyle(val, metricMode);
                  return (
                    <td key={kv} className="matrix-cell" style={{ backgroundColor: background, color }}>
                      {fmtNum(val, 1, '%')}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <div className="card-subtext" style={{ marginTop: '0.75rem' }}>
          {splitMode === 'train' ? 'Train' : 'Validation'} {metricMode === 'exact_match' ? 'exact match' : 'token accuracy'} at each
          cell's last evaluation step. {MISSING} = not recorded.
          {allValEmZero &&
            ` Validation exact match is 0.0% in all ${valEm.length} cells while validation token accuracy is ` +
              `${fmtNum(Math.min(...valTok))}–${fmtNum(Math.max(...valTok), 1, '%')} (chance ≈ 50%): with N tokens per sequence, ` +
              'a single wrong token fails the whole sequence. See PHASE1_FINAL_AUDIT.md.'}
        </div>
        <div className="card-subtext" style={{ marginTop: '0.5rem' }}>
          Cells are shaded by margin above chance
          {metricMode === 'token_acc' ? ' (50 % for binary Rule 90)' : ''}, not by low-is-bad.
          Rule 90 is degenerate on this T grid: for T = 2<sup>m</sup> the target collapses to a two-cell XOR,
          target[i] = x[i−T] ⊕ x[i+T], so larger T widens the span without deepening the computation.
          See PHASE2_BENCHMARK_DESIGN.md.
        </div>
      </div>

      <div className="grid grid-cols-2">
        <div className="card">
          <div className="card-title">
            <span>
              {splitMode === 'train' ? 'Train' : 'Val'} {metricMode === 'exact_match' ? 'Exact Match' : 'Token Accuracy'} vs K
            </span>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lineData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1b2432" />
                <XAxis dataKey="K" stroke="#64748b" />
                <YAxis stroke="#64748b" domain={[0, 100]} />
                <Tooltip contentStyle={{ backgroundColor: '#0e131c', borderColor: '#1b2432', color: '#e6ecf5' }} formatter={pct} />
                <Legend />
                {tValues.map((tv, i) => (
                  <Line key={tv} type="monotone" dataKey={`T=${tv}`} stroke={LINE_COLORS[i % LINE_COLORS.length]} strokeWidth={2} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-title">
            <span>Train vs Validation</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <select className="select-input mono" value={t} onChange={(e) => setSelectedT(Number(e.target.value))}>
                {tValues.map((tv) => (
                  <option key={tv} value={tv}>T = {tv}</option>
                ))}
              </select>
              <select className="select-input mono" value={k} onChange={(e) => setSelectedK(Number(e.target.value))}>
                {kValues.map((kv) => (
                  <option key={kv} value={kv}>K = {kv}</option>
                ))}
              </select>
            </div>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1b2432" />
                <XAxis dataKey="name" stroke="#64748b" />
                <YAxis stroke="#64748b" domain={[0, 100]} />
                <Tooltip contentStyle={{ backgroundColor: '#0e131c', borderColor: '#1b2432', color: '#e6ecf5' }} formatter={pct} />
                <Legend />
                <Bar dataKey="Train Exact Match" fill="#35c6f4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Val Exact Match" fill="#c9646f" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Train Token Acc" fill="#4fae7e" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Val Token Acc" fill="#c9963f" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">
            Steps: {fmtInt(selected?.training_steps)} ({selected?.stopped_reason ?? 'stop reason not recorded'}) · fewest wrong
            tokens in any validation sequence: {fmtInt(selected?.val_min_token_errors)} · mean:{' '}
            {fmtNum(selected?.val_mean_token_errors, 2)}
          </div>
        </div>
      </div>
      <Provenance payload={data} />
    </div>
  );
}
