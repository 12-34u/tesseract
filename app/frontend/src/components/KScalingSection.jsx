import React from 'react';
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
} from 'recharts';
import { fmtInt, fmtNum, isNum, MISSING } from '../format';
import { Provenance, Unavailable } from './Provenance';

const TITLE = 'K-Scaling & Computational Depth';

export function KScalingSection({ data, error }) {
  if (error || !data || data.status !== 'available') {
    return <Unavailable title={TITLE} payload={data} error={error} />;
  }

  const results = [...data.results].sort((a, b) => a.K - b.K);
  const device = data.run?.device?.type?.toUpperCase() ?? 'device unrecorded';
  const params = results.map((r) => r.parameter_count).filter(isNum);
  const constant = params.length === results.length && new Set(params).size === 1;
  const callsMeasured = Boolean(data.run);
  const callsMatch = results.every((r) => r.recursive_calls === r.K);
  const hasTraining = results.some((r) => r.final_loss !== undefined);
  const yPad = Math.max(1, Math.round(Math.max(...params) * 0.1));

  return (
    <div className="section">
      <div className="section-header">
        <h2 className="section-title">{TITLE}</h2>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <span className={`badge ${constant ? 'badge-pass' : 'badge-fail'}`}>
            {constant ? 'Parameter count constant' : 'Parameter count varies'}
          </span>
          <span className={`badge ${!callsMeasured ? 'badge-warn' : callsMatch ? 'badge-pass' : 'badge-fail'}`}>
            {!callsMeasured ? 'Block calls not verified (legacy)' : callsMatch ? 'Measured block calls = K' : 'Block calls ≠ K'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2" style={{ marginBottom: '1.25rem' }}>
        <div className="card">
          <div className="card-title">
            <span>Trainable Parameters vs K</span>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={results} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="K" stroke="#64748b" />
                <YAxis domain={[Math.min(...params) - yPad, Math.max(...params) + yPad]} stroke="#64748b" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                  formatter={(val) => [fmtInt(val), 'Trainable parameters']}
                />
                <Line type="monotone" dataKey="parameter_count" stroke="#06b6d4" strokeWidth={3} dot={{ r: 6, fill: '#06b6d4' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">Counted from each instantiated model in results.csv.</div>
        </div>

        <div className="card">
          <div className="card-title">
            <span>Median Forward Latency vs K ({device})</span>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={results} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="K" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                  formatter={(val) => [fmtNum(val, 2, ' ms'), 'Median latency']}
                />
                <Bar dataKey="latency_median_ms" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">Wall-clock inference time (no_grad); not a FLOP count.</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <span>Shared-Block Executions per Forward Pass</span>
          <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            recursive_calls column
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${results.length}, 1fr)`, gap: '1rem' }}>
          {results.map((r) => (
            <div key={r.K} style={{ padding: '0.75rem', backgroundColor: '#090d16', borderRadius: '0.375rem', border: '1px solid #1e293b' }}>
              <div className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>K = {r.K}</div>
              <div style={{ color: 'var(--accent-cyan)', fontSize: '1.1rem', letterSpacing: '0.15em', margin: '0.3rem 0' }}>
                {isNum(r.recursive_calls) ? '█'.repeat(r.recursive_calls) : MISSING}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{fmtInt(r.recursive_calls)} executions</div>
            </div>
          ))}
        </div>

        {hasTraining && (
          <>
            <div className="card-title" style={{ marginTop: '1.25rem' }}>
              <span>Training Diagnostic (copy task, identical initial weights)</span>
            </div>
            <table className="matrix-table">
              <thead>
                <tr>
                  <th>K</th><th>Initial loss</th><th>Final loss</th><th>Token acc.</th><th>Exact match</th><th>Steps</th><th>Stop</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.K}>
                    <td>{r.K}</td>
                    <td>{fmtNum(r.initial_loss, 4)}</td>
                    <td>{fmtNum(r.final_loss, 6)}</td>
                    <td>{fmtNum(r.token_accuracy, 1, '%')}</td>
                    <td>{fmtNum(r.exact_match_accuracy, 1, '%')}</td>
                    <td>{fmtInt(r.training_steps)}</td>
                    <td>{r.stopped_reason ?? MISSING}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="card-subtext" style={{ marginTop: '0.5rem' }}>
              Secondary diagnostic: the copy task needs no iteration, and initial loss grows with K, so steps-to-converge is confounded.
            </div>
          </>
        )}
        <Provenance payload={data} />
      </div>
    </div>
  );
}
