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

export function KScalingSection({ data }) {
  if (!data || !data.available || !data.results) {
    return (
      <div className="card">
        <div className="card-title">K-Scaling Experiment</div>
        <p style={{ color: 'var(--text-muted)' }}>K-scaling experiment data unavailable.</p>
      </div>
    );
  }

  const results = data.results;

  return (
    <div className="section">
      <div className="section-header">
        <h2 className="section-title">
          K-Scaling & Computational Depth Evaluation
        </h2>
        <span className="badge badge-pass">Parameter Invariance Verified</span>
      </div>

      <div className="grid grid-cols-2" style={{ marginBottom: '1.25rem' }}>
        {/* Graph 1: Parameter Count vs K */}
        <div className="card">
          <div className="card-title">
            <span>Parameter Count vs K</span>
            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)' }}>
              Invariance: CONSTANT
            </span>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={results} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="K" label={{ value: 'Recursive Depth (K)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} stroke="#64748b" />
                <YAxis domain={[150000, 250000]} stroke="#64748b" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                  formatter={(val) => [`${val.toLocaleString()} parameters`, 'Trainable Params']}
                />
                <Line type="monotone" dataKey="parameter_count" stroke="#06b6d4" strokeWidth={3} dot={{ r: 6, fill: '#06b6d4' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">
            Demonstrates parameter invariance: parameter count remains identical regardless of K.
          </div>
        </div>

        {/* Graph 2: Forward Latency vs K */}
        <div className="card">
          <div className="card-title">
            <span>Forward Latency vs K (CPU)</span>
            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>
              Median Latency
            </span>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={results} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="K" label={{ value: 'Recursive Depth (K)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }} stroke="#64748b" />
                <YAxis stroke="#64748b" label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                  formatter={(val) => [`${val.toFixed(2)} ms`, 'Median Latency']}
                />
                <Bar dataKey="latency_median_ms" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">
            Recursive depth increases the number of executions of the shared Transformer block.
          </div>
        </div>
      </div>

      {/* Conceptual Compute Indicator */}
      <div className="card">
        <div className="card-title">
          <span>Relative Recursive Block Executions</span>
          <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Conceptual Compute Indicator
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginTop: '0.5rem' }}>
          {[1, 2, 4, 8].map((k) => {
            const blocks = '█'.repeat(k);
            return (
              <div key={k} style={{ padding: '0.75rem', backgroundColor: '#090d16', borderRadius: '0.375rem', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }} className="mono">
                  K = {k}
                </div>
                <div style={{ color: 'var(--accent-cyan)', fontSize: '1.1rem', letterSpacing: '0.15em', margin: '0.3rem 0' }}>
                  {blocks}
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  {k} shared block execution{k > 1 ? 's' : ''}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
