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
import { fmtInt, fmtNum, isNum, MISSING, verdictStatus } from '../format';
import { Provenance, Unavailable } from './Provenance';

const TITLE = 'Crucible (Copy Task Overfitting & BPTT Verification)';

export function CrucibleSection({ data, error }) {
  if (error || !data || data.status !== 'available') {
    return <Unavailable title={TITLE} payload={data} error={error} />;
  }

  const s = data.summary;
  const bptt = data.bptt;
  const lossSeries = data.metrics.filter((m) => isNum(m.loss) && m.loss > 0);
  const bpttData = (bptt?.steps ?? []).map((step) => ({ step: `z_L^(${step.step})`, grad_norm: step.grad_z_L_norm }));
  const numExamples = s?.dataset?.num_examples;
  const exactCount =
    isNum(s?.final_exact_match_accuracy) && isNum(numExamples)
      ? Math.round((s.final_exact_match_accuracy / 100) * numExamples)
      : null;

  return (
    <div className="section">
      <div className="section-header">
        <h2 className="section-title">{TITLE}</h2>
        <span className={`badge badge-${verdictStatus(s?.verdict)}`}>
          {s ? `${s.verdict} — ${s.stopped_reason} at step ${s.training_steps}` : 'No verdict recorded (no summary.json)'}
        </span>
      </div>

      <div className="grid grid-cols-4" style={{ marginBottom: '1.25rem' }}>
        <div className="card">
          <div className="card-title">Final Loss</div>
          <div className="card-value">{fmtNum(s?.final_loss, 6)}</div>
          <div className="card-subtext">
            {s ? `Pass if < ${s.pass_criteria.max_final_loss}; initial ${fmtNum(s.initial_loss, 4)}` : MISSING}
          </div>
        </div>
        <div className="card">
          <div className="card-title">Token Accuracy</div>
          <div className="card-value">{fmtNum(s?.final_token_accuracy, 1, '%')}</div>
          <div className="card-subtext">seq_len = {s?.dataset?.seq_len ?? MISSING}</div>
        </div>
        <div className="card">
          <div className="card-title">Exact Match</div>
          <div className="card-value">{fmtNum(s?.final_exact_match_accuracy, 1, '%')}</div>
          <div className="card-subtext">
            {exactCount !== null ? `${exactCount} / ${numExamples} full sequences` : MISSING}
          </div>
        </div>
        <div className="card">
          <div className="card-title">Recursive Depth</div>
          <div className="card-value">K = {s?.k ?? bptt?.k ?? MISSING}</div>
          <div className="card-subtext">{fmtInt(s?.parameter_count)} params</div>
        </div>
      </div>

      <div className="grid grid-cols-2">
        <div className="card">
          <div className="card-title">
            <span>Training Loss vs Step</span>
            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>log scale · metrics.csv</span>
          </div>
          <div style={{ width: '100%', height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossSeries} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="step" stroke="#64748b" />
                <YAxis stroke="#64748b" scale="log" domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                  formatter={(val) => [fmtNum(val, 6), 'Loss']}
                />
                <Line type="monotone" dataKey="loss" stroke="#f43f5e" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          {s && <div className="card-subtext">{s.final_metrics_note}</div>}
        </div>

        <div className="card">
          <div className="card-title">
            <span>Recursive State Gradient Norms (BPTT)</span>
            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--accent-purple)' }}>||∂L / ∂z_L^(k)||</span>
          </div>
          {bptt ? (
            <>
              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={bpttData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="step" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111827', borderColor: '#1e293b', color: '#f8fafc' }}
                      formatter={(val) => [fmtNum(val, 6), 'Gradient norm']}
                    />
                    <Bar dataKey="grad_norm" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="card-subtext">
                {bptt.passed ? 'PASS' : `FAIL: ${bptt.failures.join('; ')}`} — {bptt.evaluated_on}
              </div>
            </>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>No bptt_verification.json in this run.</p>
          )}
        </div>
      </div>
      <Provenance payload={data} />
    </div>
  );
}
