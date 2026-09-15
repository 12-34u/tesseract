import React from 'react';

function badgeFor(exp) {
  if (exp.artifact_status === 'error') return ['badge-fail', 'ARTIFACT ERROR'];
  if (exp.artifact_status === 'missing') return ['badge-warn', 'NO ARTIFACTS'];
  if (exp.run_status && exp.run_status !== 'completed') return ['badge-fail', `RUN ${exp.run_status.toUpperCase()}`];
  if (exp.verdict === 'PASS') return ['badge-pass', 'PASS'];
  if (exp.verdict === 'FAIL') return ['badge-fail', 'FAIL'];
  if (exp.verdict) return ['badge-warn', exp.verdict];
  return ['badge-warn', 'LEGACY — UNVERIFIED'];
}

export function StatusPanel({ experiments }) {
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">Experiment Runs & Verification Status</div>
      {!experiments?.length ? (
        <p style={{ color: 'var(--text-muted)' }}>No experiment status available.</p>
      ) : (
        <div className={`grid grid-cols-${experiments.length}`}>
          {experiments.map((exp) => {
            const [badgeClass, label] = badgeFor(exp);
            return (
              <div key={exp.id} style={{ padding: '0.75rem', backgroundColor: '#090d16', borderRadius: '0.375rem', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{exp.name}</div>
                <div style={{ marginTop: '0.35rem' }}>
                  <span className={`badge ${badgeClass}`}>{label}</span>
                </div>
                <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                  {exp.run_id ?? 'no run id'}
                  {exp.finished_at ? ` · ${exp.finished_at}` : ''}
                  {exp.git_commit ? ` · ${exp.git_commit.slice(0, 10)}${exp.git_dirty ? ' (dirty)' : ''}` : ''}
                </div>
                {(exp.message || exp.provenance_note) && (
                  <div style={{ fontSize: '0.7rem', color: 'var(--accent-amber)', marginTop: '0.25rem' }}>
                    {exp.message || exp.provenance_note}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
