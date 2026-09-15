import React from 'react';

/** Where a section's numbers came from: run id, time, commit — or a legacy warning. */
export function Provenance({ payload }) {
  const run = payload?.run;
  if (!run) {
    return (
      <div className="card-subtext" style={{ color: 'var(--accent-amber)', marginTop: '0.75rem' }}>
        ⚠ {payload?.provenance_note || 'No run metadata recorded.'}
      </div>
    );
  }
  const commit = run.git?.commit ? run.git.commit.slice(0, 10) : 'unknown commit';
  const statusColor = run.status === 'completed' ? 'var(--text-muted)' : 'var(--accent-rose)';
  return (
    <div className="card-subtext mono" style={{ marginTop: '0.75rem', color: statusColor }}>
      {payload.output_dir}/ · run {run.run_id} · {run.status}
      {run.finished_at ? ` ${run.finished_at}` : ''} · {commit}
      {run.git?.dirty ? ' (uncommitted changes)' : ''} · seed {run.seed} · {run.device?.type}
      {run.error ? ` · error: ${run.error}` : ''}
    </div>
  );
}

/** Rendered instead of a section when its artifacts are missing or malformed. */
export function Unavailable({ title, payload, error }) {
  const isError = error || payload?.status === 'error';
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>{title}</span>
        <span className={`badge ${isError ? 'badge-fail' : 'badge-warn'}`}>{isError ? 'ERROR' : 'NO DATA'}</span>
      </div>
      <p style={{ color: isError ? 'var(--accent-rose)' : 'var(--text-muted)' }}>
        {error || payload?.message || 'Not loaded.'}
      </p>
    </div>
  );
}
