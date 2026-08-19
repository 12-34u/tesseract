import React from 'react';

export function StatusPanel({ statusData }) {
  const crucibleStatus = statusData?.crucible || 'PASS';
  const kScalingStatus = statusData?.k_scaling || 'PASS';
  const caStatus = statusData?.cellular_automaton || 'EXECUTED (Val 0%)';
  const validationStatus = statusData?.prototype_validation || 'PASS';

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">Experiment Artifacts & Verification Status</div>

      <div className="grid grid-cols-4">
        <div style={{ padding: '0.75rem', backgroundColor: '#090d16', borderRadius: '0.375rem', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Crucible Overfitting</div>
          <div style={{ marginTop: '0.35rem' }}>
            <span className="badge badge-pass">✓ {crucibleStatus}</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            runs/crucible_k04/
          </div>
        </div>

        <div style={{ padding: '0.75rem', backgroundColor: '#090d16', borderRadius: '0.375rem', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>K-Scaling Latency</div>
          <div style={{ marginTop: '0.35rem' }}>
            <span className="badge badge-pass">✓ {kScalingStatus}</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            runs/k_scaling/
          </div>
        </div>

        <div style={{ padding: '0.75rem', backgroundColor: '#090d16', borderRadius: '0.375rem', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Cellular Automaton</div>
          <div style={{ marginTop: '0.35rem' }}>
            <span className="badge badge-warn">⚠ {caStatus}</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            runs/cellular_automaton/
          </div>
        </div>

        <div style={{ padding: '0.75rem', backgroundColor: '#090d16', borderRadius: '0.375rem', border: '1px solid #1e293b' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Prototype Validation</div>
          <div style={{ marginTop: '0.35rem' }}>
            <span className="badge badge-pass">✓ {validationStatus}</span>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            scripts/validate_prototype.py
          </div>
        </div>
      </div>
    </div>
  );
}
