import React from 'react';
import { CheckCircle, AlertTriangle, Info } from 'lucide-react';

const ICONS = {
  ok: <CheckCircle size={18} style={{ color: 'var(--accent-emerald)', flexShrink: 0, marginTop: '2px' }} />,
  warn: <AlertTriangle size={18} style={{ color: 'var(--accent-amber)', flexShrink: 0, marginTop: '2px' }} />,
  info: <Info size={18} style={{ color: 'var(--accent-cyan)', flexShrink: 0, marginTop: '2px' }} />,
};

export function KeyFindings({ findings }) {
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>Key Findings</span>
        <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          computed by the backend from run artifacts
        </span>
      </div>

      {!findings?.length ? (
        <p style={{ color: 'var(--text-muted)' }}>No findings: no experiment artifacts could be read.</p>
      ) : (
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {findings.map((finding, idx) => (
            <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', fontSize: '0.9rem' }}>
              {ICONS[finding.level] ?? ICONS.info}
              <span>
                {finding.text}{' '}
                <span className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  [{finding.source}]
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
