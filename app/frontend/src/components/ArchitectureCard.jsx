import React from 'react';

export function ArchitectureCard({ summaryData }) {
  const dModel = summaryData?.d_model || 128;
  const heads = summaryData?.num_heads || 4;
  const dFF = summaryData?.d_ff || 512;
  const params = summaryData?.parameter_count || 210832;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>Model Overview & Shared-Weight Architecture</span>
        <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>
          configs/prototype_small.yaml
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Trainable Parameters</span>
          <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>
            {params.toLocaleString()}
          </div>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Embedding Dim (d_model)</span>
          <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{dModel}</div>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Attention Heads</span>
          <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{heads}</div>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FFN Dimension</span>
          <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{dFF}</div>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Shared Blocks</span>
          <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>1 Block</div>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Latent State Mechanism</span>
          <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold', color: 'var(--accent-emerald)' }}>Dual z_H / z_L</div>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>BPTT Mechanism</span>
          <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold', color: 'var(--accent-purple)' }}>Full BPTT</div>
        </div>
      </div>

      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
        Computational Flow Pipeline:
      </div>

      <div className="arch-flow">
        <div className="arch-step">
          <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Input</span>
          <span>Tokens [B, N]</span>
        </div>
        <div className="arch-arrow">↓</div>
        <div className="arch-step">
          <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Embedding</span>
          <span>Token + Pos [B, N, D]</span>
        </div>
        <div className="arch-arrow">↓</div>
        <div className="arch-step" style={{ borderColor: 'var(--accent-cyan)' }}>
          <span style={{ color: 'var(--accent-cyan)', fontSize: '0.7rem' }}>Dual Latents</span>
          <span>z_H^(0) / z_L^(0)</span>
        </div>
        <div className="arch-arrow">↓</div>
        <div className="arch-step" style={{ borderColor: 'var(--accent-emerald)', backgroundColor: 'rgba(16,185,129,0.05)' }}>
          <span style={{ color: 'var(--accent-emerald)', fontSize: '0.7rem' }}>Weight Shared</span>
          <span>Transformer Block (K steps)</span>
        </div>
        <div className="arch-arrow">↓</div>
        <div className="arch-step">
          <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Output</span>
          <span>Logits [B, N, V]</span>
        </div>
      </div>
    </div>
  );
}
