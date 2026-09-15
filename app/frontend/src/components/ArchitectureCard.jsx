import React from 'react';
import { fmtInt, MISSING } from '../format';

function Field({ label, value, color }) {
  return (
    <div>
      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{label}</span>
      <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 'bold', color }}>
        {value ?? MISSING}
      </div>
    </div>
  );
}

export function ArchitectureCard({ summary, crucible }) {
  const model = summary?.model;
  const bptt = crucible?.bptt;

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>Model Overview & Shared-Weight Architecture</span>
        <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>
          {model?.source ?? 'no model config found in run artifacts'}
        </span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '1rem',
          marginBottom: '1rem',
        }}
      >
        <Field label="Trainable Parameters (K-scaling)" value={fmtInt(summary?.parameter_count)} color="var(--accent-cyan)" />
        <Field label="Embedding Dim (d_model)" value={model?.d_model} />
        <Field label="Attention Heads" value={model?.num_heads} />
        <Field label="FFN Dimension" value={model?.d_ff} />
        <Field label="Max Sequence Length" value={model?.max_seq_len} />
        <Field label="EMA alpha (z_H)" value={model?.alpha} />
        <Field label="TransformerBlock Instances (measured)" value={summary?.transformer_block_instances} />
        <Field
          label="BPTT Verification (Crucible)"
          value={bptt ? `${bptt.passed ? 'PASS' : 'FAIL'} (K=${bptt.k})` : undefined}
          color={bptt ? (bptt.passed ? 'var(--accent-emerald)' : 'var(--accent-rose)') : undefined}
        />
      </div>

      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
        Computational flow (models/tesseract.py):
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
          <span>learnable z_H^(0) / z_L^(0)</span>
        </div>
        <div className="arch-arrow">↓</div>
        <div className="arch-step" style={{ borderColor: 'var(--accent-emerald)', backgroundColor: 'rgba(16,185,129,0.05)' }}>
          <span style={{ color: 'var(--accent-emerald)', fontSize: '0.7rem' }}>Weight Shared × K</span>
          <span>z_L = Block(z_L + z_H + x); z_H = αz_H + (1−α)z_L</span>
        </div>
        <div className="arch-arrow">↓</div>
        <div className="arch-step">
          <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Output</span>
          <span>Linear(z_L^(K)) → Logits [B, N, V]</span>
        </div>
      </div>
    </div>
  );
}
