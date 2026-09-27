/**
 * 04 — Benchmark and methodology.
 *
 * Two columns: the formal definition on the left, the rationale on the right.
 * Mathematical detail stays behind a disclosure so the section reads as an
 * argument rather than a specification.
 */

import React from 'react';
import { Disclosure, Equation, KeyValues, Panel, PendingNote, RuleHeading, Section, Tag } from './primitives.jsx';
import { isAvailable, tRoleRows } from '../../phase2/pipeline.js';

const ROLE_TONE = { Anchor: 'idle', Diagnostic: 'note', Primary: 'active' };

export function BenchmarkPanel({ status, showSources = true }) {
  const benchmark = status?.benchmark;
  if (!isAvailable(benchmark)) {
    return (
      <Section id="benchmark" index="04" title="Benchmark">
        <Panel>
          <PendingNote>{benchmark?.message || 'Benchmark configuration not found.'}</PendingNote>
        </Panel>
      </Section>
    );
  }

  const roles = tRoleRows(status);
  const vocab = status?.model_families?.vocab_size;

  return (
    <Section id="methodology" index="04" title="Benchmark & methodology">
      <div className="rc-grid rc-cols-wide">
        <Panel title={benchmark.group_description || `${benchmark.group} cellular automaton`}>
          <Equation>{benchmark.transition}</Equation>

          <div style={{ marginTop: '1rem' }}>
            <KeyValues
              fields={[
                { label: 'Ring size', value: `n = ${benchmark.n}`, mono: true },
                { label: 'Group order', value: vocab != null ? `|G| = ${vocab}` : '—', mono: true },
                { label: 'Depth grid', value: `T = {${(benchmark.t_values || []).join(', ')}}`, mono: true },
              ]}
            />
          </div>

          {roles.length > 0 && (
            <table className="rc-table" style={{ marginTop: '1.25rem' }}>
              <thead>
                <tr>
                  <th>Depth</th>
                  <th>Role</th>
                </tr>
              </thead>
              <tbody>
                {roles.map((row) => (
                  <tr key={row.t}>
                    <td className="rc-mono">T = {row.t}</td>
                    <td>
                      <Tag tone={ROLE_TONE[row.role] ?? 'idle'}>{row.role.toUpperCase()}</Tag>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Panel>

        <Panel title="Why this benchmark?">
          <div className="rc-prose">
            <p>
              A₅ is non-solvable, so composing its elements cannot be collapsed into a shallow parallel form. Depth
              in the task therefore corresponds to genuine sequential composition rather than to a wider receptive
              field.
            </p>
            <p>
              Phase 1 used a cellular automaton that turned out to be linear over GF(2): its targets could be
              computed by a single XOR of two distant cells, so increasing T widened the span without deepening the
              computation. The A₅ construction exists to remove exactly that shortcut.
            </p>
          </div>

          <div style={{ marginTop: '1.1rem' }}>
            <Disclosure label="View mathematical specification" hint="task, splits, seeds">
              <RuleHeading>Definition</RuleHeading>
              <KeyValues
                fields={[
                  { label: 'Transition', value: benchmark.transition, mono: true },
                  { label: 'Group', value: benchmark.group, mono: true },
                  { label: 'K grid', value: (benchmark.k_values || []).join(', '), mono: true },
                  { label: 'Tasks', value: (benchmark.tasks || []).join(', '), mono: true },
                ]}
              />
              <RuleHeading>Data</RuleHeading>
              <KeyValues
                fields={[
                  { label: 'Data seed', value: benchmark.data_seed, mono: true },
                  { label: 'Model seeds', value: (benchmark.model_seeds || []).join(', '), mono: true },
                  { label: 'Validation', value: benchmark.val_size?.toLocaleString(), mono: true },
                  { label: 'Test', value: benchmark.test_size?.toLocaleString(), mono: true },
                  { label: 'Length ablation n', value: benchmark.length_n, mono: true },
                  { label: 'Interpolation T', value: (benchmark.interpolation_t_values || []).join(', '), mono: true },
                  { label: 'tau', value: benchmark.tau, mono: true },
                  ...(showSources && benchmark.source ? [{ label: 'Source', value: benchmark.source, mono: true }] : []),
                ].filter((f) => f.value !== undefined && f.value !== null && f.value !== '')}
              />
            </Disclosure>
          </div>
        </Panel>
      </div>
    </Section>
  );
}
