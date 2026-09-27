/**
 * 05 — Gate review.
 *
 * A research-gate table rather than a set of coloured cards. The T=2 G2 result
 * is a recorded research decision, shown in muted amber as a DIAGNOSTIC
 * exception under Amendment 01.
 *
 * What it is NOT is relabelled: the backend's own wording for that row is
 * "FAIL", the raw gate verdict stays visible in the specification disclosure,
 * and the amendment annotates the failure rather than converting it to a pass.
 */

import React from 'react';
import { Disclosure, KeyValues, Panel, PendingNote, RuleHeading, Section, State } from './primitives.jsx';
import { gateGroups, primaryDepthSentence } from '../../phase2/pipeline.js';

function g2State(row) {
  if (row.applicable === false) return { tone: 'idle', label: 'N/A' };
  if (row.passed === true) return { tone: 'ok', label: 'PASS' };
  if (row.diagnostic) return { tone: 'note', label: 'DIAGNOSTIC' };
  return { tone: 'fail', label: row.state || 'FAIL' };
}

export function GatesPanel({ status, showInternals = true }) {
  const groups = gateGroups(status);
  if (!groups.available) {
    return (
      <Section id="gates" index="05" title="Gate review">
        <Panel>
          <PendingNote>{groups.message}</PendingNote>
        </Panel>
      </Section>
    );
  }

  const { g1, g2 } = groups;
  const sentence = primaryDepthSentence(status);

  return (
    <Section id="gates" index="05" title="Gate review" note="Pre-training validation of the benchmark itself.">
      <Panel>
        <table className="rc-table">
          <thead>
            <tr>
              <th style={{ width: '6rem' }}>Gate</th>
              <th>Condition</th>
              <th style={{ width: '12rem' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="rc-mono">G1</td>
              <td>Output uniformity, cycle detection, sensitivity</td>
              <td>
                <State tone={g1.passed ? 'ok' : 'fail'}>{g1.passed ? 'PASS' : 'FAIL'}</State>
              </td>
            </tr>
            <tr>
              <td className="rc-mono">G2</td>
              <td>Shallow-collapse test</td>
              <td>
                <State tone="idle">REVIEWED</State>
              </td>
            </tr>
            {g2.map((row) => {
              const s = g2State(row);
              return (
                <tr className="rc-row-sub" key={row.t}>
                  <td className="rc-mono">T = {row.t}</td>
                  <td className="rc-mono" style={{ fontSize: '0.75rem' }}>
                    {row.candidate || '—'}
                  </td>
                  <td>
                    <State tone={s.tone}>{s.label}</State>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {sentence && (
          <div className="rc-decision" style={{ marginTop: '1.15rem' }}>
            <strong style={{ color: 'var(--rc-text)' }}>{sentence}</strong>
            <br />
            T = 2 remains a reported diagnostic condition with an explicit shortcut baseline under Amendment 01, and
            is excluded from the primary inference.
          </div>
        )}

        {showInternals && (
          <div style={{ marginTop: '1rem' }}>
            <Disclosure label="Gate specification" hint="thresholds, candidates, raw verdict">
              <RuleHeading>Measured</RuleHeading>
              <KeyValues
                fields={[
                  { label: 'Raw gate verdict', value: groups.rawVerdict ?? '—', mono: true },
                  {
                    label: 'Max TV distance',
                    value: Number.isFinite(g1.maxTv) ? g1.maxTv.toFixed(4) : '—',
                    mono: true,
                  },
                  {
                    label: 'Min sensitivity',
                    value: Number.isFinite(g1.minSensitivity) ? g1.minSensitivity.toFixed(3) : '—',
                    mono: true,
                  },
                  { label: 'Cycle detection', value: g1.cyclesPassed === true ? 'pass' : '—' },
                  { label: 'Gate self-test', value: g1.selfTestPassed === true ? 'pass' : '—' },
                ]}
              />
              <RuleHeading>Shallow candidates</RuleHeading>
              <table className="rc-table">
                <thead>
                  <tr>
                    <th>T</th>
                    <th>Candidate</th>
                    <th>Token agreement</th>
                  </tr>
                </thead>
                <tbody>
                  {g2.map((row) => (
                    <tr key={row.t}>
                      <td className="rc-mono">T = {row.t}</td>
                      <td className="rc-mono">{row.candidate || '—'}</td>
                      <td className="rc-mono">{row.agreement != null ? row.agreement.toFixed(4) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Disclosure>
          </div>
        )}
      </Panel>
    </Section>
  );
}
