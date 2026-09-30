/**
 * 09 — Probes & diagnostic batches.
 *
 * The grid has not run. What has run is two P1c probes, Batch 01's arms, and
 * Batch 02's T=3 probe, and until now none of it appeared in the console at
 * all — the page went from the planned matrix straight to a pending results
 * section, which made months of recorded negative results invisible.
 *
 * Three presentation rules, each of them protecting a claim the research made
 * carefully:
 *
 * - Original and amended classifications are always shown side by side, never
 *   one in place of the other (Addendum 03 §3.3).
 * - An accuracy at T=2 or T=3 is never shown without the shortcut ceiling that
 *   belongs beside it. G2 fails at both depths at 26.6 %, above the 6.67 %
 *   LEARNABLE bar, so a bare accuracy there is the exact misreading the
 *   pre-registration was written to prevent.
 * - The K comparison is labelled descriptive. Six curriculum cells cannot
 *   separate a K effect from seed variance, and the classification governs.
 *
 * Everything derived lives in src/phase2/diagnostics.js.
 */

import React from 'react';
import {
  Disclosure,
  KeyValues,
  Panel,
  PendingNote,
  RuleHeading,
  Section,
  State,
  Tag,
} from './primitives.jsx';
import {
  arm5Summary,
  armRows,
  batch02State,
  cellRows,
  diagnosticsState,
  kComparison,
  probeRows,
  recordRows,
  stageBoundaryRows,
} from '../../phase2/diagnostics.js';

const pct = (value, digits = 2) => (Number.isFinite(value) ? `${value.toFixed(digits)} %` : '—');
const num = (value, digits = 4) => (Number.isFinite(value) ? value.toFixed(digits) : '—');
const step = (value) => (Number.isFinite(value) ? Math.round(value).toLocaleString() : 'never');
const label = (value) => (value ? String(value).replace(/_/g, ' ') : '—');

/** Original -> amended, as one cell. The arrow is the point of the column. */
function Verdicts({ row }) {
  return (
    <span style={{ display: 'inline-flex', gap: '0.4rem', alignItems: 'center', flexWrap: 'wrap' }}>
      <State tone={row.classification_tone}>{label(row.classification)}</State>
      {row.classification_amended && row.classification_amended !== row.classification ? (
        <>
          <span className="rc-mono" style={{ opacity: 0.5 }}>&rarr;</span>
          <State tone={row.classification_amended_tone}>{label(row.classification_amended)}</State>
        </>
      ) : null}
    </span>
  );
}

function ProbeTable({ rows }) {
  return (
    <table className="rc-table">
      <thead>
        <tr>
          <th style={{ width: '9rem' }}>Probe</th>
          <th style={{ width: '4rem' }}>T</th>
          <th style={{ width: '4rem' }}>K</th>
          <th>Verdict (pre-registered &rarr; amended)</th>
          <th style={{ width: '9rem' }}>Above chance</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((probe) =>
          probe.available ? (
            probe.byK.map((row, i) => (
              <tr key={`${probe.id}-${row.k}`} className={i ? 'rc-row-sub' : undefined}>
                <td className="rc-mono">
                  {i === 0 ? probe.label : ''}
                  {i === 0 && probe.warmStart ? <Tag tone="note">warm start</Tag> : null}
                </td>
                <td className="rc-mono">{i === 0 ? probe.t ?? '—' : ''}</td>
                <td className="rc-mono">{row.k}</td>
                <td><Verdicts row={row} /></td>
                <td className="rc-mono">
                  {(row.seeds_above_chance_amended ?? row.seeds_above_chance ?? []).length} / {row.n_seeds ?? '—'}
                </td>
              </tr>
            ))
          ) : (
            <tr key={probe.id}>
              <td className="rc-mono">{probe.label}</td>
              <td colSpan={4}>
                <State tone="idle">{probe.message || 'not run'}</State>
              </td>
            </tr>
          ),
        )}
      </tbody>
    </table>
  );
}

function ArmTable({ rows }) {
  return (
    <table className="rc-table">
      <thead>
        <tr>
          <th style={{ width: '6rem' }}>Arm</th>
          <th>Lever changed against the reference</th>
          <th style={{ width: '4rem' }}>K</th>
          <th>Verdict (pre-registered &rarr; amended)</th>
          <th style={{ width: '6rem' }}>Moves</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((arm) => {
          if (!arm.available) {
            return (
              <tr key={arm.id}>
                <td className="rc-mono">{arm.label}</td>
                <td>{arm.lever}</td>
                <td colSpan={3}>
                  <State tone="idle">{arm.message || 'not run'}</State>
                </td>
              </tr>
            );
          }
          if (!arm.classified) {
            // A gate or a measurement. Shown rather than dropped, so that
            // "four arms were classified" can be checked against the list.
            return (
              <tr key={arm.id}>
                <td className="rc-mono">{arm.label}</td>
                <td>
                  {arm.lever} <Tag tone="idle">{arm.isGate ? 'gate' : 'measurement'}</Tag>
                </td>
                <td className="rc-mono">—</td>
                <td>
                  {arm.isGate ? (
                    <State tone={arm.gatePassed ? 'ok' : 'fail'}>{arm.gatePassed ? 'GATE PASS' : 'GATE FAIL'}</State>
                  ) : (
                    <State tone="idle">not classified</State>
                  )}
                </td>
                <td className="rc-mono">n/a</td>
              </tr>
            );
          }
          return arm.byK.map((row, i) => (
            <tr key={`${arm.id}-${row.k}`} className={i ? 'rc-row-sub' : undefined}>
              <td className="rc-mono">{i === 0 ? arm.label : ''}</td>
              <td>{i === 0 ? arm.lever : ''}</td>
              <td className="rc-mono">{row.k}</td>
              <td><Verdicts row={row} /></td>
              <td>
                {i === 0 ? <State tone={arm.moved ? 'ok' : 'idle'}>{arm.moved ? 'yes' : 'no'}</State> : null}
              </td>
            </tr>
          ));
        })}
      </tbody>
    </table>
  );
}

function ShortcutContext({ context, reported }) {
  if (!context?.shortcutCandidate) return null;
  return (
    <div className="rc-decision" style={{ marginTop: '1.15rem' }}>
      <strong style={{ color: 'var(--rc-text)' }}>
        G2 fails at T = {context.t}, so accuracy alone cannot read this probe.
      </strong>
      <br />
      The shallow candidate <span className="rc-mono">{context.shortcutCandidate}</span> agrees with
      F<sup>{context.t}</sup> on {pct((context.shortcutAgreement ?? 0) * 100)} of tokens — exactly where{' '}
      <span className="rc-mono">{context.shortcutCondition}</span>
      {context.shortcutExact ? ', token for token' : ''}. The LEARNABLE bar is {pct(context.learnableBar)}, so a model
      that learned only the shortcut would clear it. Every accuracy is therefore{reported ? ' shown' : ' recorded'}{' '}
      with its agreement against that candidate: a model computing F<sup>{context.t}</sup> exactly agrees with it on
      ~26.7 %, and a shortcut-only model on ~100 %.
    </div>
  );
}

function Batch02Body({ state }) {
  const cells = cellRows(state);
  const comparison = kComparison(state);
  return (
    <>
      <table className="rc-table">
        <thead>
          <tr>
            <th style={{ width: '4rem' }}>K</th>
            <th>Verdict (pre-registered &rarr; amended)</th>
            <th style={{ width: '11rem' }}>Shortcut attribution</th>
            <th style={{ width: '9rem' }}>Seeds at bar</th>
          </tr>
        </thead>
        <tbody>
          {state.byK.map((row) => (
            <tr key={row.k}>
              <td className="rc-mono">{row.k}</td>
              <td><Verdicts row={row} /></td>
              <td>
                {row.attribution ? (
                  <State tone={row.attribution_tone}>{label(row.attribution)}</State>
                ) : (
                  <State tone="idle">n/a</State>
                )}
              </td>
              <td className="rc-mono">
                {(row.seeds_meeting_learnable_bar ?? []).length} / {row.n_seeds ?? '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <RuleHeading>Per cell</RuleHeading>
      <table className="rc-table">
        <thead>
          <tr>
            <th style={{ width: '3.5rem' }}>K</th>
            <th style={{ width: '4rem' }}>Seed</th>
            <th style={{ width: '7rem' }}>Best acc.</th>
            <th style={{ width: '8rem' }}>Shortcut agr.</th>
            <th style={{ width: '8rem' }}>Past bar at</th>
            <th style={{ width: '9rem' }}>Past ceiling at</th>
            <th style={{ width: '8rem' }}>Slope /1k</th>
            <th style={{ width: '6rem' }}>Tokens</th>
          </tr>
        </thead>
        <tbody>
          {cells.map((cell) => (
            <tr key={`${cell.k}-${cell.seed}`}>
              <td className="rc-mono">{cell.k}</td>
              <td className="rc-mono">{cell.seed}</td>
              <td className="rc-mono">{pct(cell.bestAccuracy)}</td>
              <td className="rc-mono">{pct(cell.shortcutAgreement)}</td>
              <td className="rc-mono">{step(cell.crossedBarAt)}</td>
              <td className="rc-mono">{step(cell.crossedCeilingAt)}</td>
              <td className="rc-mono">
                {Number.isFinite(cell.finalSlope) ? `${cell.finalSlope >= 0 ? '+' : ''}${cell.finalSlope.toFixed(3)}` : '—'}
                {cell.stillRising ? <Tag tone="note">rising</Tag> : null}
              </td>
              <td className="rc-mono">{cell.distinctTokens ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {comparison.length > 1 && (
        <>
          <RuleHeading>K comparison — descriptive only</RuleHeading>
          <table className="rc-table">
            <thead>
              <tr>
                <th style={{ width: '3.5rem' }}>K</th>
                <th style={{ width: '6rem' }}>Cells</th>
                <th>Mean best acc.</th>
                <th>Mean shortcut agr.</th>
                <th>Mean takeoff step</th>
                <th>Mean slope /1k</th>
                <th style={{ width: '7rem' }}>Rising</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr key={row.k}>
                  <td className="rc-mono">{row.k}</td>
                  <td className="rc-mono">{row.cells}</td>
                  <td className="rc-mono">{pct(row.meanBestAccuracy)}</td>
                  <td className="rc-mono">{pct(row.meanShortcutAgreement)}</td>
                  <td className="rc-mono">{step(row.meanTakeoffStep)}</td>
                  <td className="rc-mono">
                    {Number.isFinite(row.meanFinalSlope)
                      ? `${row.meanFinalSlope >= 0 ? '+' : ''}${row.meanFinalSlope.toFixed(3)}`
                      : '—'}
                  </td>
                  <td className="rc-mono">
                    {row.stillRising} / {row.cells}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="rc-prose" style={{ marginTop: '0.6rem' }}>
            Descriptive. This is a curriculum diagnostic, not a grid cell: K = 1 and K = 8 run the identical
            schedule, six cells cannot separate a K effect from seed variance, and the pre-registered
            classification above is what governs. No number here is evidence for the depth hypothesis.
          </p>
        </>
      )}

      <div style={{ marginTop: '1rem' }}>
        <Disclosure
          label="Stage-boundary validation"
          hint="the Batch 01 deviation, recorded per boundary"
        >
          <p className="rc-prose" style={{ marginTop: 0 }}>
            Batch 01 was specified to record validation accuracy at each depth at the end of each curriculum
            stage and instead measured the final model once, so its stage evidence survived only as console
            output. These are the recorded fields that replace it.
          </p>
          {state.cells.map((cell) => {
            const rows = stageBoundaryRows(cell);
            if (!rows.length) return null;
            return (
              <div key={`${cell.k}-${cell.seed}`} style={{ marginTop: '0.9rem' }}>
                <RuleHeading>
                  K = {cell.k}, seed {cell.seed}
                </RuleHeading>
                <table className="rc-table">
                  <thead>
                    <tr>
                      <th style={{ width: '6rem' }}>Step</th>
                      <th style={{ width: '7rem' }}>Ends stage</th>
                      <th style={{ width: '4rem' }}>T</th>
                      <th>Accuracy</th>
                      <th>Loss</th>
                      <th>Shortcut agr.</th>
                      <th style={{ width: '6rem' }}>Tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, i) => (
                      <tr key={`${row.step}-${row.t}`} className={row.isStageDepth ? 'rc-row-accent' : undefined}>
                        <td className="rc-mono">{i % 3 === 0 ? row.step.toLocaleString() : ''}</td>
                        <td className="rc-mono">{i % 3 === 0 ? `T = ${row.stageT}` : ''}</td>
                        <td className="rc-mono">{row.t}</td>
                        <td className="rc-mono">{pct(row.tokenAccuracy, 4)}</td>
                        <td className="rc-mono">{num(row.loss)}</td>
                        <td className="rc-mono">
                          {row.shortcutAgreement == null ? 'n/a' : pct(row.shortcutAgreement)}
                        </td>
                        <td className="rc-mono">{row.distinctTokens ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          })}
        </Disclosure>
      </div>
    </>
  );
}

function RecordTable({ rows, showInternals }) {
  if (!rows.length) return null;
  return (
    <Disclosure label="Pre-registration and correction record" hint={`${rows.length} documents`}>
      <p className="rc-prose" style={{ marginTop: 0 }}>
        The claim these results rest on is that every decision rule was committed before the results it
        governs, and that corrections were recorded rather than edited into history. That is only checkable
        if the documents are listed.
      </p>
      <table className="rc-table">
        <thead>
          <tr>
            <th style={{ width: '7rem' }}>Kind</th>
            <th>Document</th>
            <th style={{ width: '6rem' }}>Present</th>
            {showInternals && <th style={{ width: '10rem' }}>sha256</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((doc) => (
            <tr key={doc.name}>
              <td className="rc-mono">{doc.kind}</td>
              <td className="rc-mono" style={{ fontSize: '0.75rem' }} title={doc.label}>
                {doc.name}
              </td>
              <td>
                <State tone={doc.present ? 'ok' : 'idle'}>{doc.present ? 'yes' : 'no'}</State>
              </td>
              {showInternals && <td className="rc-mono" style={{ fontSize: '0.7rem' }}>{doc.shortSha || '—'}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </Disclosure>
  );
}

export function DiagnosticsPanel({ status, showInternals = true }) {
  const state = diagnosticsState(status);
  if (!state.available) {
    return (
      <Section id="probes" index="09" title="Probes & diagnostic batches">
        <Panel>
          <PendingNote>{state.message}</PendingNote>
        </Panel>
      </Section>
    );
  }

  const probes = probeRows(status);
  const arms = armRows(status);
  const arm5 = arm5Summary(status);
  const batch02 = batch02State(status);

  return (
    <Section
      id="probes"
      index="09"
      title="Probes & diagnostic batches"
      note="What has actually run. Diagnostics against a single reference, not grid cells."
    >
      {probes.length > 0 && (
        <Panel title="P1c probes" subtitle="Cold start at T = 4, and warm-started at T = 8.">
          <ProbeTable rows={probes} />
        </Panel>
      )}

      {arms.length > 0 && (
        <Panel
          title="Batch 01 — one lever each against a common reference"
          subtitle="Every arm pre-registered before it ran. Verdicts shown pre-registered and amended."
          style={{ marginTop: '1rem' }}
        >
          <ArmTable rows={arms} />
          {arm5 && (
            <div className="rc-decision" style={{ marginTop: '1.15rem' }}>
              <strong style={{ color: 'var(--rc-text)' }}>
                {arm5.approved ? 'Arm 5 approved.' : 'Arm 5 not approved, and not run.'}
              </strong>
              <br />
              {arm5.sentence}
              {arm5.stage3Row ? (
                <>
                  {' '}The Stage 3 row that applies is <span className="rc-mono">{arm5.stage3Row}</span>, whose
                  action is to report and to draft — not finalise — the negative-result write-up.
                </>
              ) : null}
            </div>
          )}
        </Panel>
      )}

      <Panel
        title={`Batch 02 — T = ${batch02.context?.t ?? 3} learnability probe`}
        subtitle={batch02.context?.question}
        aside={
          batch02.available ? (
            batch02.smoke ? <Tag tone="warn">smoke run</Tag> : <State tone="ok">reported</State>
          ) : (
            <State tone="active">running or not started</State>
          )
        }
        style={{ marginTop: '1rem' }}
      >
        {batch02.available ? (
          <Batch02Body state={batch02} />
        ) : (
          <PendingNote>{batch02.message}</PendingNote>
        )}
        <ShortcutContext context={batch02.context} reported={batch02.available} />
        {showInternals && batch02.context?.preregistration && (
          <div style={{ marginTop: '1rem' }}>
            <KeyValues
              fields={[
                { label: 'Pre-registration', value: batch02.context.preregistration, mono: true },
                { label: 'Addendum', value: batch02.context.addendum, mono: true },
                {
                  label: 'T = 3 majority-class floor',
                  value: batch02.available ? pct(batch02.majorityClass, 4) : '—',
                  mono: true,
                },
                {
                  label: 'Report',
                  value: batch02.available ? batch02.reportPath : 'not written',
                  mono: true,
                },
              ]}
            />
          </div>
        )}
      </Panel>

      <div style={{ marginTop: '1rem' }}>
        <Panel quiet>
          <RecordTable rows={recordRows(status)} showInternals={showInternals} />
        </Panel>
      </div>
    </Section>
  );
}
