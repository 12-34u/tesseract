import React from 'react';
import { fmtInt, fmtNum, MISSING } from '../format';
import {
  badgeClass,
  d5Summary,
  gateBadgeClass,
  isAvailable,
  isError,
  p1bMatrix,
  pendingMessage,
  readiness,
  roleRows,
  stageBadgeClass,
} from '../phase2/derive';

/**
 * Phase 2 status and pre-registration.
 *
 * This is not the Phase 2 results dashboard. Phase 2 has a pre-registered
 * design but no experimental results yet, so this view shows the design as
 * planned configuration and shows an explicit pending state everywhere an
 * artifact does not exist. No experimental value is hardcoded here: every
 * number is read from the /api/phase2/status payload.
 */

function Pending({ title, block, fallback }) {
  return (
    <div className="card" style={{ marginBottom: '1.25rem' }}>
      <div className="card-title">
        <span>{title}</span>
        <span className={`badge ${isError(block) ? 'badge-fail' : 'badge-warn'}`}>
          {isError(block) ? 'ERROR' : 'PENDING'}
        </span>
      </div>
      <p style={{ color: isError(block) ? 'var(--accent-rose)' : 'var(--text-muted)', margin: 0 }}>
        {pendingMessage(block, fallback)}
      </p>
    </div>
  );
}

function Field({ label, value, mono = true }) {
  return (
    <div>
      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{label}</span>
      <div className={mono ? 'mono' : ''} style={{ fontSize: '1rem', fontWeight: 600 }}>
        {value ?? MISSING}
      </div>
    </div>
  );
}

const list = (values) => (Array.isArray(values) && values.length ? values.join(', ') : MISSING);

function Source({ path }) {
  if (!path) return null;
  return (
    <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)' }}>
      {path}
    </span>
  );
}

// ---------------------------------------------------------------------------

function DecisionCards({ decisions }) {
  const cards = [
    ['Amendment 01', decisions?.amendment_01, (d) => d.state, (d) => d.summary],
    ['Amendment 02', decisions?.amendment_02, (d) => d.state, (d) => d.pending_p1b],
    ['Model width', decisions?.model_width, (d) => d.state, () => 'Chosen only after P1b completes and is approved'],
    ['Training budget', decisions?.training_budget, (d) => d.state, () => 'Fixed step budget; no early stopping'],
    ['P1b capacity diagnostic', decisions?.p1b, (d) => d.state, (d) => d.message || d.run_dir],
  ];
  return (
    <div className="grid grid-cols-3" style={{ marginBottom: '1.5rem' }}>
      {cards.map(([title, block, state, note]) => (
        <div className="card" key={title}>
          <div className="card-title">
            <span>{title}</span>
            <span className={`badge ${badgeClass(block?.tone)}`}>
              {block && block.status !== 'missing' ? state(block) : 'NOT FOUND'}
            </span>
          </div>
          <div className="card-subtext" style={{ marginTop: '0.5rem' }}>
            {block && block.status !== 'missing' ? note(block) || '' : pendingMessage(block)}
          </div>
          {block?.source ? (
            <div style={{ marginTop: '0.5rem' }}>
              <Source path={block.source} />
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function ModelFamiliesCard({ families, width }) {
  if (!isAvailable(families)) return <Pending title="Model families" block={families} />;
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>Model families</span>
        <span className={`badge ${families.selection_pending ? 'badge-warn' : 'badge-pass'}`}>
          {families.selection_pending ? 'SELECTION PENDING P1B' : `SELECTED: ${families.selected_base}`}
        </span>
      </div>

      <table className="matrix-table" style={{ marginTop: '0.75rem' }}>
        <thead>
          <tr>
            <th>Family</th>
            <th>Parameters</th>
            <th>d_model</th>
            <th>Heads</th>
            <th>Head dim</th>
            <th>d_ff</th>
            <th>Results</th>
          </tr>
        </thead>
        <tbody>
          {families.families.map((f) => (
            <tr key={f.name} style={f.selected ? { outline: '1px solid var(--accent-cyan)' } : undefined}>
              <td>
                {f.label}
                {f.selected ? <span className="badge badge-pass" style={{ marginLeft: '0.5rem' }}>SELECTED</span> : null}
              </td>
              <td className="mono">{fmtInt(f.parameter_count)}</td>
              <td className="mono">{f.d_model ?? MISSING}</td>
              <td className="mono">{f.num_heads ?? MISSING}</td>
              <td className="mono">{f.head_dim ?? MISSING}</td>
              <td className="mono">{f.d_ff ?? MISSING}</td>
              <td>
                <span className="badge badge-warn">Pending Results</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="card-subtext" style={{ marginTop: '0.75rem' }}>
        {families.note} Parameter counts are computed from each width&rsquo;s model config at vocab_size{' '}
        {families.vocab_size ?? MISSING}; they are architecture sizes, not results. The width used for Phase 2 is{' '}
        <strong>{width?.state ?? MISSING}</strong> and is chosen only after P1b.
      </div>
    </div>
  );
}

function GatesCard({ gates }) {
  if (!isAvailable(gates)) {
    return <Pending title="Validity gates G1 / G2" block={gates} fallback="Awaiting gate artifacts" />;
  }
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>Validity gates G1 / G2 — {gates.group}, n = {gates.n}</span>
        <span className={`badge ${gates.raw_verdict === 'PASS' ? 'badge-pass' : 'badge-warn'}`}>
          RAW VERDICT: {gates.raw_verdict}
        </span>
      </div>

      <table className="matrix-table" style={{ marginTop: '0.75rem' }}>
        <thead>
          <tr>
            <th>T</th>
            <th>G1</th>
            <th>G2 (raw)</th>
            <th>Max token agreement</th>
            <th>Closest shallow formula</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {gates.per_t.map((row) => (
            <tr key={row.t}>
              <td className="mono">T = {row.t}</td>
              <td>
                <span className={`badge ${row.g1_passed ? 'badge-pass' : 'badge-fail'}`}>
                  {row.g1_passed ? 'PASS' : 'FAIL'}
                </span>
              </td>
              <td>
                <span className={`badge ${gateBadgeClass(row)}`}>{row.g2_state}</span>
              </td>
              <td className="mono">{row.g2_applicable ? fmtNum(row.g2_max_token_agreement, 4) : MISSING}</td>
              <td className="mono">{row.g2_candidate || MISSING}</td>
              <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {row.g2_applicable
                  ? row.amendment_01_diagnostic
                    ? 'Amendment 01 diagnostic condition — raw result is FAIL and stays FAIL'
                    : ''
                  : 'T = 1 is the primitive itself'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="card-subtext" style={{ marginTop: '0.75rem' }}>
        The raw gate verdict is shown exactly as recorded. Amendment 01 annotates the T = 2 failure as a documented,
        analytically explained diagnostic condition; it does not turn it into a pass.
        {gates.amended ? (
          <>
            {' '}
            Amended verdict: <strong>{gates.amended.amended_verdict}</strong> (raw: {gates.amended.raw_gate_verdict}).
          </>
        ) : (
          ' Amended gate run not found.'
        )}
      </div>
    </div>
  );
}

function PlanCard({ plan }) {
  if (!isAvailable(plan)) return <Pending title="Experiment plan" block={plan} />;
  const ready = readiness(plan);
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>Experiment plan — {plan.total_planned_runs} planned runs</span>
        <span className={`badge ${badgeClass(ready.tone)}`}>{ready.state}</span>
      </div>

      {ready.reasons.length > 0 && (
        <div className="card-subtext" style={{ marginBottom: '0.75rem', color: 'var(--accent-amber)' }}>
          Blocked by: {ready.reasons.join('; ')}.
        </div>
      )}

      <table className="matrix-table">
        <thead>
          <tr>
            <th>Stage</th>
            <th>Family</th>
            <th>Task</th>
            <th>T</th>
            <th>Depth</th>
            <th>Seeds</th>
            <th>Planned</th>
            <th>Done</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {plan.stages.map((row) => (
            <tr key={`${row.stage}/${row.task}`}>
              <td className="mono">{row.label}</td>
              <td>{row.family_label}</td>
              <td>{row.task_label}</td>
              <td className="mono">{list(row.t_values)}</td>
              <td className="mono">
                {row.depth_symbol} = {list(row.depths)}
              </td>
              <td className="mono">{list(row.seeds)}</td>
              <td className="mono">{row.planned_runs}</td>
              <td className="mono">{row.completed_runs}</td>
              <td>
                <span className={`badge ${stageBadgeClass(row.state)}`}>{row.state}</span>
              </td>
            </tr>
          ))}
          <tr>
            <td className="mono" colSpan={6}>
              <strong>Total</strong>
            </td>
            <td className="mono">
              <strong>{plan.total_planned_runs}</strong>
            </td>
            <td className="mono">
              <strong>{plan.total_completed_runs}</strong>
            </td>
            <td />
          </tr>
        </tbody>
      </table>

      <div className="card-subtext" style={{ marginTop: '0.75rem' }}>
        This is the planned matrix, not completed results. Counts come from <Source path={plan.source} /> and
        completed runs are counted from <span className="mono">{plan.experiment_run_dir}</span>.
      </div>
    </div>
  );
}

function P1bCard({ p1b }) {
  if (!isAvailable(p1b)) {
    return <Pending title="P1b capacity / learnability diagnostic" block={p1b} fallback="P1b results not available yet." />;
  }
  const { widths, kValues, cell } = p1bMatrix(p1b);
  const d5 = d5Summary(p1b);
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>P1b capacity / learnability diagnostic</span>
        <span className={`badge ${badgeClass(p1b.tone)}`}>{p1b.state}</span>
      </div>

      <table className="matrix-table" style={{ marginTop: '0.75rem' }}>
        <thead>
          <tr>
            <th>Width</th>
            {kValues.map((k) => (
              <th key={k}>K = {k}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {widths.map((width) => (
            <tr key={width}>
              <td className="mono">{width}</td>
              {kValues.map((k) => {
                const c = cell(width, k);
                return (
                  <td key={k} className="mono" style={{ fontSize: '0.78rem' }}>
                    {c ? (
                      <>
                        final {fmtNum(c.final_val_chance_normalised, 4)}
                        <br />
                        loss {fmtNum(c.final_val_loss, 4)}
                        <br />
                        {fmtInt(c.parameter_count)} params
                        {c.at_floor_final ? (
                          <>
                            <br />
                            <span className="badge badge-warn">AT FLOOR</span>
                          </>
                        ) : null}
                      </>
                    ) : (
                      MISSING
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {d5 ? (
        <div style={{ marginTop: '1rem' }}>
          <div className="card-title">
            <span>Amendment 02 D5 decision</span>
            <span className={`badge ${d5.widthOutcome === 'SELECT' ? 'badge-pass' : 'badge-warn'}`}>
              {d5.widthOutcome ?? MISSING}
            </span>
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))',
              gap: '1rem',
              marginTop: '0.5rem',
            }}
          >
            <Field label="Recommended width" value={d5.selectedWidth} />
            <Field
              label="Recommended max_steps"
              value={
                d5.recommendedMaxSteps === null
                  ? MISSING
                  : `${fmtInt(d5.recommendedMaxSteps)}${d5.notConverged ? ' (not converged)' : ''}`
              }
            />
            <Field
              label={`K = ${d5.kLow ?? MISSING} at floor`}
              value={d5.kLowAtFloor === null ? 'unknown' : String(d5.kLowAtFloor)}
            />
            <Field label="Per-K budgets agree" value={d5.budgetsAgree === null ? MISSING : String(d5.budgetsAgree)} />
          </div>
        </div>
      ) : p1b.d5?.status === 'error' ? (
        <div className="card-subtext" style={{ marginTop: '0.75rem', color: 'var(--accent-rose)' }}>
          D5 could not be evaluated: {p1b.d5.message}
        </div>
      ) : null}

      <div className="card-subtext" style={{ marginTop: '0.75rem', color: 'var(--accent-amber)' }}>
        ⚠ {p1b.single_seed_note || 'P1b uses a single seed per cell.'} The D5 decision is advisory: the width and budget
        are approved by the researcher before Amendment 02 is frozen.
      </div>
    </div>
  );
}

function ResultsCard({ results }) {
  if (!isAvailable(results)) {
    return (
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title">
          <span>Phase 2 experiment results</span>
          <span className={`badge ${isError(results) ? 'badge-fail' : 'badge-warn'}`}>
            {isError(results) ? 'ERROR' : 'NO RESULTS YET'}
          </span>
        </div>
        <p style={{ color: 'var(--text-muted)' }}>{pendingMessage(results)}</p>
        <div className="card-subtext">
          {results?.total_completed_runs ?? 0} of {results?.total_planned_runs ?? MISSING} planned runs complete. When
          the matrix finishes, the pre-registered analysis at{' '}
          <span className="mono">{results?.analysis_path || 'runs/phase2/experiment/analysis/analysis.json'}</span> will
          populate this section with: {(results?.planned_metrics || []).join(', ') || MISSING}.
        </div>
      </div>
    );
  }
  const analysis = results.analysis || {};
  const best = analysis.checkpoints?.best?.decision_rule;
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-title">
        <span>Phase 2 experiment results</span>
        <span className="badge badge-pass">ANALYSIS AVAILABLE</span>
      </div>
      <div className="card-subtext">
        {results.total_completed_runs} of {results.total_planned_runs} planned runs complete ·{' '}
        <Source path={results.analysis_path} />
      </div>
      {best ? (
        <div className="card-subtext" style={{ marginTop: '0.5rem' }}>
          Decision rule outcome (best-validation checkpoint): <strong>{best.outcome}</strong>
        </div>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------

export function Phase2Section({ status, error }) {
  if (error) {
    return (
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title">
          <span>Phase 2</span>
          <span className="badge badge-fail">ERROR</span>
        </div>
        <p style={{ color: 'var(--accent-rose)' }}>{error}</p>
      </div>
    );
  }
  if (!status) {
    return (
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title">
          <span>Phase 2</span>
          <span className="badge badge-warn">LOADING</span>
        </div>
      </div>
    );
  }

  const {
    benchmark,
    t_roles: roles,
    protocol,
    model_families: modelFamilies,
    decisions,
    gates,
    plan,
    p1b,
    results,
    provenance,
  } = status;
  const ready = readiness(plan);

  return (
    <div className="section">
      <div className="section-header">
        <h2 className="section-title">Phase 2 — Status &amp; Pre-registration</h2>
        <span className={`badge ${badgeClass(ready.tone)}`}>CURRENT RESEARCH · {ready.state}</span>
      </div>

      <div className="card" style={{ marginBottom: '1.5rem', borderColor: 'var(--accent-cyan)' }}>
        <div className="card-title">
          <span>{status.project}</span>
          <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {provenance?.git?.commit ? provenance.git.commit.slice(0, 10) : 'unknown commit'}
            {provenance?.git?.dirty ? ' (uncommitted changes)' : ''}
          </span>
        </div>
        <p style={{ color: 'var(--text-secondary)', margin: '0.5rem 0 0' }}>
          Pre-registered design and current decision state. This is not the Phase 2 results dashboard: no experiment has
          run yet, and every panel below shows either checked-in configuration or an explicit pending state.
        </p>
      </div>

      {/* Benchmark */}
      {isAvailable(benchmark) ? (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-title">
            <span>Benchmark — {benchmark.group_description}</span>
            <Source path={benchmark.source} />
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
              gap: '1rem',
              marginTop: '0.5rem',
            }}
          >
            <Field label="Group" value={benchmark.group} />
            <Field label="Ring size" value={`n = ${benchmark.n ?? MISSING}`} />
            <Field label="Transition" value={benchmark.transition} />
            <Field label="T values" value={list(benchmark.t_values)} />
            <Field label="K values" value={list(benchmark.k_values)} />
            <Field label="Data seed" value={benchmark.data_seed ?? MISSING} />
            <Field label="Validation" value={fmtInt(benchmark.val_size)} />
            <Field label="Test" value={fmtInt(benchmark.test_size)} />
            <Field label="Tasks" value={list(benchmark.tasks)} />
            <Field label="τ (K* threshold)" value={benchmark.tau} />
          </div>
        </div>
      ) : (
        <Pending title="Benchmark" block={benchmark} />
      )}

      {/* T roles */}
      {isAvailable(roles) ? (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-title">
            <span>T roles (Amendment 01)</span>
            <Source path={roles.source} />
          </div>
          <table className="matrix-table" style={{ marginTop: '0.5rem' }}>
            <thead>
              <tr>
                <th>T</th>
                <th>Role</th>
                <th>Meaning</th>
              </tr>
            </thead>
            <tbody>
              {roleRows(roles).map((row) => (
                <tr key={row.t}>
                  <td className="mono">T = {row.t}</td>
                  <td>
                    <span className={`badge ${row.role === 'Primary inference' ? 'badge-pass' : 'badge-warn'}`}>
                      {row.role}
                    </span>
                  </td>
                  <td style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Pending title="T roles" block={roles} />
      )}

      {/* Protocol */}
      {isAvailable(protocol) ? (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-title">
            <span>Training protocol</span>
            <Source path={protocol.source} />
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
              gap: '1rem',
              marginTop: '0.5rem',
            }}
          >
            <Field label="Model seeds" value={list(protocol.model_seeds)} />
            <Field label="Optimizer" value={protocol.optimizer} />
            <Field label="Learning rate" value={protocol.learning_rate} />
            <Field label="Weight decay" value={protocol.weight_decay} />
            <Field label="Batch size" value={protocol.batch_size} />
            <Field label="Validation every" value={`${protocol.eval_every ?? MISSING} steps`} />
            <Field label="Early stopping" value={protocol.early_stopping ? 'yes' : 'none'} />
            <Field label="Primary checkpoint" value={protocol.primary_checkpoint} />
            <Field label="Sensitivity" value={protocol.sensitivity_checkpoint} />
            <Field label="Test policy" value={protocol.test_policy} />
          </div>
        </div>
      ) : (
        <Pending title="Training protocol" block={protocol} />
      )}

      <ModelFamiliesCard families={modelFamilies} width={decisions?.model_width} />

      <h3 style={{ fontSize: '0.95rem', margin: '0 0 0.75rem', color: 'var(--text-secondary)' }}>
        Current decision state
      </h3>
      <DecisionCards decisions={decisions} />

      <GatesCard gates={gates} />
      <PlanCard plan={plan} />
      <P1bCard p1b={p1b} />
      <ResultsCard results={results} />
    </div>
  );
}
