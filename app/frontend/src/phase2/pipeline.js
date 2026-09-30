/**
 * Derived display state for the Phase 2 research console.
 *
 * Every function here is a pure mapping from a backend payload onto view state.
 * The rules that matter:
 *
 *   * No function invents an experimental value. When the backend reports a
 *     block as missing or pending, these return the pending state.
 *   * No function decides anything the research record decides. Width
 *     selection, budget and gate verdicts are read from artifacts; they are
 *     never recomputed here. `derive.js` owns the D5 reading and is used as-is.
 *   * Counts (planned runs, seeds, K and T grids) come from the payload, never
 *     from a literal in this file.
 *
 * Kept separate from the components so the mapping is unit testable without a
 * DOM, which is where the interesting failure modes live.
 */

import { isAvailable, isPending, widthSelection } from './derive.js';

/** Stage states, ordered from "nothing yet" to "done". */
export const STAGE_DONE = 'DONE';
export const STAGE_ACTIVE = 'ACTIVE';
export const STAGE_WAITING = 'WAITING';
export const STAGE_LOCKED = 'LOCKED';
export const STAGE_ATTENTION = 'ATTENTION';

/** A pending stage is never an error, so it never gets the failure tone. */
export function stageTone(state) {
  if (state === STAGE_DONE) return 'ok';
  if (state === STAGE_ACTIVE) return 'active';
  if (state === STAGE_ATTENTION) return 'note';
  return 'idle';
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/**
 * The overall phase badge.
 *
 * "Pre-experiment" and "diagnostic" are distinguished by whether any capacity
 * diagnostic has actually produced a report, so the badge tracks the record
 * rather than the calendar.
 */
export function overallState(status) {
  const plan = status?.plan;
  const completed = plan?.total_completed_runs ?? 0;
  const planned = plan?.total_planned_runs ?? 0;

  if (completed > 0 && planned > 0 && completed >= planned) {
    return { label: 'EXPERIMENT COMPLETE', tone: 'ok', detail: 'Every planned run is recorded.' };
  }
  if (completed > 0) {
    return {
      label: 'EXPERIMENT RUNNING',
      tone: 'active',
      detail: `${completed} of ${planned} planned runs recorded.`,
    };
  }

  const capacity = status?.capacity_diagnostics;
  const anyDiagnostic = [capacity?.p1b, capacity?.p1b2].some((block) => block?.state === 'COMPLETE');
  if (anyDiagnostic) {
    return {
      label: 'DIAGNOSTIC PHASE',
      tone: 'active',
      detail: 'Capacity diagnostics in progress; the primary matrix has not started.',
    };
  }
  return {
    label: 'PRE-EXPERIMENT',
    tone: 'idle',
    detail: 'Pre-registration and diagnostics only. No primary run has started.',
  };
}

/**
 * The metadata chips under the title.
 *
 * Read from the benchmark and model-family payloads. A value the backend has
 * not supplied is omitted rather than guessed, so the row can be short.
 */
export function metadataChips(status) {
  const benchmark = status?.benchmark;
  const families = status?.model_families;
  const chips = [];
  if (isAvailable(benchmark)) {
    if (benchmark.group_description || benchmark.group) {
      chips.push({ label: 'Benchmark', value: benchmark.group_description || `${benchmark.group} CA` });
    }
    if (benchmark.n != null) chips.push({ label: 'n', value: String(benchmark.n), mono: true });
    if (Array.isArray(benchmark.k_values) && benchmark.k_values.length) {
      chips.push({ label: 'K', value: `{${benchmark.k_values.join(', ')}}`, mono: true });
    }
    if (Array.isArray(benchmark.t_values) && benchmark.t_values.length) {
      chips.push({ label: 'T', value: `{${benchmark.t_values.join(', ')}}`, mono: true });
    }
  }
  if (isAvailable(families)) {
    if (families.vocab_size != null) {
      chips.push({ label: 'Vocab', value: String(families.vocab_size), mono: true });
    }
    const constant = (families.families || []).every((f) => f.parameter_count_is_constant_across_k);
    if ((families.families || []).length && constant) {
      chips.push({ label: 'Shared block', value: '1 · P constant across K' });
    }
  }
  return chips;
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

function gateStage(gates) {
  if (!isAvailable(gates)) {
    return { state: STAGE_WAITING, summary: 'No gate report recorded yet.' };
  }
  const rows = gates.per_t || [];
  const g1 = rows.every((r) => r.g1_passed !== false);
  const diagnostics = rows.filter((r) => r.amendment_01_diagnostic);
  const undocumented = rows.filter((r) => r.g2_passed === false && !r.amendment_01_diagnostic);
  if (undocumented.length) {
    return {
      state: STAGE_ATTENTION,
      summary: `G2 fails at T=${undocumented.map((r) => r.t).join(', ')} with no recorded amendment.`,
    };
  }
  if (diagnostics.length) {
    return {
      state: STAGE_DONE,
      summary: `G1 pass · G2 diagnostic exception at T=${diagnostics.map((r) => r.t).join(', ')} (Amendment 01).`,
    };
  }
  return { state: STAGE_DONE, summary: g1 ? 'G1 and G2 pass at every T.' : 'G1 reported a failure.' };
}

function capacityStage(capacity) {
  const blocks = [capacity?.p1b, capacity?.p1b2].filter(Boolean);
  if (!blocks.length) return { state: STAGE_WAITING, summary: 'No capacity diagnostic configured.' };
  const done = blocks.filter((b) => b.state === 'COMPLETE');
  const labels = blocks
    .map((b) => `${(b.label || '').split('—')[0].trim() || 'diagnostic'}: ${b.state || b.status}`)
    .join(' · ');
  if (done.length === blocks.length) return { state: STAGE_DONE, summary: labels };
  if (done.length > 0) return { state: STAGE_ACTIVE, summary: labels };
  return { state: STAGE_WAITING, summary: labels };
}

/**
 * The seven research stages, each with its dependency and what it is waiting on.
 *
 * Dependencies are stated so a blocked stage reads as "waiting for X" rather
 * than as a fault.
 */
/**
 * The probes-and-batches rail stage.
 *
 * ATTENTION rather than DONE when arms have reported and none moved: the batch
 * completed successfully and its result is negative, which is a state the
 * reader should be sent to rather than one that reads as finished business.
 */
function diagnosticsStage(diagnostics) {
  if (!isAvailable(diagnostics)) {
    return { state: STAGE_WAITING, summary: 'No probe or batch artifacts.' };
  }
  const batch01 = diagnostics.batch01 || {};
  const batch02 = diagnostics.batch02 || {};
  const classified = batch01.classified_arms || 0;
  const moved = batch01.arms_that_moved;
  const batch02Reported = batch02.status === 'available' && !batch02.smoke;

  if (!classified) return { state: STAGE_ACTIVE, summary: 'Probes recorded; no arm classified yet.' };
  if (!batch02Reported) {
    return {
      state: STAGE_ACTIVE,
      summary: `${classified} Batch 01 arms classified, ${moved ?? 0} moved. Batch 02's T=3 probe has not reported.`,
    };
  }
  return {
    state: moved === 0 ? STAGE_ATTENTION : STAGE_DONE,
    summary: `${classified} Batch 01 arms classified, ${moved} moved. Batch 02 reported at T=3.`,
  };
}

export function pipelineStages(status) {
  const gates = status?.gates;
  const capacity = status?.capacity_diagnostics;
  const plan = status?.plan;
  const results = status?.results;
  const selection = widthSelection(capacity);
  const blocked = plan?.blocked_reasons || [];

  const benchmarkAvailable = isAvailable(status?.benchmark);
  const gate = gateStage(gates);
  const cap = capacityStage(capacity);
  const diagnostics = diagnosticsStage(status?.diagnostics);

  const widthChosen = Boolean(selection.selectedWidth);
  const budgetChosen = selection.recommendedMaxSteps != null;

  const primaryStages = (plan?.stages || []).filter((s) => s.stage === 't4' || s.stage === 't8');
  const primaryPlanned = primaryStages.reduce((sum, s) => sum + (s.planned_runs || 0), 0);
  const primaryCompleted = primaryStages.reduce((sum, s) => sum + (s.completed_runs || 0), 0);

  return [
    {
      id: 'benchmark',
      index: '01',
      title: 'Benchmark',
      state: benchmarkAvailable ? STAGE_DONE : STAGE_WAITING,
      summary: benchmarkAvailable
        ? 'Task definition recorded and independently verified.'
        : 'Benchmark configuration not found.',
      dependency: null,
    },
    {
      id: 'gates',
      index: '02',
      title: 'Gate Validation',
      state: gate.state,
      summary: gate.summary,
      dependency: 'Benchmark',
    },
    {
      id: 'capacity',
      index: '03',
      title: 'Capacity Diagnostic',
      state: cap.state,
      summary: cap.summary,
      dependency: 'Gate validation',
    },
    {
      id: 'width',
      index: '04',
      title: 'Research Width',
      state: widthChosen ? STAGE_DONE : STAGE_WAITING,
      summary: widthChosen
        ? `D5 selected ${selection.selectedWidth}.`
        : selection.awaiting?.length
          ? `Awaiting ${selection.awaiting.join(' + ')} before D5 can evaluate.`
          : selection.state || 'Width not selected.',
      dependency: 'Capacity diagnostics',
    },
    {
      id: 'budget',
      index: '05',
      title: 'Budget',
      state: budgetChosen ? STAGE_DONE : STAGE_WAITING,
      summary: budgetChosen
        ? `Recommended budget ${selection.recommendedMaxSteps.toLocaleString()} steps.`
        : 'Waiting for the D5 decision.',
      dependency: 'Research width (D5)',
    },
    {
      id: 'probes',
      index: '06',
      title: 'Probes & Batches',
      state: diagnostics.state,
      summary: diagnostics.summary,
      dependency: 'Gate validation',
    },
    {
      id: 'matrix',
      index: '07',
      title: 'Primary Matrix',
      state: primaryCompleted > 0 ? STAGE_ACTIVE : blocked.length ? STAGE_LOCKED : STAGE_WAITING,
      summary:
        primaryCompleted > 0
          ? `${primaryCompleted} of ${primaryPlanned} primary runs recorded.`
          : blocked.length
            ? `Locked: ${blocked.length} precondition${blocked.length === 1 ? '' : 's'} outstanding.`
            : `${primaryPlanned} primary runs ready to start.`,
      dependency: 'Width and budget decisions',
      blockers: blocked,
    },
    {
      id: 'analysis',
      index: '08',
      title: 'Analysis',
      state: isAvailable(results) ? STAGE_DONE : STAGE_LOCKED,
      summary: isAvailable(results) ? 'Analysis artifact recorded.' : 'Not started.',
      dependency: 'Primary matrix',
    },
  ];
}

// ---------------------------------------------------------------------------
// Benchmark and gates
// ---------------------------------------------------------------------------

/** T values with their Amendment 01 role, for the benchmark card. */
export function tRoleRows(status) {
  const roles = status?.t_roles;
  const benchmark = status?.benchmark;
  if (!isAvailable(roles)) return [];
  const tValues = isAvailable(benchmark) && Array.isArray(benchmark.t_values) ? benchmark.t_values : [];
  const label = (t) => {
    if ((roles.anchor || []).includes(t)) return { role: 'Anchor', tone: 'idle' };
    if ((roles.diagnostic || []).includes(t)) return { role: 'Diagnostic', tone: 'note' };
    if ((roles.primary_depth || []).includes(t)) return { role: 'Primary', tone: 'ok' };
    return { role: null, tone: 'idle' };
  };
  return tValues.map((t) => ({ t, ...label(t) })).filter((row) => row.role);
}

/**
 * G1 and G2 rows for the gate card.
 *
 * A documented diagnostic G2 failure keeps the word FAIL in `g2State` — the
 * backend's own wording — and carries `diagnostic: true` so the view can
 * present it as a recorded research decision without relabelling the result.
 */
export function gateGroups(status) {
  const gates = status?.gates;
  if (!isAvailable(gates)) {
    return { available: false, message: gates?.message || 'No gate report recorded yet.', g1: null, g2: [] };
  }
  const rows = gates.per_t || [];
  return {
    available: true,
    g1: {
      passed: rows.every((r) => r.g1_passed !== false),
      minSensitivity: rows.reduce(
        (min, r) => (r.g1_min_sensitivity != null ? Math.min(min, r.g1_min_sensitivity) : min),
        Infinity,
      ),
      maxTv: rows.reduce((max, r) => (r.g1_tv_distance != null ? Math.max(max, r.g1_tv_distance) : max), -Infinity),
      cyclesPassed: gates.cycles_passed,
      selfTestPassed: gates.self_test_passed,
    },
    g2: rows.map((r) => ({
      t: r.t,
      applicable: r.g2_applicable,
      state: r.g2_state,
      passed: r.g2_passed,
      diagnostic: Boolean(r.amendment_01_diagnostic),
      agreement: r.g2_max_token_agreement,
      candidate: r.g2_candidate,
    })),
    rawVerdict: gates.raw_verdict,
  };
}

/** The sentence stating which T values carry the depth claim. */
export function primaryDepthSentence(status) {
  const primary = status?.t_roles?.primary_depth;
  if (!Array.isArray(primary) || !primary.length) return null;
  return `Primary depth analysis uses T=${primary.join(' and T=')}.`;
}

// ---------------------------------------------------------------------------
// Capacity and the research-width gate
// ---------------------------------------------------------------------------

/**
 * One card per candidate width.
 *
 * `researchScale` marks the width whose diagnostic is P1b-2. That is read from
 * the payload — P1b-2 exists specifically for the research-scale width — and is
 * deliberately not a claim that the width has been selected or that it performs
 * better. `status` is the backend's own wording.
 */
export function widthCards(status) {
  const capacity = status?.capacity_diagnostics;
  const candidates = capacity?.candidate_widths;
  if (!Array.isArray(candidates) || !candidates.length) return [];
  const blockFor = (diagnostic) => {
    if (diagnostic === 'P1b') return capacity.p1b;
    if (diagnostic === 'P1b-2') return capacity.p1b2;
    return null;
  };
  const selection = widthSelection(capacity);
  return candidates.map((c) => {
    const block = blockFor(c.diagnostic);
    return {
      name: c.name,
      label: c.label,
      parameterCount: c.parameter_count,
      diagnostic: c.diagnostic,
      statusText: c.status,
      researchScale: c.diagnostic === 'P1b-2',
      diagnosticState: block?.state || block?.status || 'unknown',
      diagnosticComplete: block?.state === 'COMPLETE',
      selected: selection.selectedWidth === c.name,
    };
  });
}

/**
 * The D5 layer: the learnability test and its recorded criteria.
 *
 * Criteria wording comes from the amendment payload when present. The outcome
 * is whatever `widthSelection` read from the evaluator artifact; this adds no
 * verdict of its own.
 */
export function d5Layer(status) {
  const capacity = status?.capacity_diagnostics;
  const selection = widthSelection(capacity);
  const amendment = status?.decisions?.amendment_02;
  const d5 = (amendment?.decisions || []).find((d) => d.id === 'D5') || null;
  return {
    title: 'D5 — Learnability Test',
    question: 'Does the model learn T=8?',
    recordedTitle: d5?.title ?? null,
    amendmentFrozen: Boolean(amendment?.frozen),
    amendmentState: amendment?.state ?? null,
    evaluated: Boolean(selection.selectedWidth),
    state: selection.state,
    awaiting: selection.awaiting || [],
    selectedWidth: selection.selectedWidth,
    outcome: selection.outcome,
    recommendedMaxSteps: selection.recommendedMaxSteps,
    notConverged: selection.notConverged,
    kLow: selection.kLow,
    kLowAtFloor: selection.kLowAtFloor,
  };
}

/**
 * The D6 layer: the width policy.
 *
 * Two different things are kept apart here, because conflating them would make
 * the dashboard a source of truth:
 *
 *   `designatedWidth` — the research-scale width, read from the P1b-2
 *      configuration, which exists specifically for it. This is a
 *      pre-registered designation: the width the research runs at, fixed in
 *      advance. It is NOT a claim that the width won anything, and the view
 *      must not label it "best" or "selected because it performs better".
 *
 *   `recorded` — whether a formal D6 entry exists in Amendment 02. It does not:
 *      the amendment records D1 to D5. If a D6 decision is added later it is
 *      picked up here with no code change.
 *
 * The selection RULE stays D5's, and `recordedPolicy` carries the capacity
 * payload's own wording so the distinction remains visible.
 */
export function d6Layer(status) {
  const capacity = status?.capacity_diagnostics;
  const amendment = status?.decisions?.amendment_02;
  const recorded = (amendment?.decisions || []).find((d) => d.id === 'D6') || null;
  const designated = widthCards(status).find((w) => w.researchScale) || null;
  return {
    title: 'D6 — Width Policy',
    recorded: Boolean(recorded),
    recordedTitle: recorded?.title ?? null,
    designatedWidth: designated,
    designationSource: designated ? `${designated.diagnostic} configuration` : null,
    recordedPolicy: capacity?.note ?? null,
    note: recorded
      ? null
      : 'Designated in advance by the P1b-2 configuration. Amendment 02 records D1-D5, so no separate D6 '
        + 'entry exists; the D5 selection rule below is unchanged.',
  };
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

function matrixFromStages(status, stages) {
  const benchmark = status?.benchmark;
  const kValues = isAvailable(benchmark) && Array.isArray(benchmark.k_values) ? benchmark.k_values : [];
  const tValues = [...new Set(stages.flatMap((s) => s.t_values || []))].sort((a, b) => a - b);
  const plannedRuns = stages.reduce((sum, s) => sum + (s.planned_runs || 0), 0);
  const completedRuns = stages.reduce((sum, s) => sum + (s.completed_runs || 0), 0);
  const seeds = [...new Set(stages.flatMap((s) => s.seeds || []))];
  const locked = stages.some((s) => s.state === 'BLOCKED');
  return {
    kValues,
    tValues,
    seeds,
    plannedRuns,
    completedRuns,
    locked,
    stages: stages.map((s) => ({
      stage: s.stage,
      label: s.label,
      purpose: s.purpose,
      plannedRuns: s.planned_runs,
      completedRuns: s.completed_runs,
      state: s.state,
    })),
    /** Every cell is planned until a result artifact says otherwise. */
    cell: () => ({ state: completedRuns > 0 ? 'unknown' : 'planned', metrics: null }),
  };
}

/** The primary K x T matrix, built from the recorded plan rather than literals. */
export function primaryMatrix(status) {
  const plan = status?.plan;
  if (!isAvailable(plan)) return { available: false, kValues: [], tValues: [], plannedRuns: 0, stages: [] };
  const primaryT = status?.t_roles?.primary_depth || [];
  const stages = (plan.stages || []).filter((s) => (s.t_values || []).some((t) => primaryT.includes(t))
    && s.family === 'tesseract' && s.task === 'main');
  if (!stages.length) return { available: false, kValues: [], tValues: [], plannedRuns: 0, stages: [] };
  return { available: true, ...matrixFromStages(status, stages) };
}

/**
 * The control and baseline arms, for the collapsible secondary section.
 *
 * Anything in the plan that is not a primary tesseract/main stage is a control
 * or baseline arm, so a newly added arm appears without a code change.
 */
export function controlArms(status) {
  const plan = status?.plan;
  if (!isAvailable(plan)) return { available: false, arms: [], plannedRuns: 0 };
  const primaryT = status?.t_roles?.primary_depth || [];
  const isPrimary = (s) => s.family === 'tesseract' && s.task === 'main'
    && (s.t_values || []).some((t) => primaryT.includes(t));
  const arms = (plan.stages || [])
    .filter((s) => !isPrimary(s))
    .map((s) => ({
      key: `${s.label}-${s.stage}-${s.task}`,
      label: s.label,
      stage: s.stage,
      family: s.family_label || s.family,
      task: s.task_label || s.task,
      purpose: s.purpose,
      depths: s.depths || [],
      depthSymbol: s.depth_symbol || 'K',
      tValues: s.t_values || [],
      plannedRuns: s.planned_runs || 0,
      completedRuns: s.completed_runs || 0,
      state: s.state,
    }));
  return {
    available: arms.length > 0,
    arms,
    plannedRuns: arms.reduce((sum, a) => sum + a.plannedRuns, 0),
  };
}

// ---------------------------------------------------------------------------
// Results and analysis
// ---------------------------------------------------------------------------

/**
 * Results state.
 *
 * While no analysis artifact exists this reports the empty state together with
 * the backend's blockers and its list of planned metrics. It never produces a
 * number.
 */
export function resultsState(status) {
  const results = status?.results;
  if (isAvailable(results)) {
    return { available: true, blockers: [], plannedMetrics: results.planned_metrics || [], analysis: results.analysis };
  }
  return {
    available: false,
    message: results?.message || 'Primary experiments have not started.',
    blockers: results?.blocked_reasons || [],
    plannedMetrics: results?.planned_metrics || [],
    plannedRuns: results?.total_planned_runs ?? null,
    completedRuns: results?.total_completed_runs ?? null,
    analysis: null,
  };
}

/** Planned analyses, taken from the recorded decision rule where possible. */
export function analysisPlan(status) {
  const rule = status?.decisions?.amendment_02?.decision_rule || null;
  const primaryT = status?.t_roles?.primary_depth || [];
  const items = [
    { id: 'interaction', label: 'K x T interaction', detail: 'Regression with a log2K x log2T term' },
    { id: 'delta', label: 'Delta(T)', detail: 'Paired per-seed score(K high) - score(K low)' },
    { id: 'did', label: 'Difference in differences', detail: 'Delta at the deeper T minus Delta at the shallower T' },
    { id: 'kstar', label: 'K*(T)', detail: 'Smallest K reaching the recorded tau threshold' },
    { id: 'ci', label: 'Seed-level confidence intervals', detail: 'Paired 95 % t-intervals across seeds' },
    { id: 'compute', label: 'Compute / performance tradeoff', detail: 'Score against forward FLOPs per sequence' },
    { id: 'controls', label: 'Control comparisons', detail: 'Span, word, unrolled, parameter-matched, width-scaled' },
  ];
  return {
    primaryQuestion: 'Does useful recursive depth increase with transformation depth?',
    pending: !isAvailable(status?.results),
    primaryT,
    tau: rule?.tau ?? null,
    floorThreshold: rule?.floor_threshold ?? null,
    kLow: rule?.k_low ?? null,
    kHigh: rule?.k_high ?? null,
    score: rule?.score ?? null,
    items,
  };
}

/**
 * Recorded progress through the plan.
 *
 * Both numbers come from the plan payload, so the ring can only ever show what
 * has actually been recorded. `fraction` is null when nothing is planned yet,
 * which the view renders as an empty ring rather than as 0 % of nothing.
 */
export function runProgress(status) {
  const plan = status?.plan;
  const planned = plan?.total_planned_runs ?? 0;
  const completed = plan?.total_completed_runs ?? 0;
  const stages = pipelineStages(status);
  const stagesDone = stages.filter((s) => s.state === STAGE_DONE).length;
  return {
    planned,
    completed,
    fraction: planned > 0 ? completed / planned : null,
    percent: planned > 0 ? (completed / planned) * 100 : null,
    stagesDone,
    stagesTotal: stages.length,
    stageFraction: stages.length ? stagesDone / stages.length : null,
  };
}

// ---------------------------------------------------------------------------
// Methodology drawer
// ---------------------------------------------------------------------------

const field = (label, value, mono = false) => (value == null || value === '' ? null : { label, value, mono });

/** Grouped implementation detail, so the main view does not have to carry it. */
export function methodologyGroups(status) {
  const benchmark = status?.benchmark;
  const families = status?.model_families;
  const protocol = status?.protocol;
  const provenance = status?.provenance;
  const capacity = status?.capacity_diagnostics;
  const settings = capacity?.p1b?.settings || capacity?.p1b2?.settings || null;
  const decisions = status?.decisions;

  const groups = [];

  groups.push({
    id: 'architecture',
    title: 'Architecture',
    note: families?.note ?? null,
    fields: [
      field('Shared blocks', '1 TransformerBlock executed K times'),
      field('Latent states', 'z_H (EMA) and z_L, updated every execution'),
      field('Gradients', 'Full BPTT through every execution; no detach'),
      field('Vocabulary', families?.vocab_size, true),
      field('Ring size n', benchmark?.n, true),
    ].filter(Boolean),
  });

  groups.push({
    id: 'families',
    title: 'Model families',
    fields: (families?.families || []).map((f) =>
      field(f.label, `d_model ${f.d_model} · heads ${f.num_heads} · d_ff ${f.d_ff} · ${f.parameter_count?.toLocaleString()} params`, true),
    ).filter(Boolean),
  });

  if (settings) {
    groups.push({
      id: 'training',
      title: 'Training protocol (capacity diagnostic)',
      fields: [
        field('Optimizer', settings.optimizer ? `${settings.optimizer.name} · lr ${settings.optimizer.learning_rate} · wd ${settings.optimizer.weight_decay}` : null, true),
        field('Batch size', settings.batch_size, true),
        field('Max steps', settings.max_steps?.toLocaleString(), true),
        field('Evaluation cadence', settings.eval_every ? `every ${settings.eval_every} steps` : null, true),
        field('Early stopping', settings.early_stopping === false ? 'disabled (fixed budget)' : String(settings.early_stopping)),
        field('Test split', settings.test_split_evaluated === false ? 'not evaluated' : 'evaluated'),
        field('Floor threshold', settings.floor_threshold, true),
        field('Chance loss ln(vocab)', settings.chance_level_val_loss_ln_vocab?.toFixed?.(6), true),
      ].filter(Boolean),
    });
  }

  groups.push({
    id: 'seeds',
    title: 'Seed and data policy',
    fields: [
      field('Model seeds', Array.isArray(benchmark?.model_seeds) ? benchmark.model_seeds.join(', ') : null, true),
      field('Data seed', benchmark?.data_seed, true),
      field('Validation size', benchmark?.val_size?.toLocaleString(), true),
      field('Test size', benchmark?.test_size?.toLocaleString(), true),
      field('Tau', benchmark?.tau, true),
    ].filter(Boolean),
  });

  groups.push({
    id: 'amendments',
    title: 'Amendments',
    fields: [
      field('Amendment 01', decisions?.amendment_01?.state, false),
      field('Amendment 02', decisions?.amendment_02?.state, false),
      field('Recorded before training', decisions?.amendment_01?.recorded_before_training === true ? 'yes' : null),
    ].filter(Boolean),
  });

  groups.push({
    id: 'provenance',
    title: 'Provenance',
    fields: [
      field('Git commit', provenance?.git?.commit ? provenance.git.commit.slice(0, 12) : null, true),
      field('Working tree', provenance?.git?.dirty === true ? 'dirty' : provenance?.git?.dirty === false ? 'clean' : null),
      field('Runs root', provenance?.runs_root, true),
      field('Protocol source', protocol?.source, true),
      field('Experiment plan', status?.plan?.source, true),
    ].filter(Boolean),
  });

  return groups.filter((g) => g.fields.length > 0 || g.note);
}

/** Source paths worth surfacing, so any panel can be traced to a file. */
export function sourcePaths(status) {
  const paths = [
    status?.benchmark?.source,
    status?.decisions?.amendment_01?.source,
    status?.decisions?.amendment_02?.source,
    status?.plan?.source,
    status?.capacity_diagnostics?.p1b?.report_path,
    status?.capacity_diagnostics?.p1b2?.report_path,
  ].filter(Boolean);
  return [...new Set(paths)];
}

export { isAvailable, isPending };


// ---------------------------------------------------------------------------
// Page furniture
// ---------------------------------------------------------------------------

/**
 * The experimental contract strip: the fixed terms of the experiment.
 *
 * Every row is read from a payload. A term the backend has not supplied is
 * omitted rather than guessed, so the strip is always true even when short.
 */
export function experimentalContract(status) {
  const benchmark = status?.benchmark;
  const families = status?.model_families;
  const rows = [];
  const push = (label, value, mono = true) => {
    if (value != null && value !== '') rows.push({ label, value, mono });
  };

  if (isAvailable(benchmark)) {
    push('Benchmark', benchmark.group_description || benchmark.group, false);
    push('Sequence', benchmark.n != null ? `n = ${benchmark.n}` : null);
    push('Task depth', Array.isArray(benchmark.t_values) ? `T = ${benchmark.t_values.join(', ')}` : null);
    push('Recursion', Array.isArray(benchmark.k_values) ? `K = ${benchmark.k_values.join(', ')}` : null);
  }
  if (isAvailable(families)) {
    push('Vocabulary', families.vocab_size != null ? String(families.vocab_size) : null);
    const constant = (families.families || []).length > 0
      && (families.families || []).every((f) => f.parameter_count_is_constant_across_k);
    if (constant) push('Parameters', 'Constant across K', false);
    push('Architecture', '1 shared transformer block', false);
  }
  return rows;
}

/** The sections of the page, for the horizontal navigation. */
export const SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'methodology', label: 'Methodology' },
  { id: 'diagnostics', label: 'Diagnostics' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'probes', label: 'Probes & batches' },
  { id: 'results', label: 'Results' },
  { id: 'analysis', label: 'Analysis' },
];

/** Sections shown in presentation mode: research only, no implementation detail. */
export const PRESENTATION_SECTIONS = new Set([
  'overview',
  'methodology',
  'experiments',
  // The probes and diagnostic batches are research, not implementation detail:
  // they are the only Phase 2 results that exist, so presentation mode keeps them.
  'probes',
  'results',
  'analysis',
]);

/**
 * What presentation mode hides.
 *
 * Presentation mode is a different layout, not a different dataset: it removes
 * implementation and debug material (commit hashes, run directories, checksums,
 * raw metric names) and keeps the research. No figure changes between modes.
 */
export function presentationFilter(presentationMode) {
  return {
    showProvenance: !presentationMode,
    showMethodologyDrawer: !presentationMode,
    showSourcePaths: !presentationMode,
    showDiagnosticInternals: !presentationMode,
    showPlannedMetricNames: !presentationMode,
  };
}

/** Purpose lines for the control arms, keyed by the task the plan records. */
const CONTROL_PURPOSE = {
  c1_span: 'Tests span without sequential depth',
  c2_word: 'Tests logarithmic composition depth',
  unrolled: 'Tests independent parameters per layer',
  param_matched: 'Tests parameter-count effects',
  width_scaled: 'Tests width and capacity effects',
};

/**
 * Control arms condensed to one row each.
 *
 * The purpose line is matched on the recorded task or family name, falling back
 * to the plan's own purpose text, so an arm with no mapping still explains
 * itself rather than showing a blank cell.
 */
export function controlSummary(status) {
  const { available, arms, plannedRuns } = controlArms(status);
  return {
    available,
    plannedRuns,
    rows: arms.map((arm) => {
      // The plan carries a label ("C1 span") where the map is keyed by the
      // task id ("c1_span"), so both sides are normalised before matching.
      const normalise = (value) => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
      const haystack = `${normalise(arm.stage)} ${normalise(arm.task)} ${normalise(arm.family)}`;
      const key = Object.keys(CONTROL_PURPOSE).find((k) => haystack.includes(normalise(k)));
      return {
        key: arm.key,
        name: arm.task && arm.task !== 'A5 main' ? arm.task : arm.family,
        label: arm.label,
        purpose: key ? CONTROL_PURPOSE[key] : arm.purpose,
        plannedRuns: arm.plannedRuns,
        state: arm.state,
      };
    }),
  };
}

/** The three headline analyses, for the analysis section. */
export function headlineAnalyses() {
  return [
    { id: 'delta', symbol: 'Delta(T)', label: 'Performance gain from increased K' },
    { id: 'kstar', symbol: 'K*(T)', label: 'Minimum useful recursive depth' },
    { id: 'interaction', symbol: 'K x T', label: 'Interaction between computational depth and task depth' },
  ];
}
