import { describe, expect, it } from 'vitest';
import {
  PRESENTATION_SECTIONS,
  SECTIONS,
  STAGE_ACTIVE,
  STAGE_ATTENTION,
  STAGE_DONE,
  STAGE_LOCKED,
  STAGE_WAITING,
  analysisPlan,
  controlArms,
  controlSummary,
  experimentalContract,
  d5Layer,
  d6Layer,
  gateGroups,
  headlineAnalyses,
  metadataChips,
  methodologyGroups,
  overallState,
  presentationFilter,
  pipelineStages,
  primaryDepthSentence,
  primaryMatrix,
  resultsState,
  runProgress,
  sourcePaths,
  stageTone,
  tRoleRows,
  widthCards,
} from './pipeline';

// The payloads below mirror app/backend/services/phase2_loader.py. The numbers in
// them are structural fixtures, not experimental values: these tests check the
// mapping, and in particular that a pending artifact can never turn into a number.

const BENCHMARK = {
  status: 'available',
  source: 'configs/phase2/benchmark.yaml',
  group: 'A5',
  group_description: 'A5 non-solvable group cellular automaton',
  transition: 'F(s)[i] = s[i] · s[i+1]',
  n: 17,
  t_values: [1, 2, 4, 8],
  k_values: [1, 2, 4, 8],
  data_seed: 1729,
  val_size: 2048,
  test_size: 10000,
  model_seeds: [0, 1, 2],
  tau: 0.9,
};

const FAMILIES = {
  status: 'available',
  vocab_size: 60,
  selected_base: null,
  selection_pending: true,
  note: 'One shared TransformerBlock executed K times.',
  families: [
    { name: 'small', label: 'Small Prototype (222K)', d_model: 128, num_heads: 4, d_ff: 512, parameter_count: 222140, parameter_count_is_constant_across_k: true },
    { name: 'large', label: 'Large Research Model (~7M)', d_model: 768, num_heads: 24, d_ff: 3072, parameter_count: 7230780, parameter_count_is_constant_across_k: true },
  ],
};

const ROLES = { status: 'available', anchor: [1], diagnostic: [2], primary_depth: [4, 8] };

const GATES = {
  status: 'available',
  raw_verdict: 'FAIL',
  cycles_passed: true,
  self_test_passed: true,
  per_t: [
    { t: 1, g1_passed: true, g1_tv_distance: 0.0072, g1_min_sensitivity: 1.0, g2_applicable: false, g2_passed: null, g2_state: 'N/A', amendment_01_diagnostic: false },
    { t: 2, g1_passed: true, g1_tv_distance: 0.0073, g1_min_sensitivity: 0.929, g2_applicable: true, g2_passed: false, g2_state: 'FAIL', g2_max_token_agreement: 0.2664, g2_candidate: 's[i+0]*s[i+2]', amendment_01_diagnostic: true },
    { t: 4, g1_passed: true, g1_tv_distance: 0.0085, g1_min_sensitivity: 0.961, g2_applicable: true, g2_passed: true, g2_state: 'PASS', amendment_01_diagnostic: false },
    { t: 8, g1_passed: true, g1_tv_distance: 0.0080, g1_min_sensitivity: 0.980, g2_applicable: true, g2_passed: true, g2_state: 'PASS', amendment_01_diagnostic: false },
  ],
};

const stage = (label, stageId, tValues, planned, state, extra = {}) => ({
  stage: stageId,
  label,
  purpose: `${label} purpose`,
  family: 'tesseract',
  family_label: 'Tesseract (shared block)',
  task: 'main',
  task_label: 'A5 main',
  t_values: tValues,
  depths: [1, 2, 4, 8],
  depth_symbol: 'K',
  seeds: [0, 1, 2],
  planned_runs: planned,
  completed_runs: 0,
  state,
  ...extra,
});

const PLAN = {
  status: 'available',
  source: 'configs/phase2/phase2_experiment.yaml',
  total_planned_runs: 138,
  total_completed_runs: 0,
  blocked_reasons: ['model width is PENDING_P1B', 'Amendment 02 is a draft and is not frozen'],
  stages: [
    stage('Stage #1', 't4', [4], 12, 'BLOCKED'),
    stage('Stage #2', 't8', [8], 12, 'BLOCKED'),
    stage('Stage #2b', 'anchors', [1, 2], 24, 'BLOCKED'),
    stage('Stage #3', 'controls', [4, 8], 24, 'BLOCKED', { task: 'c1_span', task_label: 'C1 span' }),
    stage('Stage #5', 'unrolled', [4, 8], 18, 'BLOCKED', { family: 'unrolled', family_label: 'Unrolled', depth_symbol: 'L' }),
  ],
};

const CAPACITY = {
  status: 'available',
  note: 'Large is an eligible candidate, not a selection. D5 recommends the smallest learnable width.',
  candidate_widths: [
    { name: 'small', label: 'Small Prototype (222K)', parameter_count: 222140, diagnostic: 'P1b', status: 'Candidate' },
    { name: 'medium', label: 'Medium Prototype (837K)', parameter_count: 837436, diagnostic: 'P1b', status: 'Candidate' },
    { name: 'large', label: 'Large Research Model (~7M)', parameter_count: 7230780, diagnostic: 'P1b-2', status: 'Candidate — P1b-2 pending' },
  ],
  p1b: {
    label: 'P1b — Original Capacity Diagnostic',
    widths: ['small', 'medium'],
    status: 'available',
    state: 'COMPLETE',
    report_path: 'runs/phase2/p1b_capacity_diagnostic/diagnostic_report.json',
    settings: {
      optimizer: { name: 'adamw', learning_rate: 0.001, weight_decay: 0.01 },
      batch_size: 128,
      max_steps: 20000,
      eval_every: 250,
      early_stopping: false,
      test_split_evaluated: false,
      floor_threshold: 0.02,
      chance_level_val_loss_ln_vocab: 4.0943445622221,
    },
    cells: [],
  },
  p1b2: {
    label: 'P1b-2 — Large Capacity Diagnostic',
    widths: ['large'],
    status: 'missing',
    state: 'NOT STARTED',
    message: 'P1b-2 has not started. Large is a candidate width awaiting its capacity diagnostic.',
    cells: [],
  },
  combined: {
    label: 'Combined D5 — Width Selection',
    status: 'missing',
    state: 'WAITING FOR P1b + P1b-2',
    awaiting: ['P1b-2'],
    message: 'Width selection waiting for P1b + P1b-2.',
    cells: [],
  },
};

const RESULTS = {
  status: 'missing',
  message: 'No Phase 2 experiment results yet.',
  total_planned_runs: 138,
  total_completed_runs: 0,
  blocked_reasons: ['model width is PENDING_P1B'],
  planned_metrics: ['chance_normalised_token_accuracy', 'exact_match'],
  analysis: null,
};

const DECISIONS = {
  amendment_01: { status: 'available', source: 'configs/phase2/amendment_01.yaml', state: 'ACTIVE — RECORDED', recorded_before_training: true },
  amendment_02: {
    status: 'available',
    source: 'configs/phase2/amendment_02_draft.yaml',
    state: 'DRAFT — NOT FROZEN',
    frozen: false,
    decisions: [{ id: 'D1', title: 'Z60 ladder dropped' }, { id: 'D5', title: 'P1b width and budget selection rules' }],
    decision_rule: { primary_t_values: [4, 8], k_low: 1, k_high: 8, tau: 0.9, floor_threshold: 0.02, score: 'test chance-normalised token accuracy' },
  },
};

const STATUS = {
  benchmark: BENCHMARK,
  model_families: FAMILIES,
  t_roles: ROLES,
  gates: GATES,
  plan: PLAN,
  capacity_diagnostics: CAPACITY,
  results: RESULTS,
  decisions: DECISIONS,
  protocol: { status: 'available', source: 'configs/phase2/benchmark.yaml' },
  provenance: { git: { commit: 'a69054f6ca6b7a8381595927f8ff889619da365a', dirty: true }, runs_root: '/repo/runs' },
};

describe('overall state', () => {
  it('reads pre-experiment when no diagnostic has completed', () => {
    const noDiagnostics = { ...STATUS, capacity_diagnostics: { ...CAPACITY, p1b: { ...CAPACITY.p1b, state: 'NOT STARTED' } } };
    expect(overallState(noDiagnostics).label).toBe('PRE-EXPERIMENT');
  });

  it('reads diagnostic phase once a capacity diagnostic is complete', () => {
    expect(overallState(STATUS).label).toBe('DIAGNOSTIC PHASE');
  });

  it('never uses the failure tone for an ordinary pending state', () => {
    for (const s of [STATUS, { ...STATUS, plan: { ...PLAN, total_completed_runs: 0 } }]) {
      expect(overallState(s).tone).not.toBe('fail');
    }
  });

  it('tracks running and complete from recorded run counts', () => {
    expect(overallState({ ...STATUS, plan: { ...PLAN, total_completed_runs: 5 } }).label).toBe('EXPERIMENT RUNNING');
    expect(overallState({ ...STATUS, plan: { ...PLAN, total_completed_runs: 138 } }).label).toBe('EXPERIMENT COMPLETE');
  });
});

describe('metadata chips', () => {
  it('reads the grids from the payload rather than literals', () => {
    const chips = metadataChips(STATUS);
    const byLabel = Object.fromEntries(chips.map((c) => [c.label, c.value]));
    expect(byLabel.K).toBe('{1, 2, 4, 8}');
    expect(byLabel.T).toBe('{1, 2, 4, 8}');
    expect(byLabel.n).toBe('17');
    expect(byLabel.Vocab).toBe('60');
  });

  it('omits what the backend has not supplied instead of guessing', () => {
    const chips = metadataChips({ benchmark: { status: 'missing' }, model_families: { status: 'missing' } });
    expect(chips).toEqual([]);
  });

  it('states parameter constancy only when every family reports it', () => {
    const varying = {
      ...STATUS,
      model_families: { ...FAMILIES, families: [{ ...FAMILIES.families[0], parameter_count_is_constant_across_k: false }] },
    };
    expect(metadataChips(varying).some((c) => c.label === 'Shared block')).toBe(false);
  });
});

describe('pipeline stages', () => {
  it('produces the seven research stages in order', () => {
    const stages = pipelineStages(STATUS);
    expect(stages.map((s) => s.index)).toEqual(['01', '02', '03', '04', '05', '06', '07']);
    expect(stages.map((s) => s.id)).toEqual(['benchmark', 'gates', 'capacity', 'width', 'budget', 'matrix', 'analysis']);
  });

  it('treats a documented G2 diagnostic exception as done, not as a failure', () => {
    const gates = pipelineStages(STATUS).find((s) => s.id === 'gates');
    expect(gates.state).toBe(STAGE_DONE);
    expect(gates.summary).toContain('T=2');
  });

  it('flags an undocumented G2 failure for attention', () => {
    const broken = {
      ...STATUS,
      gates: { ...GATES, per_t: [{ t: 4, g1_passed: true, g2_applicable: true, g2_passed: false, g2_state: 'FAIL', amendment_01_diagnostic: false }] },
    };
    expect(pipelineStages(broken).find((s) => s.id === 'gates').state).toBe(STAGE_ATTENTION);
  });

  it('marks capacity active while one of two diagnostics is complete', () => {
    expect(pipelineStages(STATUS).find((s) => s.id === 'capacity').state).toBe(STAGE_ACTIVE);
  });

  it('keeps width and budget waiting while D5 has not run', () => {
    const stages = pipelineStages(STATUS);
    expect(stages.find((s) => s.id === 'width').state).toBe(STAGE_WAITING);
    expect(stages.find((s) => s.id === 'budget').state).toBe(STAGE_WAITING);
    expect(stages.find((s) => s.id === 'width').summary).toContain('P1b-2');
  });

  it('locks the matrix and carries the blockers when preconditions are outstanding', () => {
    const matrix = pipelineStages(STATUS).find((s) => s.id === 'matrix');
    expect(matrix.state).toBe(STAGE_LOCKED);
    expect(matrix.blockers).toEqual(PLAN.blocked_reasons);
  });

  it('states a dependency for every stage after the first', () => {
    const stages = pipelineStages(STATUS);
    expect(stages[0].dependency).toBeNull();
    expect(stages.slice(1).every((s) => typeof s.dependency === 'string' && s.dependency.length > 0)).toBe(true);
  });

  it('never gives a pending stage the failure tone', () => {
    for (const s of pipelineStages(STATUS)) expect(stageTone(s.state)).not.toBe('fail');
  });
});

describe('benchmark and gates', () => {
  it('labels T roles from the amendment partition', () => {
    expect(tRoleRows(STATUS)).toEqual([
      { t: 1, role: 'Anchor', tone: 'idle' },
      { t: 2, role: 'Diagnostic', tone: 'note' },
      { t: 4, role: 'Primary', tone: 'ok' },
      { t: 8, role: 'Primary', tone: 'ok' },
    ]);
  });

  it('keeps the backend wording for a diagnostic G2 failure', () => {
    const groups = gateGroups(STATUS);
    const t2 = groups.g2.find((r) => r.t === 2);
    expect(t2.state).toBe('FAIL');
    expect(t2.diagnostic).toBe(true);
    expect(groups.rawVerdict).toBe('FAIL');
  });

  it('summarises G1 across every T', () => {
    const g1 = gateGroups(STATUS).g1;
    expect(g1.passed).toBe(true);
    expect(g1.minSensitivity).toBeCloseTo(0.929);
    expect(g1.cyclesPassed).toBe(true);
  });

  it('reports the pending state when no gate report exists', () => {
    const groups = gateGroups({ gates: { status: 'missing', message: 'no gates' } });
    expect(groups.available).toBe(false);
    expect(groups.g1).toBeNull();
  });

  it('states which T values carry the depth claim', () => {
    expect(primaryDepthSentence(STATUS)).toBe('Primary depth analysis uses T=4 and T=8.');
    expect(primaryDepthSentence({ t_roles: {} })).toBeNull();
  });
});

describe('width cards', () => {
  it('marks the research-scale width from its diagnostic, not as a selection', () => {
    const cards = widthCards(STATUS);
    const large = cards.find((c) => c.name === 'large');
    expect(large.researchScale).toBe(true);
    expect(large.selected).toBe(false);
    expect(large.statusText).toContain('Candidate');
  });

  it('never marks a width selected while D5 has not chosen one', () => {
    expect(widthCards(STATUS).every((c) => c.selected === false)).toBe(true);
  });

  it('marks a width selected only once the combined D5 artifact says so', () => {
    const chosen = {
      ...STATUS,
      capacity_diagnostics: {
        ...CAPACITY,
        combined: {
          ...CAPACITY.combined,
          status: 'available',
          state: 'SELECTED',
          d5: { decision: { width_decision: { selected_width: 'medium', outcome: 'ok' }, budget_decision: { recommended_max_steps: 15000 } }, evaluated_once_on_combined_report: true },
        },
      },
    };
    expect(widthCards(chosen).find((c) => c.name === 'medium').selected).toBe(true);
  });

  it('returns nothing when the payload lists no candidates', () => {
    expect(widthCards({ capacity_diagnostics: { status: 'missing' } })).toEqual([]);
  });
});

describe('the research width gate', () => {
  it('reports D5 as not evaluated while the combined report is missing', () => {
    const d5 = d5Layer(STATUS);
    expect(d5.evaluated).toBe(false);
    expect(d5.selectedWidth).toBeNull();
    expect(d5.recommendedMaxSteps).toBeNull();
    expect(d5.awaiting).toEqual(['P1b-2']);
    expect(d5.amendmentFrozen).toBe(false);
  });

  it('surfaces the recorded D5 title from the amendment', () => {
    expect(d5Layer(STATUS).recordedTitle).toContain('width and budget selection rules');
  });

  it('reports no formal D6 entry, because Amendment 02 records D1-D5', () => {
    const d6 = d6Layer(STATUS);
    expect(d6.recorded).toBe(false);
    expect(d6.recordedTitle).toBeNull();
    expect(d6.note).toContain('Amendment 02 records D1-D5');
  });

  it('designates the research width from the P1b-2 configuration, not from performance', () => {
    const d6 = d6Layer(STATUS);
    expect(d6.designatedWidth.name).toBe('large');
    expect(d6.designationSource).toBe('P1b-2 configuration');
    // A designation is not a selection: D5 has still chosen nothing.
    expect(d6.designatedWidth.selected).toBe(false);
  });

  it('shows the policy that is actually recorded rather than asserting a designation', () => {
    const d6 = d6Layer(STATUS);
    expect(d6.recordedPolicy).toContain('candidate, not a selection');
    expect(d6.recordedPolicy).toContain('smallest learnable width');
  });

  it('picks up a D6 decision if one is ever added, with no code change', () => {
    const withD6 = {
      ...STATUS,
      decisions: {
        ...DECISIONS,
        amendment_02: { ...DECISIONS.amendment_02, decisions: [...DECISIONS.amendment_02.decisions, { id: 'D6', title: 'Width policy' }] },
      },
    };
    const d6 = d6Layer(withD6);
    expect(d6.recorded).toBe(true);
    expect(d6.recordedTitle).toBe('Width policy');
    expect(d6.note).toBeNull();
  });

  it('keeps the recorded D5 selection rule visible alongside the designation', () => {
    const d6 = d6Layer(STATUS);
    expect(d6.recordedPolicy).toContain('smallest learnable width');
    expect(d6.designatedWidth.name).toBe('large');
  });
});

describe('primary matrix', () => {
  it('takes K, T, seeds and run counts from the plan rather than literals', () => {
    const matrix = primaryMatrix(STATUS);
    expect(matrix.available).toBe(true);
    expect(matrix.kValues).toEqual([1, 2, 4, 8]);
    expect(matrix.tValues).toEqual([4, 8]);
    expect(matrix.seeds).toEqual([0, 1, 2]);
    expect(matrix.plannedRuns).toBe(24);
    expect(matrix.completedRuns).toBe(0);
  });

  it('excludes the anchor and diagnostic stage from the primary matrix', () => {
    expect(primaryMatrix(STATUS).tValues).not.toContain(1);
    expect(primaryMatrix(STATUS).tValues).not.toContain(2);
  });

  it('reports every cell as planned and carries no metrics before any run', () => {
    const matrix = primaryMatrix(STATUS);
    for (const t of matrix.tValues) {
      for (const k of matrix.kValues) {
        expect(matrix.cell(t, k)).toEqual({ state: 'planned', metrics: null });
      }
    }
  });

  it('is locked while any primary stage is blocked', () => {
    expect(primaryMatrix(STATUS).locked).toBe(true);
  });

  it('reports unavailable when the plan is missing', () => {
    expect(primaryMatrix({ plan: { status: 'missing' } }).available).toBe(false);
  });
});

describe('control arms', () => {
  it('collects every non-primary arm from the plan', () => {
    const controls = controlArms(STATUS);
    expect(controls.available).toBe(true);
    expect(controls.arms.map((a) => a.label)).toEqual(['Stage #2b', 'Stage #3', 'Stage #5']);
    expect(controls.plannedRuns).toBe(24 + 24 + 18);
  });

  it('keeps the recorded depth symbol for non-shared families', () => {
    const unrolled = controlArms(STATUS).arms.find((a) => a.stage === 'unrolled');
    expect(unrolled.depthSymbol).toBe('L');
    expect(unrolled.family).toBe('Unrolled');
  });

  it('picks up a newly added arm without a code change', () => {
    const extended = {
      ...STATUS,
      plan: { ...PLAN, stages: [...PLAN.stages, stage('Stage #8', 'new_control', [4], 6, 'BLOCKED', { task: 'c3_new', task_label: 'C3 new' })] },
    };
    expect(controlArms(extended).arms.some((a) => a.task === 'C3 new')).toBe(true);
  });
});

describe('results', () => {
  it('reports the empty state with the backend message and blockers', () => {
    const state = resultsState(STATUS);
    expect(state.available).toBe(false);
    expect(state.analysis).toBeNull();
    expect(state.blockers).toEqual(['model width is PENDING_P1B']);
    expect(state.plannedMetrics).toContain('chance_normalised_token_accuracy');
  });

  it('produces no numbers of its own while results are missing', () => {
    const state = resultsState(STATUS);
    expect(state.completedRuns).toBe(0);
    expect(state).not.toHaveProperty('accuracy');
  });

  it('switches to available once an analysis artifact exists', () => {
    const done = { ...STATUS, results: { status: 'available', planned_metrics: [], analysis: { ok: true } } };
    expect(resultsState(done).available).toBe(true);
  });
});

describe('run progress', () => {
  it('reports recorded and planned counts straight from the plan', () => {
    const progress = runProgress(STATUS);
    expect(progress.planned).toBe(138);
    expect(progress.completed).toBe(0);
    expect(progress.fraction).toBe(0);
  });

  it('returns a null fraction when nothing is planned, rather than 0 % of nothing', () => {
    const progress = runProgress({ plan: { status: 'missing' } });
    expect(progress.planned).toBe(0);
    expect(progress.fraction).toBeNull();
    expect(progress.percent).toBeNull();
  });

  it('tracks partial progress', () => {
    const progress = runProgress({ ...STATUS, plan: { ...PLAN, total_completed_runs: 69 } });
    expect(progress.fraction).toBeCloseTo(0.5);
    expect(progress.percent).toBeCloseTo(50);
  });

  it('counts completed stages against the pipeline length', () => {
    const progress = runProgress(STATUS);
    expect(progress.stagesTotal).toBe(7);
    expect(progress.stagesDone).toBeGreaterThan(0);
    expect(progress.stageFraction).toBeCloseTo(progress.stagesDone / 7);
  });

  it('never reports progress beyond what was recorded', () => {
    const progress = runProgress(STATUS);
    expect(progress.completed).toBeLessThanOrEqual(progress.planned);
  });
});

describe('analysis plan', () => {
  it('stays marked pending while no results exist', () => {
    const plan = analysisPlan(STATUS);
    expect(plan.pending).toBe(true);
    expect(plan.primaryQuestion).toContain('recursive depth');
  });

  it('reads thresholds from the recorded decision rule', () => {
    const plan = analysisPlan(STATUS);
    expect(plan.tau).toBe(0.9);
    expect(plan.floorThreshold).toBe(0.02);
    expect(plan.kLow).toBe(1);
    expect(plan.kHigh).toBe(8);
    expect(plan.primaryT).toEqual([4, 8]);
  });

  it('leaves thresholds null when no decision rule is recorded', () => {
    const plan = analysisPlan({ results: { status: 'missing' } });
    expect(plan.tau).toBeNull();
    expect(plan.floorThreshold).toBeNull();
  });
});

describe('methodology', () => {
  it('groups architecture, training, seeds and provenance', () => {
    const ids = methodologyGroups(STATUS).map((g) => g.id);
    expect(ids).toContain('architecture');
    expect(ids).toContain('training');
    expect(ids).toContain('seeds');
    expect(ids).toContain('provenance');
  });

  it('drops fields the backend did not supply instead of showing blanks', () => {
    const groups = methodologyGroups({ benchmark: { status: 'missing' }, model_families: { status: 'missing' } });
    for (const group of groups) {
      for (const f of group.fields) expect(f.value).not.toBeNull();
    }
  });

  it('shortens the git commit for display but keeps it traceable', () => {
    const provenance = methodologyGroups(STATUS).find((g) => g.id === 'provenance');
    expect(provenance.fields.find((f) => f.label === 'Git commit').value).toBe('a69054f6ca6b');
  });

  it('collects the source paths behind the panels', () => {
    const paths = sourcePaths(STATUS);
    expect(paths).toContain('configs/phase2/benchmark.yaml');
    expect(paths).toContain('configs/phase2/phase2_experiment.yaml');
    expect(new Set(paths).size).toBe(paths.length);
  });
});

describe('experimental contract', () => {
  it('reads every term from the payload', () => {
    const rows = Object.fromEntries(experimentalContract(STATUS).map((r) => [r.label, r.value]));
    expect(rows.Benchmark).toBe('A5 non-solvable group cellular automaton');
    expect(rows.Sequence).toBe('n = 17');
    expect(rows.Recursion).toBe('K = 1, 2, 4, 8');
    expect(rows['Task depth']).toBe('T = 1, 2, 4, 8');
    expect(rows.Vocabulary).toBe('60');
    expect(rows.Parameters).toBe('Constant across K');
  });

  it('omits terms the backend has not supplied rather than guessing', () => {
    expect(experimentalContract({ benchmark: { status: 'missing' }, model_families: { status: 'missing' } })).toEqual([]);
  });

  it('does not claim constant parameters unless every family reports it', () => {
    const varying = {
      ...STATUS,
      model_families: { ...FAMILIES, families: [{ ...FAMILIES.families[0], parameter_count_is_constant_across_k: false }] },
    };
    expect(experimentalContract(varying).some((r) => r.label === 'Parameters')).toBe(false);
  });
});

describe('sections and presentation mode', () => {
  it('lists the six research sections in reading order', () => {
    expect(SECTIONS.map((s) => s.id)).toEqual(
      ['overview', 'methodology', 'diagnostics', 'experiments', 'results', 'analysis'],
    );
  });

  it('hides implementation detail in presentation mode but keeps the research', () => {
    const on = presentationFilter(true);
    expect(on.showProvenance).toBe(false);
    expect(on.showMethodologyDrawer).toBe(false);
    expect(on.showSourcePaths).toBe(false);
    const off = presentationFilter(false);
    expect(Object.values(off).every(Boolean)).toBe(true);
  });

  it('drops only the diagnostics section from the presentation running order', () => {
    const hidden = SECTIONS.filter((s) => !PRESENTATION_SECTIONS.has(s.id)).map((s) => s.id);
    expect(hidden).toEqual(['diagnostics']);
  });
});

describe('control summary', () => {
  it('gives every arm a purpose line', () => {
    const summary = controlSummary(STATUS);
    expect(summary.available).toBe(true);
    expect(summary.rows.every((r) => typeof r.purpose === 'string' && r.purpose.length > 0)).toBe(true);
  });

  it('maps the recorded control tasks onto their research purpose', () => {
    const rows = controlSummary(STATUS).rows;
    expect(rows.find((r) => r.name === 'C1 span').purpose).toContain('span without sequential depth');
    expect(rows.find((r) => r.label === 'Stage #5').purpose).toContain('independent parameters');
  });

  it('falls back to the plan purpose for an unmapped arm', () => {
    const extended = {
      ...STATUS,
      plan: { ...PLAN, stages: [stage('Stage #9', 'mystery', [4], 3, 'BLOCKED', { task: 'c9_new', task_label: 'C9 new' })] },
    };
    const row = controlSummary(extended).rows.find((r) => r.label === 'Stage #9');
    expect(row.purpose).toBe('Stage #9 purpose');
  });
});

describe('headline analyses', () => {
  it('names the three primary outputs', () => {
    expect(headlineAnalyses().map((a) => a.id)).toEqual(['delta', 'kstar', 'interaction']);
  });
});
