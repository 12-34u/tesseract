export const projectMilestones = [
  { label: 'Phase 1 prototype + audit', status: 'DONE', source: 'PHASE1_FINAL_AUDIT.md' },
  { label: 'Phase 2 benchmark, gates, verifier', status: 'DONE', source: 'PHASE2_EXECUTION_PRD.md' },
  { label: 'Hidden-state confound found + per-step norm fix', status: 'DONE', source: 'PHASE2_BATCH_01_REPORT.md' },
  { label: 'Curriculum', status: 'DONE', source: 'PHASE2_BATCH_02_REPORT.md' },
  { label: 'T=3 solved at K>=4', status: 'DONE', source: 'PHASE2_BATCH_03_REPORT.md' },
  { label: 'T=4 learnable', status: 'DONE', source: 'PHASE2_BATCH_04_REPORT.md' },
  { label: 'Compute-matched test at T=3', status: 'DONE', source: 'PHASE2_BATCH_05_REPORT.md' },
  { label: 'T=4 solved at K=8', status: 'DONE', source: 'PHASE2_BATCH_06_REPORT.md' },
  { label: 'Compute-matched test at T=4', status: 'RUNNING', source: 'PHASE2_BATCH_07_PREREG.md' },
  { label: 'Learning-rate check', status: 'RUNNING', source: 'PHASE2_BATCH_07_PREREG.md' },
  { label: 'T=8', status: 'NOT STARTED', source: 'PHASE2_EXECUTION_PRD.md' },
  { label: 'Full pre-registered grid with span/word controls', status: 'NOT STARTED', source: 'PHASE2_EXECUTION_PRD.md' },
  { label: 'Benchmark amendment', status: 'NOT STARTED', source: 'PHASE2_BENCHMARK_DESIGN.md' },
  { label: 'Phase 3 (7M parameters)', status: 'NOT STARTED', source: 'ROADMAP.md' },
];

export const keyStats = [
  ['837,436', 'parameters at every K', 'PHASE2_BATCH_05_REPORT.md'],
  ['A5 order 60 · ring 17 · chance 1.67%', 'benchmark constants', 'PHASE2_BATCH_03_REPORT.md'],
  ['17.7x -> 1.0x', 'hidden-state growth to bounded growth; 0 added params', 'PHASE2_BATCH_01_REPORT.md'],
  ['T=3 · K=4 · 100.0000%', 'exact match', 'PHASE2_BATCH_03_REPORT.md'],
  ['T=4 · K=8 · 98.7793 / 90.4297 / 84.1309%', 'exact match by seed', 'PHASE2_BATCH_06_REPORT.md'],
  ['100% vs ~0%', 'Batch 05 shared arm vs 3.8x-larger alternatives', 'PHASE2_BATCH_05_REPORT.md'],
  ['~970', 'tests', 'PHASE2_EXECUTION_PRD.md'],
  ['4 decimals', 'results reproduced to this precision', 'PHASE2_BATCH_05_REPORT.md'],
];

export const parameterGroups = [
  {
    title: 'Model',
    rows: [
      ['Width', 'd_model 256 · 8 heads · d_ff 1024', 'configs/phase2/prototype_medium.yaml'],
      ['Parameters', '837,436; K-invariant', 'PHASE2_BATCH_05_REPORT.md'],
    ],
  },
  {
    title: 'Optimiser',
    rows: [
      ['Default', 'AdamW · lr 1e-3 · wd 0.01 · batch 128', 'PHASE2_BATCH_03_PREREG.md'],
      ['Batch 07 rate check', '1e-3 in phase 1; 3e-4 in conditional phase 2', 'PHASE2_BATCH_07_PREREG.md'],
    ],
  },
  {
    title: 'Batch protocol',
    rows: [
      ['Batch 01', 'T=1,2,4: 5,000 each; target T=4: 20,000; 20,000/40,000 arms; seeds 0,1,2; fix OFF', 'PHASE2_BATCH_01_PREREG.md'],
      ['Batch 02', 'T=1 -> 2 -> 3: 5,000 each; target T=3: 20,000; 35,000 total; seeds 0,1,2; fix OFF', 'PHASE2_BATCH_02_PREREG.md'],
      ['Batch 03', 'T=1 -> 2 -> 3: 5,000 each; target T=3: 20,000; 35,000 total; seeds 0,1,2; fix ON', 'PHASE2_BATCH_03_PREREG.md'],
      ['Batch 04', 'T=1 -> 2 -> 3 -> 4: 5,000 each; target T=4: 20,000; 40,000 total; seeds 0,1,2; fix ON', 'PHASE2_BATCH_04_PREREG.md'],
      ['Batch 05', 'Batch 03 curriculum; 35,000 total; seeds 0,1,2; fix ON', 'PHASE2_BATCH_05_PREREG.md'],
      ['Batch 06', 'T=1 (5,000) -> T=2 (5,000) -> T=3 (15,000) -> T=4 (5,000); target T=4: 60,000; 90,000 total; seeds 0,1,2; fix ON', 'PHASE2_BATCH_06_PREREG.md'],
      ['Batch 07', 'Same as Batch 06; report not run; seeds 0,1,2; fix ON', 'PHASE2_BATCH_07_PREREG.md'],
    ],
  },
];

export const batch07ReportExists = false;

export const limitations = [
  'Learning rate is inherited from the arm-A family.',
  'Three seeds per condition.',
  'T=3 fails the shortcut gate.',
  'The full pre-registered grid was not run.',
  'Runs used one laptop GPU.',
];

export const completedMilestones = projectMilestones.filter(({ status }) => status === 'DONE').length;