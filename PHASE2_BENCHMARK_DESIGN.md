# Phase 2 Benchmark Design Proposal — Sequential Computational Depth

Status: **approved. Amended before any model training by Amendment 01 (§8).**
Phase 1 is untouched.
No experimental results are reported here. The only computed facts are
symbolic properties of the task definitions (marked *computed*).

Variables: **P** = trainable parameters, **K** = recursive executions of the
shared block, **T** = sequential transformations the task requires.

---

## 1. What a valid benchmark must rule out

Phase 1 Rule 90 failed because F^T collapsed algebraically: for T = 2^m,
F^T is a two-cell XOR. A valid Phase 2 task must rule out four cheaper
explanations for "more T is harder":

| Confound | Why it is dangerous | Required property |
|---|---|---|
| Algebraic shortcut | F^T collapses to a shallow expression (linearity, periodicity, nilpotence) | F nonlinear; no known closed form; checked empirically |
| Receptive field / span | Harder only because targets depend on farther cells | A control with the same span but constant depth |
| Lookup memorisation | F^T learned as a flat table over its input window | Window pattern space ≫ data seen and ≫ model capacity for T ≥ 4 |
| Sequence length | Harder only because inputs get longer | Input length fixed across T |

---

## 2. Candidate designs

### Candidate A — Nonlinear elementary CA (Rule 30 / Rule 110)

| | |
|---|---|
| **Definition** | Binary ring of n cells; F = one Rule 30 (or 110) update; predict F^T(s₀). |
| **Input / target** | n bits → n bits. |
| **One transformation** | One synchronous CA update. |
| **Why T ≈ depth** | Each step is a depth-1 nonlinear local map; no closed form is known for Rule 30. |
| **Shortcut risk** | **High, via lookup.** The target cell is a Boolean function of 2T+1 bits: only 2¹⁷ = 131,072 patterns at T = 8, and 8 at T = 1. A model can learn F^T as a window table with width instead of depth. Rule 30 is also partly linear, and Rule 110 has periodic background structure. |
| **Difficulty** | Moderate. |
| **Data** | Unlimited (uniform random bits). |
| **Verification** | Independent simulator (lookup-table vs. Boolean-formula implementations). |
| **Failure modes** | Width-based memorisation mimics depth; span grows as 2T+1; target bits are biased per rule. |
| **Suitable?** | **No** as the primary benchmark: it fails requirement 8 (finite lookup) at the T values we need. |

### Candidate B — S₅ word problem (prefix product of T given permutations)

| | |
|---|---|
| **Definition** | Input: state π₀ ∈ S₅ and T generator tokens g₁…g_T. Target: π₀·g₁·…·g_T. |
| **Input / target** | 1 + T tokens → 1 token (or all prefix products). |
| **One transformation** | Multiply the state by the next generator. |
| **Why T ≈ depth** | S₅ is non-solvable; its word problem is NC¹-complete (Barrington), so no constant-depth shortcut exists unless TC⁰ = NC¹. |
| **Shortcut risk** | Low for constant depth, **but depth is only log₂T** (parallel prefix via associativity). |
| **Difficulty** | Well studied ("state tracking"). |
| **Data** | Unlimited. |
| **Verification** | Permutation composition. |
| **Failure modes** | **Input length grows with T** (violates requirement 5). T generators are separate inputs, so each "step" consumes new input, which confounds depth with sequence length. |
| **Suitable?** | **No** as the primary benchmark. It is a useful *secondary* benchmark with established literature. |

### Candidate C — Pointer chasing (T-hop lookup in a random function)

| | |
|---|---|
| **Definition** | Input: a random function f:[m]→[m] written as m tokens, plus a start x₀. Target: f^T(x₀). |
| **Input / target** | m + 1 tokens → 1 token. |
| **One transformation** | One hop x ← f(x). |
| **Why T ≈ depth** | Each hop needs the result of the previous hop. |
| **Shortcut risk** | **Pointer doubling:** one layer can build f∘f for every entry in parallel, so f^T takes ⌈log₂T⌉ layers. f^T is also eventually periodic for small m. |
| **Difficulty** | Attention-native and easy per hop. |
| **Data** | Unlimited (fresh f per example, so no memorisation). |
| **Verification** | Trivial simulation. |
| **Failure modes** | One output token gives a weak training signal; the depth scaling is log T, not T. |
| **Suitable?** | **Partial.** A good ablation, not the primary benchmark. |

### Candidate D — Non-solvable group cellular automaton (A₅-CA) ★

| | |
|---|---|
| **Definition** | Ring of n cells, each an element of the alternating group A₅ (order 60). F(s)[i] = s[i] · s[i+1] (non-commutative group product). Predict F^T(s₀). |
| **Input / target** | n tokens (vocabulary 60) → n tokens. |
| **One transformation** | One synchronous group-multiplication update. |
| **Why T ≈ depth** | The target word has 2^T letters and 3·2^(T−2) runs after merging repeats (*computed*: 2, 3, 12, 192 runs at T = 1, 2, 4, 8). Generic pairwise composition therefore needs depth ⌈log₂ runs⌉ = **T exactly** (*computed* for T = 1…8 and 16). A₅ is the smallest non-solvable group, so no commutative or solvable simplification applies. |
| **Shortcut risk** | Low. Evaluating arbitrary A₅ words is NC¹-complete; the collapse that killed Rule 90 needs commutativity; window lookup needs 60^(T+1) entries. |
| **Difficulty** | High. Tunable via n, group, and T. |
| **Data** | Unlimited (uniform over A₅ⁿ ≈ 60¹⁷ ≈ 1.7·10³⁰ states at n = 17). |
| **Verification** | Two independent group implementations plus symbolic word expansion (§4.10). |
| **Failure modes** | Floor effects at T = 8 for the ~211K model; learning a 60×60 table as the primitive; ring wrap with absolute positions (§6). |
| **Suitable?** | **Yes — recommended.** It is the only candidate that meets requirements 1–12 together, and it comes with built-in controls (§5). |

---

## 3. Recommendation

**Candidate D: the A₅ group cellular automaton**, with two matched control
tasks and a group-solvability ladder. The controls are what separate *depth*
from *span*, *lookup*, and *algebraic collapse*.

---

## 4. Formal definition of the recommended benchmark

**4.1 Group.** G = A₅, the even permutations of {0,…,4}, |G| = 60. The
product is composition, (a·b)(x) = a(b(x)). Element ids 0–59 follow the
lexicographic order of the permutation arrays. Ids are arbitrary labels
with no arithmetic meaning. An ablation re-randomises labels per seed.

**4.2 State space.** S = Gⁿ, a ring of n cells with periodic indices (i mod n).

**4.3 Transition.** F: S → S, F(s)[i] = s[i] · s[(i+1) mod n].

**4.4 Initial state.** S₀ ~ Uniform(Gⁿ): each cell is independent and uniform.

**4.5 Target.** S_T = F^T(S₀), computed by T explicit iterations.

**4.6 Model input.** The n token ids of S₀. There is no T token in the
main grid: one model per (T, K), as in Phase 1. A mixed-T variant with a
T-prefix token is used only for the generalisation tests.

**4.7 Model output.** n tokens, one 60-way prediction per cell: all of S_T.

**4.8 Grid.** T ∈ {1, 2, 4, 8}, K ∈ {1, 2, 4, 8}. Evaluation-only T values
∈ {3, 5, 6, 7} (§4.12).

**4.9 Sequence length.** n = 17.
- The window T+1 = 9 at T = 8 never wraps onto itself.
- n is prime, avoiding the power-of-two ring resonances behind Phase 1's
  collapse (i−8 ≡ i+8 mod 16).
- n = 33 is used only for the length test.

**4.10 Data generation and verification**
- **Unlimited stream.** Training batches are sampled fresh every step (primary protocol), so memorisation of individual examples is impossible.
- **Fixed splits.** Validation: 2,048 states per T. Test: 10,000 states per T. Length test: 2,000 states at n = 33. Validation and test are generated once, stored, and SHA-256 checksummed.
- **Sample-efficiency variant** (secondary). Fixed training sets of 2¹², 2¹⁵ and 2¹⁸ states.
- **Leakage guard.** Validation and test states are hashed; any exact match in a training batch is rejected and counted. The expected count is ≈ 0 given the state-space size, but it is logged anyway.
- **Independent verification** (a benchmark gate before any training):
  1. Implementation A: numpy permutation composition.
  2. Implementation B: a Cayley table built from generators (0 1 2)(3 4)… closure. Check isomorphism with A, associativity, identity, and inverses, and that A₅ is perfect ([G,G] = G, i.e. non-solvable).
  3. Cross-check F^T from iteration against direct evaluation of the symbolically expanded word (2^T letters) for T ≤ 8.

**4.11 Seeds.** Use independent `numpy.random.SeedSequence` streams keyed
by (root seed, split, T, n):
- The training stream is independent of the validation and test streams.
- Model initialisation seeds are {0, 1, 2}, with a minimum of 3 seeds per cell.
- Seeds are recorded in `run_metadata.json`. Within a seed, all (T, K) cells share both the initial weights (Phase 1 practice) and the training stream.

**4.12 Evaluation metrics**

| Metric | Role |
|---|---|
| Sequence exact match (all 17 cells) | Primary |
| Token accuracy, and chance-normalised (acc − 1/60)/(1 − 1/60) | Primary partial credit, avoiding exact-match floor effects |
| Per-position accuracy | Detects position-specific shortcuts (seen in Phase 1: 39–100 %) |
| K*(T) = smallest K reaching ≥ τ normalised token accuracy (τ fixed before the runs, e.g. 0.9) | **Main depth statistic** |
| Interaction Δ(T) = acc(K=8,T) − acc(K=1,T) | Tests whether K becomes more useful as T grows |
| Accuracy vs. training compute (FLOPs ∝ K) | Compute-matched comparison |

**Statistics.** Mean ± 95 % CI over seeds. Bootstrap CIs over test
examples. A regression of accuracy on log₂K × log₂T with a seed random
effect; the pre-registered hypothesis is a positive interaction.

**Training protocol.**
- A **fixed step budget** for every cell, instead of early stopping on
  training loss, which was a Phase 1 confound. The budget is chosen in a
  pilot such that T = 1, K = 1 converges.
- The learning rate is tuned once per model family, on validation at T = 2,
  then frozen.
- The checkpoint with the best validation score is evaluated once on test.

**4.13 Generalisation tests**
1. **IID test** on the frozen test split.
2. **T interpolation.** Train mixed-T {1, 2, 4, 8} with a T token; test T ∈ {3, 5, 6, 7}.
3. **Test-time recursion.** Train at K; evaluate at K′ ∈ {K/2, 2K} (Tesseract-specific).
4. **Length.** Train n = 17, test n = 33 (absolute positions are expected to fail; this is reported, not hidden).
5. **Shift equivariance.** Test that rotating S₀ by r rotates the prediction by r.
6. **Coverage diagnostic.** The fraction of test windows (T+1 cells) whose exact pattern occurred during training. It is high at T ≤ 2 and ≈ 0 at T ≥ 4, which marks where lookup is even possible.

**4.14 Baseline requirements** (same tokenizer, positions, optimizer, step budget, splits, seeds)

| Baseline | Matches Tesseract's | Tests |
|---|---|---|
| **Unrolled non-shared**, L = K distinct blocks | Compute (depth) | Does sharing cost accuracy at equal depth? |
| **Parameter-matched non-shared**, L ∈ {1,2,4,8} with d_model shrunk so P ≈ P_Tesseract | Parameters | Is recursion better than depth at a fixed budget? |
| **Width-scaled K = 1**, a single block with d_ff raised to match Tesseract-K=8's compute or parameters | Compute / params, spent on width | Can width replace depth? This is the direct lookup-shortcut test. |
| **Trivial:** chance (1/60), copy-input, oracle simulator | — | Sanity bounds |

**4.15 Required ablations**
1. **Group ladder** (same vocabulary of 60, same F): Z₆₀ (abelian: F^T is the closed-form Σ C(T,j)·s[i+j] mod 60, a shallow expression) → D₃₀ (solvable, non-abelian: constant/log-depth shortcuts exist in principle) → A₅ (non-solvable).
2. **Span and word controls** (§5).
3. **Tesseract internals:** single state (z_L only), fixed vs. learned α, no x re-injection, truncated BPTT (detach every step), plus a final LayerNorm variant (addresses the Phase 1 residual-growth confound as a *named* variant).
4. **Label randomisation** of element ids per seed.
5. **Positional scheme:** learned absolute vs. cyclic/relative. The task is ring-equivariant, so this ablation is recommended but secondary.

---

## 5. Why this measures SEQUENTIAL DEPTH rather than receptive field

The argument has four independent parts. Parts 1–3 are theory; part 4 is
the empirical test that decides.

**1. Receptive field is never the model's bottleneck.** One attention layer
already sees all n = 17 cells. The largest dependency window, T+1 = 9 at
T = 8, fits inside it at K = 1. A failure at small K therefore cannot be
"the model could not see far enough"; it can only be "the model could not
compute enough".

**2. Flat lookup over the window is infeasible where it matters.** The
target cell is a function of T+1 A₅ elements:

| T | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Window patterns 60^(T+1) | 3.6·10³ | 2.2·10⁵ | 7.8·10⁸ | 1.0·10¹⁶ |

T = 1 *is* the multiplication table, the primitive the model must learn.
At T = 2 lookup is conceivable, and the coverage diagnostic reports it.
For T ≥ 4 the table exceeds any Phase 2 model (~6–7M parameters) by orders
of magnitude, so F^T must be **computed**.

**3. The computation has depth that grows linearly in T.** F^T(s)[i] is a
word of 2^T letters whose run count is 3·2^(T−2) (*computed*). Evaluating it
by pairwise composition needs exactly T levels, which is F's own iteration
structure. The Rule 90 collapse required commutativity, which A₅ lacks.
Because A₅ is non-solvable, evaluating general A₅ words is NC¹-complete,
while bounded-depth, bounded-precision transformers lie in TC⁰. So **no
constant-depth solution is expected unless TC⁰ = NC¹**.

*Caveat:* this family of words is specific and T ≤ 8 is finite, so the
theory motivates the design but does not prove it. Part 4 carries the claim.

**4. Matched controls make the claim falsifiable.** All three tasks below
use the same group, n, input distribution, output space, and dependency
window. **At T = 1 all three are the identical task.**

| Task | Target y[i] | Span | Composition depth | Predicted K*(T) |
|---|---|---|---|---|
| **C1 span control** | s[i] · s[i+T] | T+1 | **1** for every T | flat |
| **C2 word control** | s[i] · s[i+1] · … · s[i+T] | T+1 | ⌈log₂(T+1)⌉ = 1, 2, 3, 4 | logarithmic |
| **Main A₅-CA** | F^T(s)[i] | T+1 | **T** = 1, 2, 4, 8 | increasing ~T |
| Main, Z₆₀ | same F, abelian | T+1 | shallow (closed form) | flat / weak |

**Decision rules**, fixed before any runs:
- **Supports the depth claim:** K*(T) or Δ(T) increases with T on the
  **main A₅** task, more than on **C2**, while **C1** stays flat and **Z₆₀**
  shows a weaker effect.
- **Span-type explanation:** C1 shows the same K-dependence as the main
  task.
- **Hardness unrelated to non-solvable composition:** Z₆₀ shows the same
  K-dependence.
- **Inconclusive:** all cells at chance (floor). Rescale with a pilot
  (smaller n, more steps, larger model) before interpreting.

---

## 6. Risks and pre-implementation gates

| Risk | Mitigation / gate |
|---|---|
| Degenerate dynamics (convergence to attractors, short cycles) in A₅ⁿ under F | **Gate G1:** output-token distribution ≈ uniform at every T; no short cycles on sampled states; sensitivity — changing any one of the T+1 window cells changes y[i] for a large fraction of inputs |
| Hidden algebraic collapse at some T | **Gate G2:** for sampled states, F^T is not reproduced by C1/C2-style shallow formulas or by F^(T′), T′ < T |
| Floor effect at T = 8 for the ~211K model | **Pilot P1:** confirm T = 1 learns the table, and chance-normalised token accuracy stays informative. Scale d_model before the full grid; never loosen the task. |
| Exact match too strict at n = 17 | Chance-normalised token accuracy is a co-primary metric |
| Ring wrap plus learned absolute positions (Phase 1 per-position spread) | Per-position metric; positional ablation |
| BPTT instability and residual growth at K = 8 | Report gradient norms; final LayerNorm variant as a named ablation |
| Unequal training confounds (Phase 1) | Fixed step budget, no train-loss early stopping |
| Compute | Grid = 3 tasks × 4 T × 4 K × 3 seeds = 144 runs (plus baselines and the group ladder). The step budget is set by pilot; use GPU runs with deterministic settings recorded. |

**Scaling to ~6–7M parameters.** The task is unchanged. Increase T to
{8, 16} and n to 33, keeping the window from wrapping (n > T+1). The group
stays A₅, so the vocabulary stays 60.

---

## 7. Scope guard

- **Phase 1 stays frozen.** Configs, results, architecture and CA experiment
  are untouched. Phase 2 lives in new files (e.g. `data/group_ca.py`,
  `configs/phase2_*.yaml`).
- **Benchmark before models.** Implementation order after approval:
  generator + independent verifier → gates G1/G2 → pilot P1 → baselines →
  grid.
- **Pre-registration.** τ, the hypotheses, and the decision rules in §5 are
  fixed before any Phase 2 training run.

---

## 8. Amendment 01 — T = 2 partial shortcut (pre-training, approved option A)

Date: 2026-09-15. Recorded **before any model training**. Machine-readable
form: `configs/phase2/amendment_01.yaml`.

### 8.1 Finding

The first gate run (`runs/phase2/gates`, run `phase2_gates-20260915T090022Z`,
verdict **FAIL**) found:
* independent verification: pass;
* G1: pass at every T;
* G2: pass at T = 4 and T = 8;
* **G2 at T = 2: fail.** The best shallow candidate `s[i]·s[i+2]` agrees with
  F² on 26.65 % of tokens, against a pre-registered limit of 5 % and chance of
  1.67 %. It never agrees on a whole sequence.

Mechanism (verified exactly on sampled states):

```text
F²(s)[i] = s[i] · s[i+1]² · s[i+2]
         = s[i] · s[i+2]      ⇔   s[i+1]² = e
```

In A₅, 16 of 60 elements square to the identity (the identity plus 15
involutions). So on 16/60 ≈ 26.7 % of tokens the T = 2 target has a depth-1
formula. The §5 argument, "a word of 2^T letters", did not account for these
cancellations. At T = 4 the same candidate family peaks at 3.57 % (below the
limit); at T = 8 it is 1.73 % (≈ chance).

### 8.2 What is unchanged

* The group (A₅), n = 17, F(s)[i] = s[i]·s[i+1], and T ∈ {1, 2, 4, 8}.
* The tasks, splits, seeds, K values, metrics and τ.
* Every G1/G2 threshold, including G2's 5 % limit. G2 is **not** loosened.
* `configs/phase2/benchmark.yaml` is not edited.
* The original failing gate run is preserved as recorded. The amended re-run
  writes to `runs/phase2/gates_amendment01` and must reproduce the original
  raw gate numbers exactly.

### 8.3 What changes

| T | Role | Used in the primary depth inference? |
|---|---|---|
| 1 | anchor (the primitive; all three tasks identical) | no (reported) |
| 2 | **diagnostic**: known partial depth-1 shortcut | **no** (reported with its shortcut baseline) |
| 4 | primary depth condition | yes |
| 8 | primary depth condition | yes |

* **New baseline.** `s[i]·s[i+2]` is evaluated on the T = 2 test split (main
  task) alongside chance, copy and oracle. It is reported together with the
  rate of the condition s[i+1]² = e, and a check that agreement occurs exactly
  when the condition holds.
* **Primary inference** (K*(T) trend, Δ(T) trend, interaction regression, and
  the §5 decision rules) uses only T ∈ {4, 8}. T = 1 and T = 2 are reported
  separately and labelled with their role.

### 8.4 Amended gate verdict

The raw gate verdict stays **FAIL** and is reported. The amended verdict,
`PASS_WITH_AMENDMENT_01`, is granted only if **all** of the following hold:

1. independent verification passes;
2. G1 (uniformity, sensitivity, cycles) passes at every T, and the Z₂ self-test detects collapse;
3. G2 passes at every primary depth T (4 and 8);
4. every G2 failure is at a diagnostic T, and matches its documented shortcut: same candidate formula, and token agreement within ±0.01 of 16/60;
5. the mechanism is verified exactly (agreement ⇔ s[i+1]² = e);
6. the re-run's raw gate results equal the original run's, and the benchmark definition equals the original's;
7. the oracle scores 100 % on every test split, and the T = 2 shortcut baseline is reported;
8. no pilot or grid training run exists yet.

Pilot P1 and the grid accept either `PASS` or `PASS_WITH_AMENDMENT_01`.

### 8.5 Consequence for claims

Evidence that "K becomes more useful as required depth grows" may rest only on
T = 4 and T = 8, compared against the C1/C2 controls, the Z₆₀ ladder and the
baselines. A T = 2 result is interpreted against the `s[i]·s[i+2]` baseline and
never counted as a depth condition.

---

## 9. P1b — capacity / learnability diagnostic (approved; recorded before execution)

Date: 2026-09-15. Machine-readable form: `configs/phase2/capacity_diagnostic_p1b.yaml`.

### 9.1 Why

Pilot P1 (`runs/phase2/pilot_p1`, verdict `COMPLETED_FLOOR_HIT`) calibrated a
1500-step budget on T = 1 / K = 1, which reached the criterion in 750 steps.
At that budget, T = 8 / K = 1 and T = 8 / K = 8 both hit the pre-registered
floor: chance-normalised validation token accuracy 0.0005 and 0.0004. T = 1 /
K = 8 reached 99.95 % exact match. Validation loss at T = 8 / K = 8 was still
above chance level (ln 60) and falling when the budget ran out. P1 therefore
cannot distinguish insufficient **capacity** from insufficient **training
budget**.

### 9.2 Design: balanced 2 × 2

| | 20 000 steps |
|---|---|
| `prototype_small` (222,140 params) | T = 8, K = 1 and K = 8 |
| `prototype_medium` (837,436 params) | T = 8, K = 1 and K = 8 |

* `prototype_medium` (`configs/phase2/prototype_medium.yaml`) differs from
  `prototype_small` **only in width**: d_model 128 → 256, heads 4 → 8 (head
  dim stays 32), d_ff 512 → 1024. The pre-LN block, custom attention, α = 0.9,
  dropout 0, learned positions, recursion, vocabulary (60) and max_seq_len 64
  are unchanged. `prototype_small` (Phase 1) is not modified.
* Everything else is identical to P1:
  * data and benchmark: the A₅ main task, n = 17, T = 8, data seed, validation split;
  * run settings: model seed 0, AdamW, lr 1e-3, weight decay 0.01, batch 128, validation every 250 steps.
* Every run lasts a fixed 20,000 steps, with no early stopping.
* Only the validation split is used; the test split is never evaluated.
* The small-model 20k runs remove the capacity/budget confound. P1's 1500-step
  runs are kept, read-only, as the original references.

### 9.3 Pre-declared reading rules

* **Floor:** chance-normalised validation token accuracy < 0.02 (unchanged).
  It is reported for both the best checkpoint (selected on validation) and the
  final checkpoint.
* **Learnable at this budget:** final-checkpoint chance-normalised accuracy ≥
  the floor, read together with the validation-loss curve (below ln 60 and
  falling). A single best checkpoint at chance level is noise.
* **K = 8 vs K = 1 at equal width:** the parameter count is identical, so any
  difference reflects recursion (compute), not parameters.
* **Across widths:** the comparison is descriptive only, reported with
  parameter count and forward FLOPs per sequence.
* **Single seed.** Differences are descriptive, not statistical claims.

### 9.4 Scope

P1b is a capacity/learnability diagnostic only. It is **not** evidence for or
against the Tesseract depth hypothesis: that inference requires the pre-registered
grid on T ∈ {4, 8} with controls and baselines. The P1b results do not change
the benchmark. They only inform the model-size and budget decision for the
grid, which is taken after review.

---

## 10. Phase 2 experiment and Amendment 02 — **DRAFT, NOT FROZEN**

Status: **draft / pre-freeze.** Machine-readable forms:
`configs/phase2/amendment_02_draft.yaml` (`status: draft`, `frozen: false`) and
`configs/phase2/phase2_experiment.yaml`. The model width (`model_base`) and the
fixed step budget (`max_steps`) are **PENDING_P1B**. The runner refuses to
train until both are set, Amendment 02 is frozen, the benchmark integrity check
passes, P1b has completed, and each stage is explicitly confirmed.

### 10.1 Amendment 02 decisions (approved; recorded as draft)

* **D1 — Z₆₀ ladder dropped.** It is removed from the experiment. The §5
  decision rule is amended (test chance-normalised token accuracy,
  best-validation checkpoint, 3 seeds, 95 % t-intervals, T ∈ {4, 8} only):
  * **Supports a sequential-depth effect** only if all three hold:
    1. main Δ(8) = K8 − K1 is > 0 with an interval excluding 0;
    2. main DiD = Δ(8) − Δ(4) is > 0 with an interval excluding 0;
    3. the main task's mean DiD exceeds C2's, and neither C1 Δ(4) nor C1 Δ(8)
       has an interval excluding 0.
  * **Span / receptive-field explanation:** a C1 Δ interval excludes 0.
  * **Composition without sequential depth:** C2's DiD is at least the main task's.
  * **Inconclusive:** the main task is at the floor for every K at T = 8, or K = 1
    is at the ceiling at T = 4, or the result is mixed.
  * The baselines are interpretive only.
* **D2 — per-family learning-rate sweep dropped.** AdamW, lr 1e-3, weight decay
  0.01 for every family, with no tuning. Baseline comparisons are conditional on
  this shared learning rate.
* **D3 — parameter-matched head-dimension confound accepted.** Heads stay fixed
  and the head dimension shrinks. It is recorded per run and reported in every
  baseline table.
* **D4 — deviations from §4:**
  * the budget comes from P1b, not P1;
  * controls and baselines run only at T ∈ {4, 8};
  * unrolled / param-matched L = 1 are not run (identical to Tesseract K = 1);
  * the test split is evaluated after training on the best-validation
    checkpoint (primary) and the final checkpoint (sensitivity), never during
    training;
  * lookup coverage is reported analytically;
  * T = 1 (anchor) and T = 2 (diagnostic) are excluded from the primary
    inference.
* **D5 — P1b width/budget selection rules (declared before P1b completed).**
  * *Learnable:* some K has final chance-normalised accuracy ≥ 0.02, a mean over
    the last 20 evaluations ≥ 0.02, and final validation loss < ln 60.
  * *Width:* the smallest learnable width, stating whether K = 1 is at the floor.
  * *Budget:* the smallest multiple of 5,000 steps at which the 8-evaluation
    rolling mean reaches 90 % of its final value, or 20,000 marked
    "not converged" if still rising.
  * *Neither width learnable:* do not proceed.

### 10.2 Matrix (138 runs, seeds {0, 1, 2})

| Stage | Family / task | T | K or L | Runs |
|---|---|---|---|---|
| t4 | Tesseract main | 4 | 1, 2, 4, 8 | 12 |
| t8 | Tesseract main | 8 | 1, 2, 4, 8 | 12 |
| anchors | Tesseract main (anchor / diagnostic) | 1, 2 | 1, 2, 4, 8 | 24 |
| controls | Tesseract C1 span, C2 word | 4, 8 | 1, 2, 4, 8 | 48 |
| unrolled | non-shared | 4, 8 | 2, 4, 8 | 18 |
| param_matched | parameter-matched | 4, 8 | 2, 4, 8 | 18 |
| width_scaled | K = 1, FLOPs ≈ Tesseract K = 8 | 4, 8 | (8) | 6 |

### 10.3 Stage t4 review (declared in advance)

This review uses best-checkpoint **validation** chance-normalised accuracy.

* **Floor:** every K on all seeds < 0.02 → stop and diagnose.
* **Ceiling:** K = 1 ≥ 0.9 on all seeds → report, and continue only with approval.
* **Otherwise:** stop for review before stage t8.

Stage summaries contain validation metrics only. Test metrics are analysed
only once every planned cell is complete.
