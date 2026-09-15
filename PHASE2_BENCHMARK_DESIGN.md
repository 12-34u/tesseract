# Phase 2 Benchmark Design Proposal — Sequential Computational Depth

Status: **design for review. Nothing is implemented, and Phase 1 is untouched.**
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
