"""Batch 02's T = 3 probe machinery (PHASE2_BATCH_02_PREREG.md).

Three things are pinned here.

The shortcut diagnostic has to be right about the *group*, not just about
accuracy: if `shortcut_prediction` did not reproduce the candidate that fails
G2 at T = 3, the probe could not tell depth-3 composition from the shortcut and
a 26.6 % result would be read as learnability.

The probe has to be Arm 3 with one thing changed. Anything else that drifted
would make a T = 3 / T = 4 comparison uninterpretable, so the configuration is
compared against Arm 3's committed values field by field.

The stage-boundary measurement has to actually fire at the boundaries, because
Batch 01's recorded deviation was precisely that this measurement was taken
once at the end instead.
"""

import numpy as np
import pytest
import torch

from phase2.automaton import iterate
from phase2.batch01 import ARM_CONFIG, CurriculumStream
from phase2.batch02 import (
    COMPOSITION,
    DIAGNOSTIC_T_VALUES,
    PROBE_CONFIG,
    SHORTCUT,
    SHORTCUT_CANDIDATE,
    SHORTCUT_TOKEN_AGREEMENT_A5,
    UNRESOLVED,
    attribute_classification,
    shortcut_agreement_percent,
    shortcut_attribution,
    shortcut_prediction,
    stage_boundary_steps,
)
from phase2.config import TrainingConfig, load_benchmark
from phase2.data import SplitStore, TrainingStream, generate_split
from phase2.gates import shallow_candidates
from phase2.groups import get_group
from phase2.models import build_model
from phase2.runner import splits_dir
from phase2.training import train_fixed_steps
from utils.config import ModelConfig, OptimizerConfig

VOCAB = 60


@pytest.fixture(scope="module")
def states():
    return np.random.default_rng(11).integers(0, 60, size=(2000, 17), dtype=np.int64)


# ---------------------------------------------------------------------------
# The shortcut is the candidate the gate found, not something adjacent to it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [2, 3])
def test_shortcut_prediction_is_the_candidate_that_fails_g2(states, t):
    group = get_group("A5")
    expected = shallow_candidates(group, states, t)[SHORTCUT_CANDIDATE[t]]
    assert np.array_equal(shortcut_prediction(group, states, t), expected)


@pytest.mark.parametrize("t", [2, 3])
def test_shortcut_agreement_with_the_true_function_is_16_of_60(states, t):
    """Both depths' shortcuts are right on exactly 16/60 of tokens.

    This is the number that makes accuracy useless as a discriminator at T = 3:
    26.67 % clears a LEARNABLE bar set at 6.67 %.
    """
    group = get_group("A5")
    agreement = (shortcut_prediction(group, states, t) == iterate(group, states, t)).mean()
    assert agreement == pytest.approx(SHORTCUT_TOKEN_AGREEMENT_A5[t], abs=0.01)


def test_the_t3_shortcut_is_right_exactly_when_a_local_product_is_an_involution(states):
    """Token-for-token, not on average: the mechanism, not a matching rate.

    F³[i] = s_i s_{i+1}² s_{i+2} s_{i+1} s_{i+2}² s_{i+3} and the candidate is
    s_i s_{i+1} s_{i+2} s_{i+3}; they agree iff (s_{i+1} s_{i+2})² = e. 16 of
    A5's 60 elements square to the identity, which is where 26.67 % comes from.
    """
    group = get_group("A5")
    table = group.table
    identity = next(e for e in range(60) if all(table[e, x] == x for x in range(60)))
    product = table[np.roll(states, -1, axis=1), np.roll(states, -2, axis=1)]
    condition = table[product, product] == identity
    agrees = shortcut_prediction(group, states, 3) == iterate(group, states, 3)
    assert np.array_equal(agrees, condition)


def test_the_t2_shortcut_is_right_exactly_when_the_middle_cell_is_an_involution(states):
    """The same mechanism one depth down, and this one has an independent record.

    Amendment 01 documented the T = 2 G2 failure at 0.26619 token agreement with
    16/60 elements squaring to the identity
    (`runs/phase2/gates_amendment01/amendment_evaluation.json`). Reproducing that
    from the same code that handles T = 3 is what makes the T = 3 mechanism
    claim more than an assertion about one depth.
    """
    group = get_group("A5")
    table = group.table
    identity = next(e for e in range(60) if all(table[e, x] == x for x in range(60)))
    middle = np.roll(states, -1, axis=1)
    condition = table[middle, middle] == identity
    agrees = shortcut_prediction(group, states, 2) == iterate(group, states, 2)
    assert np.array_equal(agrees, condition)
    assert agrees.mean() == pytest.approx(16 / 60, abs=0.01)


def test_a_true_solver_and_a_shortcut_solver_are_far_apart_on_the_diagnostic(states):
    """The two hypotheses differ by ~73 points on agreement, and not at all on it being above the bar."""
    group = get_group("A5")
    truth = iterate(group, states, 3)
    shortcut = shortcut_prediction(group, states, 3)
    assert shortcut_agreement_percent(shortcut, group, states, 3) == pytest.approx(100.0)
    assert shortcut_agreement_percent(truth, group, states, 3) == pytest.approx(26.67, abs=1.0)


def test_shortcut_prediction_refuses_a_depth_with_no_recorded_candidate(states):
    group = get_group("A5")
    for t in (1, 4, 8):
        with pytest.raises(ValueError):
            shortcut_prediction(group, states, t)


# ---------------------------------------------------------------------------
# The pre-registered reading of a LEARNABLE verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agreement,expected", [
    (100.0, SHORTCUT), (95.0, SHORTCUT), (90.0, SHORTCUT),
    (89.9, UNRESOLVED), (70.0, UNRESOLVED), (50.1, UNRESOLVED),
    (50.0, COMPOSITION), (26.7, COMPOSITION), (0.0, COMPOSITION),
    (None, UNRESOLVED),
])
def test_attribution_thresholds_are_the_preregistered_ones(agreement, expected):
    assert shortcut_attribution(agreement) == expected


def test_attribution_cannot_promote_a_seed_the_amended_rule_did_not_count():
    """A refinement of the classification, never a change to it."""
    amended = {"seeds_meeting_learnable_bar_amended": [], "seeds_above_chance_amended": [],
               "per_seed": [{"seed": 0}, {"seed": 1}, {"seed": 2}]}
    result = attribute_classification(amended, {0: 26.0, 1: 26.0, 2: 26.0})
    assert result["attribution"] is None
    assert all(row["attribution"] is None for row in result["per_seed"])


def test_one_composition_seed_is_enough_to_keep_the_verdict_composition():
    amended = {"seeds_meeting_learnable_bar_amended": [0, 1], "seeds_above_chance_amended": [],
               "per_seed": [{"seed": 0}, {"seed": 1}, {"seed": 2}]}
    assert attribute_classification(amended, {0: 99.0, 1: 27.0, 2: 99.0})["attribution"] == COMPOSITION
    assert attribute_classification(amended, {0: 99.0, 1: 99.0, 2: 27.0})["attribution"] == SHORTCUT


# ---------------------------------------------------------------------------
# The probe is Arm 3 with the ladder's top moved from 4 to 3
# ---------------------------------------------------------------------------


def test_probe_matches_arm3_except_for_the_target_depth():
    arm3 = ARM_CONFIG["3"]
    assert PROBE_CONFIG["fix"] == arm3["fix"] == "none"
    assert PROBE_CONFIG["curriculum"] is arm3["curriculum"] is True
    assert PROBE_CONFIG["steps"] == arm3["steps"] == 35000
    assert PROBE_CONFIG["seeds"] == arm3["seeds"] == 3
    assert PROBE_CONFIG["fixed_set"] is arm3["fixed_set"] is None
    # The one change, and the shape of Arm 3's schedule preserved around it.
    assert PROBE_CONFIG["target_t"] == 3
    assert PROBE_CONFIG["curriculum_stages"] == [1, 2, 3]
    assert PROBE_CONFIG["steps_per_stage"] == 5000
    assert PROBE_CONFIG["target_steps"] == 20000


def test_stage_boundaries_match_what_the_stream_will_actually_do():
    stream = CurriculumStream("A5", "main", 17, 8, 1729, 0, [1, 2, 3], 5000, 3, 20000)
    assert [s["ends_at_step"] for s in stream.stage_boundaries()] == stage_boundary_steps([1, 2, 3], 5000, 20000)
    assert stage_boundary_steps([1, 2, 3], 5000, 20000) == [5000, 10000, 15000, 35000]
    assert stage_boundary_steps([1, 2, 3], 5000, 20000)[-1] == PROBE_CONFIG["steps"]


def test_diagnostic_depths_cover_every_stage_of_the_ladder():
    assert set(PROBE_CONFIG["curriculum_stages"]) <= set(DIAGNOSTIC_T_VALUES)
    assert PROBE_CONFIG["target_t"] in DIAGNOSTIC_T_VALUES


# ---------------------------------------------------------------------------
# The measurement fires at the boundaries — Batch 01's deviation, fixed
# ---------------------------------------------------------------------------


TINY_MODEL = ModelConfig(vocab_size=VOCAB, max_seq_len=64, d_model=16, num_heads=2, d_ff=32, alpha=0.9, dropout=0.0)
OPT = OptimizerConfig("adamw", 1e-3, 0.01)


def _tiny_run(max_steps, eval_every, step_hooks=None):
    val = generate_split("A5", "main", "val", 1, 17, 32, 1729)
    torch.manual_seed(0)
    model = build_model("tesseract", 1, TINY_MODEL, 17).model
    stream = TrainingStream("A5", "main", 1, 17, 16, 1729, 0, excluded_inputs=[val.inputs])
    return train_fixed_steps(model, stream, val, None,
                             TrainingConfig(OPT, 16, max_steps, eval_every, 64), torch.device("cpu"), VOCAB,
                             step_hooks=step_hooks, log=lambda _: None)


def test_step_hooks_fire_at_their_steps_and_land_in_the_outcome():
    seen = []

    def hook(step):
        seen.append(step)
        return {"probe": step * 2}

    outcome = _tiny_run(6, 3, step_hooks={2: hook, 5: hook})
    assert seen == [2, 5]
    assert [r["step"] for r in outcome.hook_records] == [2, 5]
    assert [r["probe"] for r in outcome.hook_records] == [4, 10]


def test_a_hook_on_an_evaluation_step_still_fires_once():
    """The boundaries of the real schedule are multiples of the eval interval."""
    seen = []
    outcome = _tiny_run(6, 3, step_hooks={3: lambda step: seen.append(step) or {}, 6: lambda step: seen.append(step) or {}})
    assert seen == [3, 6]
    assert [r["step"] for r in outcome.hook_records] == [3, 6]


def test_no_hooks_means_no_hook_records():
    assert _tiny_run(4, 2).hook_records == []


# ---------------------------------------------------------------------------
# The T = 3 validation split
# ---------------------------------------------------------------------------


def test_t3_validation_split_is_independent_of_the_t3_test_split():
    """T = 3 was an evaluation-only depth, so its val split is new; it must not overlap test."""
    benchmark = load_benchmark()
    store = SplitStore(splits_dir(), benchmark.data_seed)
    val = store.get(benchmark.group, "main", "val", 3, benchmark.n, benchmark.val_size)
    test = store.get(benchmark.group, "main", "test", 3, benchmark.n, benchmark.test_size)
    val_rows = {row.astype(np.uint8).tobytes() for row in val.inputs}
    assert not val_rows & {row.astype(np.uint8).tobytes() for row in test.inputs}
    assert val.sha256 != test.sha256


def test_a_ladder_may_have_per_stage_lengths():
    """Batch 06 needs a 15,000-step T=3 rung beside 5,000-step ones, because
    Batch 03 measured K=1 taking 12,000 steps to pass the T=3 shortcut ceiling.
    An int must keep every earlier caller's uniform ladder unchanged."""
    uneven = CurriculumStream("A5", "main", 17, 8, 1729, 0, [1, 2, 3, 4], [5000, 5000, 15000, 5000], 4, 60000)
    assert [s["ends_at_step"] for s in uneven.stage_boundaries()] == [5000, 10000, 25000, 30000, 90000]
    assert stage_boundary_steps([1, 2, 3, 4], [5000, 5000, 15000, 5000], 60000) == [5000, 10000, 25000, 30000, 90000]
    # An int still means a uniform ladder.
    even = CurriculumStream("A5", "main", 17, 8, 1729, 0, [1, 2, 3], 5000, 3, 20000)
    assert [s["ends_at_step"] for s in even.stage_boundaries()] == [5000, 10000, 15000, 35000]


def test_a_mismatched_length_list_is_refused():
    with pytest.raises(ValueError):
        CurriculumStream("A5", "main", 17, 8, 1729, 0, [1, 2, 3], [5000, 5000], 3, 20000)
