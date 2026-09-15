"""Phase 2 data: seeding independence, checksummed splits, leakage guard, coverage."""

import json

import numpy as np
import pytest

from phase2.data import (
    ChecksumError,
    SplitStore,
    TrainingStream,
    digest,
    generate_split,
    seen_window_codes,
    window_codes,
    window_coverage,
)
from phase2.groups import get_group
from phase2.tasks import make_targets

SEED = 1729


def test_inputs_shared_across_tasks_and_groups():
    a = generate_split("A5", "main", "val", 2, 17, 64, SEED)
    b = generate_split("Z60", "c1_span", "val", 2, 17, 64, SEED)
    assert np.array_equal(a.inputs, b.inputs) and not np.array_equal(a.targets, b.targets)


def test_splits_and_t_values_use_independent_streams():
    val = generate_split("A5", "main", "val", 2, 17, 64, SEED).inputs
    test = generate_split("A5", "main", "test", 2, 17, 64, SEED).inputs
    other_t = generate_split("A5", "main", "val", 4, 17, 64, SEED).inputs
    train = TrainingStream("A5", "main", 2, 17, 64, SEED, model_seed=0).next_batch()[0]
    assert not np.array_equal(val, test) and not np.array_equal(val, other_t) and not np.array_equal(val, train)


def test_targets_are_the_task_targets():
    split = generate_split("A5", "c2_word", "test", 4, 17, 32, SEED)
    assert np.array_equal(split.targets, make_targets(get_group("A5"), "c2_word", split.inputs, 4))
    assert split.inputs.shape == (32, 17) and split.inputs.dtype == np.int64


def test_regeneration_is_deterministic():
    assert generate_split("A5", "main", "test", 8, 17, 100, SEED).sha256 == generate_split("A5", "main", "test", 8, 17, 100, SEED).sha256
    assert generate_split("A5", "main", "test", 8, 17, 100, SEED).sha256 != generate_split("A5", "main", "test", 8, 17, 100, SEED + 1).sha256


def test_train_split_is_not_a_fixed_split():
    with pytest.raises(ValueError):
        generate_split("A5", "main", "train", 1, 17, 10, SEED)


def test_training_stream_seeding():
    first = TrainingStream("A5", "main", 4, 17, 16, SEED, model_seed=0)
    same = TrainingStream("A5", "main", 4, 17, 16, SEED, model_seed=0)
    other_seed = TrainingStream("A5", "main", 4, 17, 16, SEED, model_seed=1)
    batch = first.next_batch()
    assert np.array_equal(batch[0], same.next_batch()[0])  # every K in a seed sees the same stream
    assert not np.array_equal(batch[0], other_seed.next_batch()[0])
    assert not np.array_equal(batch[0], first.next_batch()[0])  # fresh data each step


def test_split_store_writes_and_verifies(tmp_path):
    store = SplitStore(tmp_path, SEED)
    split = store.get("A5", "main", "val", 2, 17, 50)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["splits"][split.key] == split.sha256
    assert store.get("A5", "main", "val", 2, 17, 50).sha256 == split.sha256


def test_split_store_detects_tampered_file(tmp_path):
    store = SplitStore(tmp_path, SEED)
    split = store.get("A5", "main", "val", 2, 17, 50)
    path = next(tmp_path.glob("*.npz"))
    targets = split.targets.astype(np.uint8)
    targets[0, 0] = (targets[0, 0] + 1) % 60
    np.savez_compressed(path, inputs=split.inputs.astype(np.uint8), targets=targets)
    with pytest.raises(ChecksumError, match="stored file"):
        store.get("A5", "main", "val", 2, 17, 50)


def test_split_store_detects_changed_generation(tmp_path):
    store = SplitStore(tmp_path, SEED)
    split = store.get("A5", "main", "val", 2, 17, 50)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["splits"][split.key] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ChecksumError, match="regenerated checksum"):
        store.get("A5", "main", "val", 2, 17, 50)


def test_split_store_rejects_other_data_seed(tmp_path):
    SplitStore(tmp_path, SEED).get("A5", "main", "val", 1, 17, 10)
    with pytest.raises(ChecksumError, match="data_seed"):
        SplitStore(tmp_path, SEED + 1).get("A5", "main", "val", 1, 17, 10)


def test_leakage_guard_rejects_evaluation_inputs():
    unguarded = TrainingStream("A5", "main", 1, 17, 32, SEED, model_seed=0)
    leaked_row = unguarded.next_batch()[0][5:6]
    guarded = TrainingStream("A5", "main", 1, 17, 32, SEED, model_seed=0, excluded_inputs=[leaked_row])
    inputs, targets = guarded.next_batch()
    assert guarded.rejected_rows >= 1
    assert not (inputs == leaked_row).all(axis=1).any()
    assert np.array_equal(targets, make_targets(get_group("A5"), "main", inputs, 1))


def test_leakage_guard_checks_ring_size():
    with pytest.raises(ValueError):
        TrainingStream("A5", "main", 1, 17, 4, SEED, model_seed=0, excluded_inputs=[np.zeros((1, 33), dtype=np.int64)])


def test_digest_covers_shape_inputs_and_targets():
    x = np.zeros((2, 4), dtype=np.int64)
    assert digest(x, x) != digest(x.reshape(4, 2), x.reshape(4, 2))
    assert digest(x, x) != digest(x, x + 1)


def test_window_codes_and_coverage_by_hand():
    inputs = np.array([[1, 2, 3]])
    assert window_codes(inputs, 1, 60).tolist() == [1 + 2 * 60, 2 + 3 * 60, 3 + 1 * 60]
    seen = np.array(sorted(window_codes(inputs, 1, 60)))
    assert window_coverage(inputs, 1, 60, seen) == 1.0
    assert window_coverage(np.array([[4, 5, 6]]), 1, 60, seen) == 0.0


def test_seen_window_codes_replays_stream():
    def make():
        return TrainingStream("A5", "main", 1, 17, 8, SEED, model_seed=0)
    stream = make()
    batches = [stream.next_batch()[0] for _ in range(3)]
    expected = np.unique(np.concatenate([window_codes(b, 1, 60) for b in batches]))
    assert np.array_equal(seen_window_codes(make, 3, chunk=2), expected)
