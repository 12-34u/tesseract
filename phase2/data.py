"""Phase 2 data: fixed evaluation splits with SHA-256 checksums, and an
unlimited training stream (PHASE2_BENCHMARK_DESIGN.md §4.10–4.11).

Seeding. Every stream is a ``numpy.random.SeedSequence`` keyed by
(data_seed, split, T, n), plus the model seed for the training stream. So:
* inputs are identical across tasks and groups for the same (split, T, n);
  the controls differ from the main task only in their targets;
* train, val, test and length splits come from independent streams;
* validation and test sets are fixed across model seeds; the training stream
  depends on the model seed and is shared by every K within that seed.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from phase2.groups import get_group
from phase2.tasks import make_targets

SPLITS = {"train": 0, "val": 1, "test": 2, "length": 3}


class ChecksumError(RuntimeError):
    """Evaluation data does not match its recorded SHA-256 checksum."""


def seed_sequence(data_seed: int, split: str, t: int, n: int, model_seed: Optional[int] = None) -> np.random.SeedSequence:
    key = (SPLITS[split], t, n) + (() if model_seed is None else (model_seed,))
    return np.random.SeedSequence(data_seed, spawn_key=key)


def sample_inputs(rng: np.random.Generator, count: int, n: int, order: int) -> np.ndarray:
    return rng.integers(0, order, size=(count, n), dtype=np.int64)


def digest(inputs: np.ndarray, targets: np.ndarray) -> str:
    if inputs.shape != targets.shape:
        raise ValueError("inputs and targets must have the same shape")
    if max(int(inputs.max(initial=0)), int(targets.max(initial=0))) > 255:
        raise ValueError("ids must fit in uint8 for checksumming")
    h = hashlib.sha256()
    h.update(np.asarray(inputs.shape, dtype=np.int64).tobytes())
    h.update(inputs.astype(np.uint8).tobytes())
    h.update(targets.astype(np.uint8).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Split:
    group: str
    task: str
    split: str
    t: int
    n: int
    inputs: np.ndarray  # int64 [size, n]
    targets: np.ndarray  # int64 [size, n]
    sha256: str

    @property
    def key(self) -> str:
        return f"{self.group}/{self.task}/{self.split}/T{self.t}/n{self.n}/N{len(self.inputs)}"


def generate_split(group_name: str, task: str, split: str, t: int, n: int, size: int, data_seed: int) -> Split:
    if split == "train":
        raise ValueError("training data is an unlimited stream; use TrainingStream")
    group = get_group(group_name)
    inputs = sample_inputs(np.random.default_rng(seed_sequence(data_seed, split, t, n)), size, n, group.order)
    targets = make_targets(group, task, inputs, t)
    return Split(group_name, task, split, t, n, inputs, targets, digest(inputs, targets))


class SplitStore:
    """Evaluation splits frozen on disk, guarded by ``manifest.json``.

    Every access regenerates the split from its seed and requires that
    (a) the result matches the checksum in the manifest, and (b) the stored
    file matches too. A changed generator, seed or file therefore fails loudly
    instead of silently producing a different test set.
    """

    def __init__(self, root: Path, data_seed: int) -> None:
        self.root = Path(root)
        self.data_seed = data_seed
        self.manifest_path = self.root / "manifest.json"

    def _manifest(self) -> dict:
        if not self.manifest_path.is_file():
            return {"data_seed": self.data_seed, "splits": {}}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("data_seed") != self.data_seed:
            raise ChecksumError(f"manifest data_seed {manifest.get('data_seed')} != configured {self.data_seed}")
        return manifest

    def get(self, group: str, task: str, split: str, t: int, n: int, size: int) -> Split:
        fresh = generate_split(group, task, split, t, n, size, self.data_seed)
        manifest = self._manifest()
        recorded = manifest["splits"].get(fresh.key)
        if recorded is not None and recorded != fresh.sha256:
            raise ChecksumError(f"{fresh.key}: regenerated checksum {fresh.sha256} != manifest {recorded}")

        path = self.root / (fresh.key.replace("/", "_") + ".npz")
        if path.is_file():
            with np.load(path) as stored:
                stored_digest = digest(stored["inputs"], stored["targets"])
            if stored_digest != fresh.sha256:
                raise ChecksumError(f"{path.name}: stored file checksum {stored_digest} != expected {fresh.sha256}")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, inputs=fresh.inputs.astype(np.uint8), targets=fresh.targets.astype(np.uint8))

        if recorded is None:
            manifest["splits"][fresh.key] = fresh.sha256
            self.root.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return fresh


class TrainingStream:
    """Fresh uniformly random training batches, never repeating an evaluation input."""

    def __init__(self, group_name: str, task: str, t: int, n: int, batch_size: int, data_seed: int,
                 model_seed: int, excluded_inputs: Iterable[np.ndarray] = ()) -> None:
        self.group = get_group(group_name)
        self.task, self.t, self.n, self.batch_size = task, t, n, batch_size
        self.rng = np.random.default_rng(seed_sequence(data_seed, "train", t, n, model_seed))
        self.excluded = set()
        for array in excluded_inputs:
            if array.shape[-1] != n:
                raise ValueError(f"excluded inputs have ring size {array.shape[-1]}, stream uses {n}")
            self.excluded.update(row.astype(np.uint8).tobytes() for row in array)
        self.batches_drawn = 0
        self.rejected_rows = 0

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        inputs = sample_inputs(self.rng, self.batch_size, self.n, self.group.order)
        if self.excluded:
            while True:
                clashes = [r for r in range(self.batch_size) if inputs[r].astype(np.uint8).tobytes() in self.excluded]
                if not clashes:
                    break
                self.rejected_rows += len(clashes)
                inputs[clashes] = sample_inputs(self.rng, len(clashes), self.n, self.group.order)
        self.batches_drawn += 1
        return inputs, make_targets(self.group, self.task, inputs, self.t)


# ============================================================================
# Lookup-coverage diagnostic (§4.13 test 6)
# ============================================================================


def window_codes(inputs: np.ndarray, t: int, order: int) -> np.ndarray:
    """Integer code of every dependency window s[i..i+T] (all rows × all cells)."""
    if order ** (t + 1) >= 2 ** 62:
        raise ValueError("window too large to encode in int64")
    codes = np.zeros(inputs.shape, dtype=np.int64)
    for j in range(t + 1):
        codes += np.roll(inputs.astype(np.int64), -j, axis=-1) * order ** j
    return codes.ravel()


def seen_window_codes(make_stream: Callable[[], TrainingStream], num_batches: int, chunk: int = 256) -> np.ndarray:
    """Unique window codes over the first ``num_batches`` batches of a replayed stream."""
    stream = make_stream()
    order = stream.group.order
    unique = np.empty(0, dtype=np.int64)
    pending = []
    for b in range(num_batches):
        inputs, _ = stream.next_batch()
        pending.append(window_codes(inputs, stream.t, order))
        if len(pending) == chunk or b == num_batches - 1:
            unique = np.union1d(unique, np.concatenate(pending))
            pending = []
    return unique


def window_coverage(test_inputs: np.ndarray, t: int, order: int, seen_codes: np.ndarray) -> float:
    """Fraction of test windows whose exact pattern occurred in training."""
    return float(np.isin(window_codes(test_inputs, t, order), seen_codes).mean())
