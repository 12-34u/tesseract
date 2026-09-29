#!/usr/bin/env python3
"""Apply the Addendum 03 amended classification to every Batch 01 arm and probe.

Post-hoc and read-only with respect to the pipeline: the arms classify
themselves as pre-registered while they run, and this re-derives the
classification afterwards from their recorded artifacts. Applying it in one
pass is what makes it uniform — no arm is measured by a different rule than any
other, including arms that had already reported when the addendum was written.

Distinct predicted tokens are recomputed from each cell's best checkpoint, so
the collapse check reaches arms whose reports predate the addendum.

CPU only; it never touches the GPU jobs.

    python scripts/phase2_reclassify.py [--output runs/phase2/batch01/amended_classification.json]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from phase2.batch01 import apply_final_layer_norm, apply_per_step_norm  # noqa: E402
from phase2.config import ModelVariant, load_benchmark, resolve_model_variant  # noqa: E402
from phase2.data import SplitStore  # noqa: E402
from phase2.models import build_model  # noqa: E402
from phase2.reclassify import (  # noqa: E402
    amend_report,
    distinct_predicted_tokens,
    majority_class_percent,
    moved_under_arm5_criteria,
)
from phase2.runner import splits_dir  # noqa: E402
from phase2.training import predict_with_loss  # noqa: E402
from utils.run_artifacts import write_json  # noqa: E402
from utils.seed import set_seed  # noqa: E402

DEVICE = torch.device("cpu")
WIDTH = ModelVariant(name="medium", base="phase2/prototype_medium")

#: Every source of a pre-registered classification, with the depth it ran at and
#: the residual fix it used (needed to rebuild the model for the collapse check).
SOURCES = [
    ("probe_a", "runs/phase2/probe_a/probe_report.json", 4, "none"),
    ("probe_b", "runs/phase2/probe_b/probe_report.json", 8, "none"),
    ("arm2b", "runs/phase2/batch01/arm2b_residual_fix*/arm2b_residual_fix_report.json", 4, "output"),
    ("arm3", "runs/phase2/batch01/arm3_curriculum*/arm3_curriculum_report.json", 4, "none"),
    ("arm4", "runs/phase2/batch01/arm4_warmup*/arm4_warmup_report.json", 4, "none"),
    ("arm2c", "runs/phase2/batch01/arm2c_per_step_norm_train*/arm2c_per_step_norm_train_report.json", 4, "per_step"),
]


def resolve_cell_dir(root: Path, k: int, t: int, seed: int) -> Path:
    """The directory holding a cell's checkpoints.

    An interrupted cell was retried into a sibling ``__attemptN`` directory and
    the later attempt is the one that produced the result, so prefer the
    highest-numbered attempt over the original stub.
    """
    base = f"tesseract_d{k}_A5_main_T{t}_s{seed}"
    candidates = sorted(
        (p for p in (root / "cells").glob(base + "*") if (p / "best_checkpoint.pt").is_file()),
        key=lambda p: p.name)
    return candidates[-1] if candidates else root / "cells" / base


def count_distinct(cell_dir: Path, k: int, seed: int, fix: str, cfg, benchmark, split) -> int | None:
    checkpoint = Path(cell_dir) / "best_checkpoint.pt"
    if not checkpoint.is_file():
        return None
    set_seed(seed)
    model = build_model("tesseract", k, cfg, benchmark.n).model
    if fix == "output":
        apply_final_layer_norm(model, 60)
    elif fix == "per_step":
        apply_per_step_norm(model)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    predictions, _ = predict_with_loss(model, split.inputs, split.targets, DEVICE, 512)
    return distinct_predicted_tokens(predictions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/phase2/batch01/amended_classification.json")
    args = parser.parse_args()

    benchmark = load_benchmark()
    cfg = resolve_model_variant(WIDTH, 60)
    store = SplitStore(splits_dir(), benchmark.data_seed)
    splits = {t: store.get(benchmark.group, "main", "val", t, benchmark.n, benchmark.val_size) for t in (4, 8)}
    majority = {t: majority_class_percent(s.targets) for t, s in splits.items()}
    for t, pct in majority.items():
        print(f"majority-class baseline, T={t} validation split: {pct:.4f} %  (uniform chance 1.6667 %)")
    print()

    out = {"majority_class_percent_by_t": majority, "sources": {}}
    for name, pattern, t, fix in SOURCES:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"{name:8s} no report yet — skipped")
            continue
        report = json.loads(Path(matches[-1]).read_text())
        by_k = report.get("classification_by_k", {})

        distinct = {}
        for k_str in by_k:
            k = int(k_str)
            distinct[k_str] = {}
            for row in by_k[k_str].get("per_seed", []):
                seed = row["seed"]
                distinct[k_str][seed] = count_distinct(
                    resolve_cell_dir(Path(matches[-1]).parent, k, t, seed), k, seed, fix, cfg, benchmark, splits[t])

        amended = amend_report(by_k, majority[t], distinct)
        moved = moved_under_arm5_criteria(amended)
        out["sources"][name] = {"report": matches[-1], "t": t, "fix": fix,
                                "amended_by_k": amended, "moves_under_arm5_criteria": moved}

        print(f"=== {name}  (T={t}, fix={fix})  -> moves: {moved}")
        for k_str, a in amended.items():
            flag = "  CHANGED" if a["changed"] else ""
            print(f"   K={k_str}: {a['classification_original']:>14} -> {a['classification_amended']:<14}{flag}")
            for s in a["per_seed"]:
                d = s["distinct_predicted_tokens"]
                print(f"       seed {s['seed']}: acc={s['best_val_token_accuracy']:.4f}%  "
                      f"distinct={d if d is not None else '?':>3}  "
                      f"beats_majority={str(s['beats_majority_class']):>5}  "
                      f"degenerate={str(s['degenerate']):>5}  "
                      f"above: {s['above_chance_original']} -> {s['above_chance_amended']}")
        print()

    write_json(Path(args.output), out)
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
