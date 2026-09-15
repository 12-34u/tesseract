"""Shared command-line plumbing for experiment scripts."""

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, Optional, Sequence, TypeVar

import torch

from utils.config import load_config, resolve_config_path
from utils.device import resolve_device
from utils.paths import resolve_run_dir

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class ExperimentSetup(Generic[ConfigT]):
    config: ConfigT
    config_path: Path
    device: torch.device
    run_dir: Path


def parse_setup(
    argv: Optional[Sequence[str]],
    description: str,
    default_config: str,
    config_cls: type[ConfigT],
) -> ExperimentSetup[ConfigT]:
    """Parse ``--config/--device/--output-dir`` and return the resolved setup.

    CLI overrides are written back into the config object, so the saved
    ``config.yaml`` records what was actually requested.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", default=default_config, help=f"YAML path or name in configs/ (default: {default_config})"
    )
    parser.add_argument("--device", default=None, help="Override experiment.device: auto | cpu | cuda[:N]")
    parser.add_argument("--output-dir", default=None, help="Override experiment.output_dir")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    config = load_config(config_path, config_cls)

    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if overrides:
        config = replace(config, experiment=replace(config.experiment, **overrides))

    return ExperimentSetup(
        config=config,
        config_path=config_path,
        device=resolve_device(config.experiment.device),
        run_dir=resolve_run_dir(config.experiment.output_dir),
    )
