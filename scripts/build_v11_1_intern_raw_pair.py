#!/usr/bin/env python3
"""V11.1 row utilities using percentile-distance CDF soft targets."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping

import pipeline.pair_core as pair_core
import scripts.build_raw_pair as pair_driver
import scripts.build_v7_intern_raw_pair as v7
from pipeline.v11_1_targets import TARGET_CONTRACT, is_decisive, target_for
from pipeline.v11_contract import DEFAULT_REGISTRY
from scripts import build_v11_intern_raw_pair as v11


DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v11_1")
DEFAULT_COMPONENT_ROOT = Path("tmp/v11_1_percentile_distance_cdf_release/components")


def configure_engine(
    task_id: str, registry: Mapping[str, Any], registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    v11.configure_engine(task_id, registry, registry_path)
    pair_core.target_for = target_for
    pair_driver.target_for = target_for
    v7.target_for = target_for
    v7._fix_metadata = functools.partial(
        v11._fix_metadata_v11, target_contract=TARGET_CONTRACT
    )
    v7._is_decisive = is_decisive


__all__ = [
    "DEFAULT_COMPONENT_ROOT", "DEFAULT_OUTPUT_ROOT", "configure_engine",
    "is_decisive", "target_for",
]
