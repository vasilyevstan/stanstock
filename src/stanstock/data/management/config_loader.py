"""Small YAML config loading helpers shared by data-pipeline commands.

Mirrors the pattern already used by ``stanstock.research.config`` (load a
YAML mapping, hash its canonical JSON form for `config_hash` fields) so
universe/benchmark configs behave consistently with scoring configs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[4] / "config"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def config_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_universe_config_path() -> Path:
    return CONFIG_ROOT / "universes" / "demo_us_europe_synthetic_v1.yaml"


def default_benchmark_config_path() -> Path:
    return CONFIG_ROOT / "benchmarks" / "demo_synthetic_balanced_v1.yaml"


def default_us_universe_config_path() -> Path:
    return CONFIG_ROOT / "universes" / "us_liquid_starter_v1.yaml"


def default_us_scoring_config_path() -> Path:
    return CONFIG_ROOT / "scoring" / "us-price-baseline-v2.yml"


def default_sec_fundamentals_config_path() -> Path:
    return CONFIG_ROOT / "fundamentals" / "us-sec-fundamentals-v1.yml"


def default_sec_cik_mapping_path() -> Path:
    return CONFIG_ROOT / "fundamentals" / "us-sec-cik-v1.yml"
