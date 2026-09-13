from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    MOMENTUM_METHOD_VERSION,
    PRODUCT_CONFIG_FILE_SHA256,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PriceProductConfig,
    default_price_product_config_path,
    load_price_product_config,
    price_product_config_hash,
)


def test_default_price_product_config_has_literal_physical_and_effective_pins() -> None:
    path = default_price_product_config_path()
    config = load_price_product_config()

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "21dcfcb4a3560fe94a7e614bc8e659d6778312b21398cb249caf6e09df78b832"
    )
    assert PRODUCT_CONFIG_FILE_SHA256 == (
        "21dcfcb4a3560fe94a7e614bc8e659d6778312b21398cb249caf6e09df78b832"
    )
    assert price_product_config_hash(config) == (
        "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
    )
    assert PRODUCT_EFFECTIVE_CONFIG_HASH == (
        "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
    )


def test_price_product_config_freezes_methods_paths_and_replay_protocol() -> None:
    config = load_price_product_config()

    assert config.momentum.method_version == MOMENTUM_METHOD_VERSION
    assert (
        config.momentum.lookback_sessions,
        config.momentum.skip_sessions,
        config.momentum.decision_horizon_sessions,
    ) == (252, 21, 126)
    assert config.simulation.method_version == FHS_METHOD_VERSION
    assert config.simulation.production_paths == 8192
    assert config.simulation.diagnostic_max_paths == 16384
    assert config.simulation.horizons == (
        ("6m", 126),
        ("12m", 252),
        ("3y", 756),
        ("5y", 1260),
    )
    assert config.simulation.quantiles == (0.2, 0.5, 0.8)
    assert config.universe.core_config == "config/universes/us_liquid_starter_v1.yaml"
    assert config.universe.maximum_saved_names == 20
    assert config.replay.fixed_epoch.isoformat() == "2019-09-03"
    assert config.replay.development_end_exclusive.isoformat() == "2024-01-01"
    assert config.replay.validation_end_exclusive.isoformat() == "2025-01-01"
    assert config.replay.holdout_complete_through.isoformat() == "2026-09-11"
    assert config.replay.purge_partition_crossings is True


@pytest.mark.parametrize(
    "mutation",
    [
        "\nunknown_key: true\n",
        "\nsimulation:\n  production_paths: 1\n",
    ],
)
def test_price_product_config_rejects_unknown_or_duplicate_content(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = default_price_product_config_path().read_text(encoding="utf-8")
    path = tmp_path / "research-product-v1.yml"
    path.write_text(payload + mutation, encoding="utf-8")

    with pytest.raises(ValueError):
        load_price_product_config(path)


def test_price_product_config_rejects_yaml_aliases(tmp_path: Path) -> None:
    path = tmp_path / "research-product-v1.yml"
    path.write_text("schema_version: &schema 1\ncopy: *schema\n", encoding="utf-8")

    with pytest.raises(ValueError, match="anchors, aliases"):
        load_price_product_config(path)


@pytest.mark.parametrize("method", ["nearest", None, 1, True, ["linear"]])
def test_quantile_convention_is_validated_in_mappings_and_explicit_files(tmp_path, method):
    mapping = yaml.safe_load(default_price_product_config_path().read_text())
    mapping["simulation"]["quantile_method"] = method
    with pytest.raises(ValueError, match="quantile_method"):
        PriceProductConfig.from_mapping(mapping)
    path = tmp_path / "candidate.yml"
    path.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError, match="quantile_method"):
        load_price_product_config(path)
