from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

import yaml

LONG_FORECAST_HORIZONS = ("3y", "5y")
LONG_METRIC_FAMILIES = ("fcf_per_share", "eps_per_share")
LONG_SCENARIOS = ("bear", "base", "bull")


@dataclass(frozen=True, slots=True)
class LongEligibilityConfig:
    maximum_metric_age_days: int
    minimum_share_consistency_periods: int
    share_consistency_relative_tolerance: float
    balance_sheet_date_tolerance_days: int
    fcf_failure_blocks_eps: bool
    unsupported_sic_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class LongMetricFamilyConfig:
    source_metric: str
    minimum_annual_periods: int
    minimum_current_value: float
    multiple_minimum: float
    multiple_maximum: float


@dataclass(frozen=True, slots=True)
class LongGrowthConfig:
    historical_recency_decay: float
    historical_minimum: float
    historical_maximum: float
    tax_rate_minimum: float
    tax_rate_maximum: float
    roic_minimum: float
    roic_maximum: float
    reinvestment_minimum: float
    reinvestment_maximum: float
    sustainable_minimum: float
    sustainable_maximum: float
    peer_minimum: float
    peer_maximum: float
    historical_weight: float
    sustainable_weight: float
    peer_weight: float
    initial_growth_minimum: float
    initial_growth_maximum: float
    terminal_growth: float


@dataclass(frozen=True, slots=True)
class LongPeerConfig:
    sic_prefix_levels: tuple[int, ...]
    minimum_peers: dict[int, int]


@dataclass(frozen=True, slots=True)
class LongHorizonConfig:
    years: int
    fade: tuple[float, ...]
    multiple_reversion: float


@dataclass(frozen=True, slots=True)
class LongScenarioConfig:
    growth_delta: float
    reinvestment_multiplier: float
    peer_multiple_multiplier: float


@dataclass(frozen=True, slots=True)
class LongForecastConfig:
    schema_version: int
    version: str
    enabled_scoring_versions: tuple[str, ...]
    price_provider: str
    fundamentals_provider: str
    return_basis: str
    dividends_included: bool
    probability_positive_enabled: bool
    eligibility: LongEligibilityConfig
    metric_families: dict[str, LongMetricFamilyConfig]
    growth: LongGrowthConfig
    peer: LongPeerConfig
    horizons: dict[str, LongHorizonConfig]
    scenarios: dict[str, LongScenarioConfig]
    adjacent_selected_annual_diluted_share_continuity: bool | None
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> Self:
        schema_version = _positive_int(mapping, "schema_version")
        if schema_version != 1:
            raise ValueError("Long forecast config schema_version must be 1")
        return_basis = _required_text(mapping, "return_basis")
        if return_basis != "split_adjusted_price_return":
            raise ValueError("Long forecasts require split_adjusted_price_return")
        dividends_included = mapping.get("dividends_included")
        if dividends_included is not False:
            raise ValueError("Long forecasts must exclude dividends")
        probability_enabled = mapping.get("probability_positive_enabled")
        if probability_enabled is not False:
            raise ValueError("Long forecast probability must remain disabled")

        eligibility_raw = _mapping(mapping, "eligibility")
        unsupported_ranges = tuple(
            _sic_range(value) for value in _list(eligibility_raw, "unsupported_sic_ranges")
        )
        eligibility = LongEligibilityConfig(
            maximum_metric_age_days=_positive_int(
                eligibility_raw,
                "maximum_metric_age_days",
            ),
            minimum_share_consistency_periods=_positive_int(
                eligibility_raw,
                "minimum_share_consistency_periods",
            ),
            share_consistency_relative_tolerance=_bounded_float(
                eligibility_raw,
                "share_consistency_relative_tolerance",
                minimum=0,
                maximum=0.15,
                minimum_inclusive=False,
            ),
            balance_sheet_date_tolerance_days=_nonnegative_int(
                eligibility_raw,
                "balance_sheet_date_tolerance_days",
            ),
            fcf_failure_blocks_eps=_required_bool(
                eligibility_raw,
                "fcf_failure_blocks_eps",
            ),
            unsupported_sic_ranges=unsupported_ranges,
        )

        metric_raw = _mapping(mapping, "metric_families")
        if set(metric_raw) != set(LONG_METRIC_FAMILIES):
            raise ValueError("metric_families must contain exactly fcf_per_share and eps_per_share")
        metric_families: dict[str, LongMetricFamilyConfig] = {}
        for family in LONG_METRIC_FAMILIES:
            raw_family = _mapping(metric_raw, family)
            minimum = _positive_float(raw_family, "multiple_minimum")
            maximum = _positive_float(raw_family, "multiple_maximum")
            if minimum >= maximum:
                raise ValueError(f"{family} multiple bounds must increase")
            metric_families[family] = LongMetricFamilyConfig(
                source_metric=_required_text(raw_family, "source_metric"),
                minimum_annual_periods=_integer_at_least(
                    raw_family,
                    "minimum_annual_periods",
                    minimum=2,
                ),
                minimum_current_value=_positive_float(
                    raw_family,
                    "minimum_current_value",
                ),
                multiple_minimum=minimum,
                multiple_maximum=maximum,
            )
        required_share_periods = max(
            family.minimum_annual_periods for family in metric_families.values()
        )
        if eligibility.minimum_share_consistency_periods < required_share_periods:
            raise ValueError(
                "minimum_share_consistency_periods must cover every selected annual period"
            )

        growth_raw = _mapping(mapping, "growth")
        growth = LongGrowthConfig(
            historical_recency_decay=_bounded_float(
                growth_raw,
                "historical_recency_decay",
                minimum=0,
                maximum=1,
                minimum_inclusive=False,
                maximum_inclusive=False,
            ),
            historical_minimum=_finite_float(growth_raw, "historical_minimum"),
            historical_maximum=_finite_float(growth_raw, "historical_maximum"),
            tax_rate_minimum=_finite_float(growth_raw, "tax_rate_minimum"),
            tax_rate_maximum=_finite_float(growth_raw, "tax_rate_maximum"),
            roic_minimum=_finite_float(growth_raw, "roic_minimum"),
            roic_maximum=_finite_float(growth_raw, "roic_maximum"),
            reinvestment_minimum=_finite_float(growth_raw, "reinvestment_minimum"),
            reinvestment_maximum=_finite_float(growth_raw, "reinvestment_maximum"),
            sustainable_minimum=_finite_float(growth_raw, "sustainable_minimum"),
            sustainable_maximum=_finite_float(growth_raw, "sustainable_maximum"),
            peer_minimum=_finite_float(growth_raw, "peer_minimum"),
            peer_maximum=_finite_float(growth_raw, "peer_maximum"),
            historical_weight=_positive_float(growth_raw, "historical_weight"),
            sustainable_weight=_positive_float(growth_raw, "sustainable_weight"),
            peer_weight=_positive_float(growth_raw, "peer_weight"),
            initial_growth_minimum=_finite_float(
                growth_raw,
                "initial_growth_minimum",
            ),
            initial_growth_maximum=_finite_float(
                growth_raw,
                "initial_growth_maximum",
            ),
            terminal_growth=_finite_float(growth_raw, "terminal_growth"),
        )
        for name, lower, upper in (
            ("historical", growth.historical_minimum, growth.historical_maximum),
            ("tax rate", growth.tax_rate_minimum, growth.tax_rate_maximum),
            ("ROIC", growth.roic_minimum, growth.roic_maximum),
            ("reinvestment", growth.reinvestment_minimum, growth.reinvestment_maximum),
            ("sustainable growth", growth.sustainable_minimum, growth.sustainable_maximum),
            ("peer growth", growth.peer_minimum, growth.peer_maximum),
            (
                "initial growth",
                growth.initial_growth_minimum,
                growth.initial_growth_maximum,
            ),
        ):
            if lower >= upper:
                raise ValueError(f"{name} bounds must increase")
        if not math.isclose(
            growth.historical_weight + growth.sustainable_weight + growth.peer_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Long forecast growth weights must sum to 1")

        peer_raw = _mapping(mapping, "peer")
        levels = tuple(
            _positive_int_value(value, label="sic_prefix_levels")
            for value in _list(peer_raw, "sic_prefix_levels")
        )
        if not levels or len(levels) != len(set(levels)) or any(level > 4 for level in levels):
            raise ValueError("sic_prefix_levels must contain unique values from 1 through 4")
        if tuple(sorted(levels, reverse=True)) != levels:
            raise ValueError("sic_prefix_levels must be ordered from most to least specific")
        floors_raw = _mapping(peer_raw, "minimum_peers")
        floors = {
            int(level): _positive_int_value(value, label=f"minimum_peers[{level}]")
            for level, value in floors_raw.items()
        }
        if set(floors) != set(levels):
            raise ValueError("minimum_peers must define every configured SIC prefix level")
        peer = LongPeerConfig(sic_prefix_levels=levels, minimum_peers=floors)

        horizons_raw = _mapping(mapping, "horizons")
        if set(horizons_raw) != set(LONG_FORECAST_HORIZONS):
            raise ValueError("horizons must contain exactly 3y and 5y")
        expected_years = {"3y": 3, "5y": 5}
        horizons: dict[str, LongHorizonConfig] = {}
        for horizon in LONG_FORECAST_HORIZONS:
            raw_horizon = _mapping(horizons_raw, horizon)
            years = _positive_int(raw_horizon, "years")
            if years != expected_years[horizon]:
                raise ValueError(f"{horizon} years must remain {expected_years[horizon]}")
            fade = tuple(
                _bounded_float_value(
                    value,
                    label=f"{horizon} fade",
                    minimum=0,
                    maximum=1,
                )
                for value in _list(raw_horizon, "fade")
            )
            if len(fade) != years:
                raise ValueError(f"{horizon} fade must contain one value per year")
            if any(left < right for left, right in zip(fade, fade[1:], strict=False)):
                raise ValueError(f"{horizon} fade must not increase")
            horizons[horizon] = LongHorizonConfig(
                years=years,
                fade=fade,
                multiple_reversion=_bounded_float(
                    raw_horizon,
                    "multiple_reversion",
                    minimum=0,
                    maximum=1,
                ),
            )

        scenarios_raw = _mapping(mapping, "scenarios")
        if set(scenarios_raw) != set(LONG_SCENARIOS):
            raise ValueError("scenarios must contain exactly bear, base, and bull")
        scenarios = {
            name: LongScenarioConfig(
                growth_delta=_finite_float(_mapping(scenarios_raw, name), "growth_delta"),
                reinvestment_multiplier=_positive_float(
                    _mapping(scenarios_raw, name),
                    "reinvestment_multiplier",
                ),
                peer_multiple_multiplier=_positive_float(
                    _mapping(scenarios_raw, name),
                    "peer_multiple_multiplier",
                ),
            )
            for name in LONG_SCENARIOS
        }
        if not (
            scenarios["bear"].growth_delta
            <= scenarios["base"].growth_delta
            <= scenarios["bull"].growth_delta
            and scenarios["bear"].reinvestment_multiplier
            <= scenarios["base"].reinvestment_multiplier
            <= scenarios["bull"].reinvestment_multiplier
            and scenarios["bear"].peer_multiple_multiplier
            <= scenarios["base"].peer_multiple_multiplier
            <= scenarios["bull"].peer_multiple_multiplier
        ):
            raise ValueError("Long forecast scenario assumptions must increase bear/base/bull")

        adjacent_selected_annual_diluted_share_continuity = _optional_capability_enabled(
            mapping,
            "adjacent_selected_annual_diluted_share_continuity",
        )

        return cls(
            schema_version=schema_version,
            version=_required_text(mapping, "version"),
            enabled_scoring_versions=_string_tuple(mapping, "enabled_scoring_versions"),
            price_provider=_required_text(mapping, "price_provider"),
            fundamentals_provider=_required_text(mapping, "fundamentals_provider"),
            return_basis=return_basis,
            dividends_included=dividends_included,
            probability_positive_enabled=probability_enabled,
            eligibility=eligibility,
            metric_families=metric_families,
            growth=growth,
            peer=peer,
            horizons=horizons,
            scenarios=scenarios,
            adjacent_selected_annual_diluted_share_continuity=(
                adjacent_selected_annual_diluted_share_continuity
            ),
            raw=mapping,
        )


def default_long_forecast_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "forecasts" / "us-sec-long-v2.yml"


def load_long_forecast_config(path: Path | None = None) -> LongForecastConfig:
    config_path = path or default_long_forecast_config_path()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Long forecast config must be a mapping: {config_path}")
    return LongForecastConfig.from_mapping(cast(dict[str, Any], data))


def long_forecast_config_hash(config: LongForecastConfig) -> str:
    effective = asdict(config)
    effective.pop("raw")
    if effective.get("adjacent_selected_annual_diluted_share_continuity") is None:
        effective.pop("adjacent_selected_annual_diluted_share_continuity")
    payload = json.dumps(
        effective,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return cast(dict[str, Any], value)


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return value


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    values = _list(mapping, key)
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{key} contains a blank value")
    return normalized


def _required_bool(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _optional_capability_enabled(mapping: dict[str, Any], key: str) -> bool | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping with exactly an enabled boolean")
    if set(value) != {"enabled"}:
        raise ValueError(f"{key} must be a mapping with exactly an enabled boolean")
    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(f"{key}.enabled must be boolean")
    return enabled


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    return _positive_int_value(mapping.get(key), label=key)


def _nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _integer_at_least(
    mapping: dict[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _positive_int_value(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(mapping: dict[str, Any], key: str) -> float:
    return _finite_float_value(mapping.get(key), label=key)


def _positive_float(mapping: dict[str, Any], key: str) -> float:
    value = _finite_float(mapping, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _bounded_float(
    mapping: dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    return _bounded_float_value(
        mapping.get(key),
        label=key,
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
    )


def _bounded_float_value(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    normalized = _finite_float_value(value, label=label)
    lower_valid = normalized >= minimum if minimum_inclusive else normalized > minimum
    upper_valid = normalized <= maximum if maximum_inclusive else normalized < maximum
    if not lower_valid or not upper_valid:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        raise ValueError(f"{label} must be in {left}{minimum}, {maximum}{right}")
    return normalized


def _finite_float_value(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _sic_range(value: object) -> tuple[int, int]:
    if not isinstance(value, str) or "-" not in value:
        raise ValueError("unsupported_sic_ranges entries must use NNNN-NNNN")
    raw_start, raw_end = value.split("-", maxsplit=1)
    if len(raw_start) != 4 or len(raw_end) != 4 or not raw_start.isdigit() or not raw_end.isdigit():
        raise ValueError("unsupported_sic_ranges entries must use NNNN-NNNN")
    start = int(raw_start)
    end = int(raw_end)
    if start > end:
        raise ValueError("unsupported_sic_ranges must increase")
    return start, end
