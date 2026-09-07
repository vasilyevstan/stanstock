from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Self, cast

import yaml

MEDIUM_FORECAST_HORIZONS = ("6m", "12m")
MATCH_DIMENSIONS = (
    "relative_momentum_bucket",
    "drawdown_bucket",
    "volatility_bucket",
    "market_trend_bucket",
    "market_volatility_bucket",
)


@dataclass(frozen=True, slots=True)
class ForecastFeatureWindows:
    momentum_sessions: int
    drawdown_sessions: int
    volatility_sessions: int
    short_trend_sessions: int
    long_trend_sessions: int
    liquidity_sessions: int

    @property
    def maximum(self) -> int:
        return max(
            self.momentum_sessions,
            self.drawdown_sessions,
            self.volatility_sessions + 1,
            self.short_trend_sessions,
            self.long_trend_sessions,
            self.liquidity_sessions,
        )


@dataclass(frozen=True, slots=True)
class ForecastQuantiles:
    bear: float
    base: float
    bull: float


@dataclass(frozen=True, slots=True)
class ForecastFallback:
    name: str
    dimensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ForecastHorizonConfig:
    sessions: int
    minimum_raw_matches: int
    minimum_effective_cohorts: int
    minimum_distinct_listings: int
    shrinkage_prior_cohorts: float
    probability_minimum_effective_cohorts: int
    probability_minimum_distinct_listings: int
    probability_minimum_calendar_span_days: int
    probability_minimum_distinct_market_regimes: int


@dataclass(frozen=True, slots=True)
class ForecastCalibrationConfig:
    minimum_training_cohorts: int
    minimum_test_cohorts: int
    maximum_baseline_mae_ratio: float
    maximum_brier_score: float


@dataclass(frozen=True, slots=True)
class MediumForecastConfig:
    schema_version: int
    version: str
    enabled_scoring_versions: tuple[str, ...]
    calendar: str
    fixed_epoch: date
    return_basis: str
    dividends_included: bool
    feature_windows: ForecastFeatureWindows
    minimum_history_coverage: float
    minimum_dollar_volume: float
    quantiles: ForecastQuantiles
    bucket_boundaries: dict[str, tuple[float, ...]]
    fallback_order: tuple[ForecastFallback, ...]
    horizons: dict[str, ForecastHorizonConfig]
    calibration: ForecastCalibrationConfig
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> Self:
        schema_version = _positive_int(mapping, "schema_version")
        if schema_version != 1:
            raise ValueError("Medium forecast config schema_version must be 1")
        version = _required_text(mapping, "version")
        enabled = _string_tuple(mapping, "enabled_scoring_versions")
        calendar = _required_text(mapping, "calendar")
        raw_fixed_epoch = mapping.get("fixed_epoch")
        if isinstance(raw_fixed_epoch, date):
            fixed_epoch = raw_fixed_epoch
        elif isinstance(raw_fixed_epoch, str):
            try:
                fixed_epoch = date.fromisoformat(raw_fixed_epoch)
            except ValueError as exc:
                raise ValueError("fixed_epoch must use YYYY-MM-DD") from exc
        else:
            raise ValueError("fixed_epoch must use YYYY-MM-DD")
        return_basis = _required_text(mapping, "return_basis")
        if return_basis != "split_adjusted_price_return":
            raise ValueError("Medium forecasts require split_adjusted_price_return")
        dividends_included = mapping.get("dividends_included")
        if dividends_included is not False:
            raise ValueError("Medium forecasts must exclude dividends")

        raw_windows = _mapping(mapping, "feature_windows")
        windows = ForecastFeatureWindows(
            momentum_sessions=_positive_int(raw_windows, "momentum_sessions"),
            drawdown_sessions=_positive_int(raw_windows, "drawdown_sessions"),
            volatility_sessions=_positive_int(raw_windows, "volatility_sessions"),
            short_trend_sessions=_positive_int(raw_windows, "short_trend_sessions"),
            long_trend_sessions=_positive_int(raw_windows, "long_trend_sessions"),
            liquidity_sessions=_positive_int(raw_windows, "liquidity_sessions"),
        )
        coverage = _finite_float(mapping, "minimum_history_coverage")
        if not 0 < coverage <= 1:
            raise ValueError("minimum_history_coverage must be in (0, 1]")
        minimum_dollar_volume = _finite_float(mapping, "minimum_dollar_volume")
        if minimum_dollar_volume < 0:
            raise ValueError("minimum_dollar_volume cannot be negative")

        raw_quantiles = _mapping(mapping, "quantiles")
        quantiles = ForecastQuantiles(
            bear=_finite_float(raw_quantiles, "bear"),
            base=_finite_float(raw_quantiles, "base"),
            bull=_finite_float(raw_quantiles, "bull"),
        )
        if not 0 < quantiles.bear < quantiles.base < quantiles.bull < 1:
            raise ValueError("Forecast quantiles must increase strictly inside (0, 1)")

        raw_boundaries = _mapping(mapping, "bucket_boundaries")
        expected_boundaries = {
            "relative_momentum",
            "drawdown",
            "volatility",
            "market_trend",
            "market_volatility",
        }
        if set(raw_boundaries) != expected_boundaries:
            raise ValueError(
                "bucket_boundaries must contain exactly " + ", ".join(sorted(expected_boundaries))
            )
        boundaries = {
            name: _strictly_increasing_floats(raw_boundaries[name], name=name)
            for name in sorted(expected_boundaries)
        }

        raw_fallbacks = mapping.get("fallback_order")
        if not isinstance(raw_fallbacks, list) or not raw_fallbacks:
            raise ValueError("fallback_order must be a non-empty list")
        fallbacks: list[ForecastFallback] = []
        for raw_fallback in raw_fallbacks:
            if not isinstance(raw_fallback, dict):
                raise ValueError("Each fallback_order entry must be a mapping")
            name = _required_text(raw_fallback, "name")
            dimensions = _string_tuple(raw_fallback, "dimensions", allow_empty=True)
            if len(dimensions) != len(set(dimensions)):
                raise ValueError(f"Fallback {name!r} repeats matching dimensions")
            unsupported = set(dimensions) - set(MATCH_DIMENSIONS)
            if unsupported:
                raise ValueError(
                    f"Fallback {name!r} contains unsupported dimensions: "
                    + ", ".join(sorted(unsupported))
                )
            fallbacks.append(ForecastFallback(name=name, dimensions=dimensions))
        if len({fallback.name for fallback in fallbacks}) != len(fallbacks):
            raise ValueError("fallback_order names must be unique")
        if fallbacks[-1].dimensions:
            raise ValueError("fallback_order must end with an unconditional empty dimension set")

        raw_horizons = _mapping(mapping, "horizons")
        if set(raw_horizons) != set(MEDIUM_FORECAST_HORIZONS):
            raise ValueError("horizons must contain exactly 6m and 12m")
        expected_sessions = {"6m": 126, "12m": 252}
        horizons: dict[str, ForecastHorizonConfig] = {}
        for horizon in MEDIUM_FORECAST_HORIZONS:
            raw_horizon = _mapping(raw_horizons, horizon)
            horizon_config = ForecastHorizonConfig(
                sessions=_positive_int(raw_horizon, "sessions"),
                minimum_raw_matches=_positive_int(raw_horizon, "minimum_raw_matches"),
                minimum_effective_cohorts=_positive_int(
                    raw_horizon,
                    "minimum_effective_cohorts",
                ),
                minimum_distinct_listings=_positive_int(
                    raw_horizon,
                    "minimum_distinct_listings",
                ),
                shrinkage_prior_cohorts=_positive_float(
                    raw_horizon,
                    "shrinkage_prior_cohorts",
                ),
                probability_minimum_effective_cohorts=_positive_int(
                    raw_horizon,
                    "probability_minimum_effective_cohorts",
                ),
                probability_minimum_distinct_listings=_positive_int(
                    raw_horizon,
                    "probability_minimum_distinct_listings",
                ),
                probability_minimum_calendar_span_days=_positive_int(
                    raw_horizon,
                    "probability_minimum_calendar_span_days",
                ),
                probability_minimum_distinct_market_regimes=_positive_int(
                    raw_horizon,
                    "probability_minimum_distinct_market_regimes",
                ),
            )
            if horizon_config.sessions != expected_sessions[horizon]:
                raise ValueError(f"{horizon} sessions must remain {expected_sessions[horizon]}")
            horizons[horizon] = horizon_config

        raw_calibration = _mapping(mapping, "calibration")
        calibration = ForecastCalibrationConfig(
            minimum_training_cohorts=_positive_int(
                raw_calibration,
                "minimum_training_cohorts",
            ),
            minimum_test_cohorts=_positive_int(
                raw_calibration,
                "minimum_test_cohorts",
            ),
            maximum_baseline_mae_ratio=_positive_float(
                raw_calibration,
                "maximum_baseline_mae_ratio",
            ),
            maximum_brier_score=_finite_float(
                raw_calibration,
                "maximum_brier_score",
            ),
        )
        if calibration.maximum_baseline_mae_ratio < 1:
            raise ValueError("maximum_baseline_mae_ratio must be at least 1")
        if not 0 <= calibration.maximum_brier_score <= 1:
            raise ValueError("maximum_brier_score must be in [0, 1]")

        return cls(
            schema_version=schema_version,
            version=version,
            enabled_scoring_versions=enabled,
            calendar=calendar,
            fixed_epoch=fixed_epoch,
            return_basis=return_basis,
            dividends_included=dividends_included,
            feature_windows=windows,
            minimum_history_coverage=coverage,
            minimum_dollar_volume=minimum_dollar_volume,
            quantiles=quantiles,
            bucket_boundaries=boundaries,
            fallback_order=tuple(fallbacks),
            horizons=horizons,
            calibration=calibration,
            raw=mapping,
        )


def default_medium_forecast_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "forecasts" / "us-price-medium-v1.yml"


def load_medium_forecast_config(path: Path | None = None) -> MediumForecastConfig:
    config_path = path or default_medium_forecast_config_path()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Medium forecast config must be a mapping: {config_path}")
    return MediumForecastConfig.from_mapping(cast(dict[str, Any], data))


def medium_forecast_config_hash(config: MediumForecastConfig) -> str:
    effective_config = asdict(config)
    effective_config.pop("raw")
    payload = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return cast(dict[str, Any], value)


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _string_tuple(
    mapping: dict[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or (not value and not allow_empty):
        suffix = "" if allow_empty else " non-empty"
        raise ValueError(f"{key} must be a{suffix} list")
    normalized = tuple(str(item).strip() for item in value)
    if any(not item for item in normalized):
        raise ValueError(f"{key} contains a blank value")
    return normalized


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _finite_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{key} must be finite")
    return normalized


def _positive_float(mapping: dict[str, Any], key: str) -> float:
    value = _finite_float(mapping, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _strictly_increasing_floats(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Bucket boundaries for {name} must be a non-empty list")
    boundaries: list[float] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Bucket boundary for {name} must be numeric")
        boundary = float(raw)
        if not math.isfinite(boundary):
            raise ValueError(f"Bucket boundary for {name} must be finite")
        boundaries.append(boundary)
    if any(left >= right for left, right in zip(boundaries, boundaries[1:], strict=False)):
        raise ValueError(f"Bucket boundaries for {name} must increase strictly")
    return tuple(boundaries)
