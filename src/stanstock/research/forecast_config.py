from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Self, cast, overload

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, ScalarToken

MEDIUM_FORECAST_HORIZONS = ("6m", "12m")
MATCH_DIMENSIONS = (
    "relative_momentum_bucket",
    "drawdown_bucket",
    "volatility_bucket",
    "market_trend_bucket",
    "market_volatility_bucket",
)
MEDIUM_V2_VERSION = "us-price-medium-v2"
MEDIUM_V2_LITERAL_SHA256 = "c9a02ad8cc417c48271a80d4ef832d9d32d01f57afd96d624e70fbfad1accc5f"
MEDIUM_V2_EFFECTIVE_CONFIG_HASH = "79b2bcd3e5dd4a67221c4137f0adf6ea908502a10b2c03765c7924fb12999a5b"
_MEDIUM_V2_BASENAME = "us-price-medium-v2.yml"
_MEDIUM_V2_MALFORMED = "us-price-medium-v2 config is malformed"
_MEDIUM_V2_DUPLICATE = "us-price-medium-v2 config forbids duplicate keys"
_MEDIUM_V2_YAML_FEATURES = "us-price-medium-v2 config forbids YAML anchors, aliases, and merges"
_MEDIUM_V2_IDENTITY = "us-price-medium-v2 config identity mismatch"
_MEDIUM_V2_EFFECTIVE_IDENTITY = "us-price-medium-v2 typed effective config identity mismatch"


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
class ForecastWalkForwardConfig:
    minimum_training_cohorts: int
    minimum_test_cohorts: int
    maximum_baseline_mae_ratio: float
    probability_reference: str
    minimum_brier_skill_exclusive: float
    interval_alpha: float


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


@dataclass(frozen=True, slots=True)
class MediumForecastV2Config:
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
    walk_forward: ForecastWalkForwardConfig
    raw: dict[str, Any]


def default_medium_forecast_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "forecasts" / "us-price-medium-v1.yml"


def is_medium_forecast_v2_candidate(path: Path) -> bool:
    """Classify one explicit path without constructing either config type."""
    payload = _read_medium_forecast_config_bytes(path, explicit=True)
    text = _decode_medium_forecast_config(path, payload, explicit=True)
    return _is_medium_v2_candidate(path, text)


@overload
def load_medium_forecast_config(path: None = None) -> MediumForecastConfig: ...


@overload
def load_medium_forecast_config(
    path: Path,
) -> MediumForecastConfig | MediumForecastV2Config: ...


def load_medium_forecast_config(
    path: Path | None = None,
) -> MediumForecastConfig | MediumForecastV2Config:
    config_path = path or default_medium_forecast_config_path()
    payload = _read_medium_forecast_config_bytes(
        config_path,
        explicit=path is not None,
    )
    return _load_medium_forecast_config_bytes(
        config_path,
        payload,
        explicit=path is not None,
    )


def _read_medium_forecast_config_bytes(path: Path, *, explicit: bool) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        if explicit and path.name == _MEDIUM_V2_BASENAME:
            raise ValueError(_MEDIUM_V2_MALFORMED) from None
        raise


def _decode_medium_forecast_config(path: Path, payload: bytes, *, explicit: bool) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        if explicit and path.name == _MEDIUM_V2_BASENAME:
            raise ValueError(_MEDIUM_V2_MALFORMED) from None
        raise


def _load_medium_forecast_config_bytes(
    path: Path,
    payload: bytes,
    *,
    explicit: bool,
) -> MediumForecastConfig | MediumForecastV2Config:
    """Parse one immutable byte snapshot without consulting its path again."""
    text = _decode_medium_forecast_config(path, payload, explicit=explicit)
    if explicit and _is_medium_v2_candidate(path, text):
        return _load_medium_v2_config(payload)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Medium forecast config must be a mapping: {path}")
    return MediumForecastConfig.from_mapping(cast(dict[str, Any], data))


def medium_forecast_config_hash(
    config: MediumForecastConfig | MediumForecastV2Config,
) -> str:
    effective_config = asdict(config)
    effective_config.pop("raw")
    payload = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_medium_v2_candidate(path: Path, text: str) -> bool:
    canonical_name = path.name == _MEDIUM_V2_BASENAME
    tokens: list[object] = []
    try:
        for token in yaml.scan(text):
            tokens.append(token)
    except yaml.YAMLError:
        return (
            canonical_name
            or any(
                isinstance(token, ScalarToken) and token.value == MEDIUM_V2_VERSION
                for token in tokens
            )
            or _partial_root_schema_v2_claim(text)
        )
    if any(isinstance(token, ScalarToken) and token.value == MEDIUM_V2_VERSION for token in tokens):
        return True
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return canonical_name or _partial_root_schema_v2_claim(text)
    return canonical_name or _root_schema_v2_claim(root)


def _root_schema_v2_claim(root: Node | None) -> bool:
    if not isinstance(root, MappingNode):
        return False
    for key_node, value_node in root.value:
        if (
            isinstance(key_node, ScalarNode)
            and key_node.value == "schema_version"
            and key_node.tag == "tag:yaml.org,2002:str"
            and isinstance(value_node, ScalarNode)
            and value_node.tag == "tag:yaml.org,2002:int"
        ):
            try:
                if int(value_node.value) == 2:
                    return True
            except ValueError:
                continue
    return False


def _partial_root_schema_v2_claim(text: str) -> bool:
    """Retain a root integer schema-2 claim made before later malformed YAML."""
    boundaries = [index + 1 for index, character in enumerate(text) if character == "\n"]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    for boundary in boundaries:
        try:
            root = yaml.compose(text[:boundary], Loader=yaml.SafeLoader)
        except yaml.YAMLError:
            continue
        if _root_schema_v2_claim(root):
            return True
    return False


def _load_medium_v2_config(payload: bytes) -> MediumForecastV2Config:
    try:
        text = payload.decode("utf-8")
        tokens = list(yaml.scan(text))
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError):
        raise ValueError(_MEDIUM_V2_MALFORMED) from None
    if any(isinstance(token, (AnchorToken, AliasToken)) for token in tokens):
        raise ValueError(_MEDIUM_V2_YAML_FEATURES)
    if root is None:
        raise ValueError(_MEDIUM_V2_MALFORMED)
    _validate_medium_v2_yaml_nodes(root)
    if len(payload) != 2_311 or hashlib.sha256(payload).hexdigest() != MEDIUM_V2_LITERAL_SHA256:
        raise ValueError(_MEDIUM_V2_IDENTITY)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        raise ValueError(_MEDIUM_V2_MALFORMED) from None
    if not isinstance(data, dict):
        raise ValueError(_MEDIUM_V2_MALFORMED)
    config = _medium_v2_from_mapping(cast(dict[str, Any], data))
    if medium_forecast_config_hash(config) != MEDIUM_V2_EFFECTIVE_CONFIG_HASH:
        raise ValueError(_MEDIUM_V2_EFFECTIVE_IDENTITY)
    return config


def _validate_medium_v2_yaml_nodes(node: Node) -> None:
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if isinstance(key_node, ScalarNode) and (
                key_node.value == "<<" or key_node.tag == "tag:yaml.org,2002:merge"
            ):
                raise ValueError(_MEDIUM_V2_YAML_FEATURES)
            if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                raise ValueError(_MEDIUM_V2_MALFORMED)
            if key_node.value in seen:
                raise ValueError(_MEDIUM_V2_DUPLICATE)
            seen.add(key_node.value)
            _validate_medium_v2_yaml_nodes(value_node)
        return
    if isinstance(node, SequenceNode):
        for child in node.value:
            _validate_medium_v2_yaml_nodes(child)
        return
    if isinstance(node, ScalarNode) and node.tag == "tag:yaml.org,2002:merge":
        raise ValueError(_MEDIUM_V2_YAML_FEATURES)


def _medium_v2_from_mapping(mapping: dict[str, Any]) -> MediumForecastV2Config:
    try:
        _v2_exact_keys(
            mapping,
            {
                "schema_version",
                "version",
                "enabled_scoring_versions",
                "calendar",
                "fixed_epoch",
                "return_basis",
                "dividends_included",
                "feature_windows",
                "minimum_history_coverage",
                "minimum_dollar_volume",
                "quantiles",
                "bucket_boundaries",
                "fallback_order",
                "horizons",
                "walk_forward",
            },
        )
        if _v2_int(mapping, "schema_version") != 2:
            raise ValueError
        if _v2_text(mapping, "version") != MEDIUM_V2_VERSION:
            raise ValueError
        enabled = _v2_string_list(mapping, "enabled_scoring_versions")
        if enabled != ("us-price-baseline-v2",):
            raise ValueError
        if _v2_text(mapping, "calendar") != "XNYS":
            raise ValueError
        fixed_epoch = mapping["fixed_epoch"]
        if type(fixed_epoch) is not date or fixed_epoch != date(2010, 1, 4):
            raise ValueError
        if _v2_text(mapping, "return_basis") != "split_adjusted_price_return":
            raise ValueError
        if mapping["dividends_included"] is not False:
            raise ValueError

        raw_windows = _v2_mapping(mapping, "feature_windows")
        window_names = {
            "momentum_sessions",
            "drawdown_sessions",
            "volatility_sessions",
            "short_trend_sessions",
            "long_trend_sessions",
            "liquidity_sessions",
        }
        _v2_exact_keys(raw_windows, window_names)
        expected_windows = {
            "momentum_sessions": 252,
            "drawdown_sessions": 252,
            "volatility_sessions": 63,
            "short_trend_sessions": 50,
            "long_trend_sessions": 200,
            "liquidity_sessions": 20,
        }
        parsed_windows = {key: _v2_int(raw_windows, key) for key in window_names}
        if parsed_windows != expected_windows:
            raise ValueError
        windows = ForecastFeatureWindows(**parsed_windows)

        coverage = _v2_float(mapping, "minimum_history_coverage")
        minimum_dollar_volume = _v2_float(mapping, "minimum_dollar_volume")
        if coverage != 0.95 or minimum_dollar_volume != 1_000_000.0:
            raise ValueError

        raw_quantiles = _v2_mapping(mapping, "quantiles")
        _v2_exact_keys(raw_quantiles, {"bear", "base", "bull"})
        quantiles = ForecastQuantiles(
            bear=_v2_float(raw_quantiles, "bear"),
            base=_v2_float(raw_quantiles, "base"),
            bull=_v2_float(raw_quantiles, "bull"),
        )
        if quantiles != ForecastQuantiles(bear=0.2, base=0.5, bull=0.8):
            raise ValueError

        raw_boundaries = _v2_mapping(mapping, "bucket_boundaries")
        expected_boundaries = {
            "relative_momentum": (-0.25, -0.05, 0.05, 0.25),
            "drawdown": (-0.4, -0.2, -0.08),
            "volatility": (0.2, 0.35, 0.55),
            "market_trend": (-0.1, 0.0, 0.1),
            "market_volatility": (0.15, 0.25, 0.4),
        }
        _v2_exact_keys(raw_boundaries, set(expected_boundaries))
        boundaries = {key: _v2_float_list(raw_boundaries, key) for key in expected_boundaries}
        if boundaries != expected_boundaries:
            raise ValueError

        expected_fallbacks = (
            ("exact_state", MATCH_DIMENSIONS),
            ("without_market_volatility", MATCH_DIMENSIONS[:-1]),
            ("stock_state", MATCH_DIMENSIONS[:3]),
            ("momentum_drawdown", MATCH_DIMENSIONS[:2]),
            ("relative_momentum", MATCH_DIMENSIONS[:1]),
            ("unconditional", ()),
        )
        raw_fallbacks = mapping["fallback_order"]
        if not isinstance(raw_fallbacks, list) or len(raw_fallbacks) != len(expected_fallbacks):
            raise ValueError
        fallbacks: list[ForecastFallback] = []
        for raw_fallback, (expected_name, expected_dimensions) in zip(
            raw_fallbacks, expected_fallbacks, strict=True
        ):
            if not isinstance(raw_fallback, dict):
                raise ValueError
            typed_fallback = cast(dict[str, Any], raw_fallback)
            _v2_exact_keys(typed_fallback, {"name", "dimensions"})
            name = _v2_text(typed_fallback, "name")
            dimensions = _v2_string_list(typed_fallback, "dimensions", allow_empty=True)
            if (name, dimensions) != (expected_name, expected_dimensions):
                raise ValueError
            fallbacks.append(ForecastFallback(name=name, dimensions=dimensions))

        raw_horizons = _v2_mapping(mapping, "horizons")
        _v2_exact_keys(raw_horizons, set(MEDIUM_FORECAST_HORIZONS))
        horizons: dict[str, ForecastHorizonConfig] = {}
        expected_sessions = {"6m": 126, "12m": 252}
        expected_probability_cohorts = {"6m": 8, "12m": 6}
        expected_spans = {"6m": 1095, "12m": 1460}
        horizon_keys = {
            "sessions",
            "minimum_raw_matches",
            "minimum_effective_cohorts",
            "minimum_distinct_listings",
            "shrinkage_prior_cohorts",
            "probability_minimum_effective_cohorts",
            "probability_minimum_distinct_listings",
            "probability_minimum_calendar_span_days",
            "probability_minimum_distinct_market_regimes",
        }
        for horizon in MEDIUM_FORECAST_HORIZONS:
            raw_horizon = _v2_mapping(raw_horizons, horizon)
            _v2_exact_keys(raw_horizon, horizon_keys)
            horizon_config = ForecastHorizonConfig(
                sessions=_v2_int(raw_horizon, "sessions"),
                minimum_raw_matches=_v2_int(raw_horizon, "minimum_raw_matches"),
                minimum_effective_cohorts=_v2_int(raw_horizon, "minimum_effective_cohorts"),
                minimum_distinct_listings=_v2_int(raw_horizon, "minimum_distinct_listings"),
                shrinkage_prior_cohorts=_v2_float(raw_horizon, "shrinkage_prior_cohorts"),
                probability_minimum_effective_cohorts=_v2_int(
                    raw_horizon, "probability_minimum_effective_cohorts"
                ),
                probability_minimum_distinct_listings=_v2_int(
                    raw_horizon, "probability_minimum_distinct_listings"
                ),
                probability_minimum_calendar_span_days=_v2_int(
                    raw_horizon, "probability_minimum_calendar_span_days"
                ),
                probability_minimum_distinct_market_regimes=_v2_int(
                    raw_horizon, "probability_minimum_distinct_market_regimes"
                ),
            )
            if horizon_config != ForecastHorizonConfig(
                sessions=expected_sessions[horizon],
                minimum_raw_matches=20,
                minimum_effective_cohorts=3,
                minimum_distinct_listings=10,
                shrinkage_prior_cohorts=4.0,
                probability_minimum_effective_cohorts=expected_probability_cohorts[horizon],
                probability_minimum_distinct_listings=30,
                probability_minimum_calendar_span_days=expected_spans[horizon],
                probability_minimum_distinct_market_regimes=3,
            ):
                raise ValueError
            horizons[horizon] = horizon_config

        raw_walk_forward = _v2_mapping(mapping, "walk_forward")
        walk_keys = {
            "minimum_training_cohorts",
            "minimum_test_cohorts",
            "maximum_baseline_mae_ratio",
            "probability_reference",
            "minimum_brier_skill_exclusive",
            "interval_alpha",
        }
        _v2_exact_keys(raw_walk_forward, walk_keys)
        walk_forward = ForecastWalkForwardConfig(
            minimum_training_cohorts=_v2_int(raw_walk_forward, "minimum_training_cohorts"),
            minimum_test_cohorts=_v2_int(raw_walk_forward, "minimum_test_cohorts"),
            maximum_baseline_mae_ratio=_v2_float(raw_walk_forward, "maximum_baseline_mae_ratio"),
            probability_reference=_v2_text(raw_walk_forward, "probability_reference"),
            minimum_brier_skill_exclusive=_v2_float(
                raw_walk_forward, "minimum_brier_skill_exclusive"
            ),
            interval_alpha=_v2_float(raw_walk_forward, "interval_alpha"),
        )
        if walk_forward != ForecastWalkForwardConfig(
            minimum_training_cohorts=3,
            minimum_test_cohorts=4,
            maximum_baseline_mae_ratio=1.0,
            probability_reference="prequential_unconditional",
            minimum_brier_skill_exclusive=0.0,
            interval_alpha=0.4,
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError(_MEDIUM_V2_MALFORMED) from None
    return MediumForecastV2Config(
        schema_version=2,
        version=MEDIUM_V2_VERSION,
        enabled_scoring_versions=enabled,
        calendar="XNYS",
        fixed_epoch=fixed_epoch,
        return_basis="split_adjusted_price_return",
        dividends_included=False,
        feature_windows=windows,
        minimum_history_coverage=coverage,
        minimum_dollar_volume=minimum_dollar_volume,
        quantiles=quantiles,
        bucket_boundaries=boundaries,
        fallback_order=tuple(fallbacks),
        horizons=horizons,
        walk_forward=walk_forward,
        raw=mapping,
    )


def _v2_exact_keys(mapping: dict[str, Any], expected: set[str]) -> None:
    if set(mapping) != expected:
        raise ValueError


def _v2_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError
    return cast(dict[str, Any], value)


def _v2_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise ValueError
    return value


def _v2_string_list(
    mapping: dict[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError
    return tuple(cast(list[str], value))


def _v2_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError
    return value


def _v2_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError
    return normalized


def _v2_float_list(mapping: dict[str, Any], key: str) -> tuple[float, ...]:
    value = mapping[key]
    if not isinstance(value, list) or not value:
        raise ValueError
    return tuple(_v2_float({"value": item}, "value") for item in value)


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
