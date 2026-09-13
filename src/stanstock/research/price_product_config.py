from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal, cast

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken

PRODUCT_VERSION = "research-product-v1"
MOMENTUM_METHOD_VERSION = "us-relative-momentum-v1"
FHS_METHOD_VERSION = "us-price-fhs-v1"
PRODUCT_PAYLOAD_SCHEMA = "research-product@1"
PRODUCT_BENCHMARK_SUBJECT = "SPY"
PRODUCT_CONFIG_FILE_SHA256 = "21dcfcb4a3560fe94a7e614bc8e659d6778312b21398cb249caf6e09df78b832"
PRODUCT_EFFECTIVE_CONFIG_HASH = "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"

_CONFIG_BASENAME = "research-product-v1.yml"
_MALFORMED = "research-product-v1 config is malformed"
_DUPLICATE = "research-product-v1 config forbids duplicate keys"
_YAML_FEATURES = "research-product-v1 config forbids YAML anchors, aliases, and merges"


@dataclass(frozen=True, slots=True)
class MomentumConfig:
    method_version: str
    lookback_sessions: int
    skip_sessions: int
    decision_horizon_sessions: int


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    method_version: str
    return_observations: int
    filter_burn_in: int
    production_paths: int
    diagnostic_max_paths: int
    maximum_horizon_sessions: int
    prng: str
    random_index_order: str
    residual_sampling: str
    variance_target_weight: float
    variance_persistence: float
    innovation_weight: float
    quantile_method: Literal["linear"]
    quantiles: tuple[float, float, float]
    horizons: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ProductRiskConfig:
    annualization_sessions: int
    drawdown_return_sessions: int
    liquidity_sessions: int
    buy_max_relative_volatility: float
    buy_minimum_drawdown: float
    buy_minimum_dollar_turnover: float
    buy_minimum_target_close: float


@dataclass(frozen=True, slots=True)
class ProductRoundingConfig:
    return_decimal_places: int
    price_decimal_places: int
    mode: str


@dataclass(frozen=True, slots=True)
class ProspectiveUniverseConfig:
    core_config: str
    maximum_saved_names: int
    benchmark_excluded_from_membership: bool


@dataclass(frozen=True, slots=True)
class FrozenReplayConfig:
    evidence_label: str
    fixed_epoch: date
    development_end_exclusive: date
    validation_start: date
    validation_end_exclusive: date
    holdout_start: date
    holdout_complete_through: date
    anchor_spacing: str
    purge_partition_crossings: bool
    comparators: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PriceProductConfig:
    schema_version: int
    product_version: str
    payload_schema: str
    calendar: str
    currency: str
    price_provider: str
    benchmark_subject: str
    return_basis: str
    dividends_included: bool
    required_closes: int
    momentum: MomentumConfig
    simulation: SimulationConfig
    risk: ProductRiskConfig
    rounding: ProductRoundingConfig
    universe: ProspectiveUniverseConfig
    replay: FrozenReplayConfig

    @classmethod
    def from_mapping(cls, value: object) -> PriceProductConfig:
        mapping = _mapping(value, "config")
        _exact_keys(
            mapping,
            {
                "schema_version",
                "product_version",
                "payload_schema",
                "calendar",
                "currency",
                "price_provider",
                "benchmark_subject",
                "return_basis",
                "dividends_included",
                "required_closes",
                "momentum",
                "simulation",
                "risk",
                "rounding",
                "universe",
                "replay",
            },
            "config",
        )
        _literal_int(mapping, "schema_version", 1)
        _literal_text(mapping, "product_version", PRODUCT_VERSION)
        _literal_text(mapping, "payload_schema", PRODUCT_PAYLOAD_SCHEMA)
        _literal_text(mapping, "calendar", "XNYS")
        _literal_text(mapping, "currency", "USD")
        _literal_text(mapping, "price_provider", "twelve_data")
        _literal_text(mapping, "benchmark_subject", PRODUCT_BENCHMARK_SUBJECT)
        _literal_text(mapping, "return_basis", "split_adjusted_price_return")
        _literal_bool(mapping, "dividends_included", False)
        _literal_int(mapping, "required_closes", 757)

        momentum_raw = _mapping(mapping["momentum"], "momentum")
        _exact_keys(
            momentum_raw,
            {
                "method_version",
                "lookback_sessions",
                "skip_sessions",
                "decision_horizon_sessions",
            },
            "momentum",
        )
        momentum = MomentumConfig(
            method_version=_literal_text(momentum_raw, "method_version", MOMENTUM_METHOD_VERSION),
            lookback_sessions=_literal_int(momentum_raw, "lookback_sessions", 252),
            skip_sessions=_literal_int(momentum_raw, "skip_sessions", 21),
            decision_horizon_sessions=_literal_int(momentum_raw, "decision_horizon_sessions", 126),
        )

        simulation_raw = _mapping(mapping["simulation"], "simulation")
        _exact_keys(
            simulation_raw,
            {
                "method_version",
                "return_observations",
                "filter_burn_in",
                "production_paths",
                "diagnostic_max_paths",
                "maximum_horizon_sessions",
                "prng",
                "random_index_order",
                "residual_sampling",
                "variance_target_weight",
                "variance_persistence",
                "innovation_weight",
                "quantile_method",
                "quantiles",
                "horizons",
            },
            "simulation",
        )
        quantiles = _number_tuple(simulation_raw["quantiles"], "simulation.quantiles")
        if quantiles != (0.2, 0.5, 0.8):
            raise ValueError("simulation.quantiles must be exactly [0.20, 0.50, 0.80]")
        horizon_raw = _mapping(simulation_raw["horizons"], "simulation.horizons")
        expected_horizons = (("6m", 126), ("12m", 252), ("3y", 756), ("5y", 1260))
        if set(horizon_raw) != {name for name, _sessions in expected_horizons}:
            raise ValueError("simulation.horizons must contain exactly 6m, 12m, 3y, and 5y")
        horizons = tuple(
            (name, _literal_int(horizon_raw, name, sessions))
            for name, sessions in expected_horizons
        )
        simulation = SimulationConfig(
            method_version=_literal_text(simulation_raw, "method_version", FHS_METHOD_VERSION),
            return_observations=_literal_int(simulation_raw, "return_observations", 756),
            filter_burn_in=_literal_int(simulation_raw, "filter_burn_in", 252),
            production_paths=_literal_int(simulation_raw, "production_paths", 8192),
            diagnostic_max_paths=_literal_int(simulation_raw, "diagnostic_max_paths", 16384),
            maximum_horizon_sessions=_literal_int(simulation_raw, "maximum_horizon_sessions", 1260),
            prng=_literal_text(simulation_raw, "prng", "PCG64"),
            random_index_order=_literal_text(simulation_raw, "random_index_order", "path_major"),
            residual_sampling=_literal_text(
                simulation_raw, "residual_sampling", "iid_with_replacement"
            ),
            variance_target_weight=_literal_float(simulation_raw, "variance_target_weight", 0.01),
            variance_persistence=_literal_float(simulation_raw, "variance_persistence", 0.94),
            innovation_weight=_literal_float(simulation_raw, "innovation_weight", 0.05),
            quantile_method="linear",
            quantiles=(0.2, 0.5, 0.8),
            horizons=horizons,
        )
        if not math.isclose(
            simulation.variance_target_weight
            + simulation.variance_persistence
            + simulation.innovation_weight,
            1.0,
            abs_tol=1e-15,
        ):
            raise ValueError("simulation variance coefficients must sum to 1")

        risk_raw = _mapping(mapping["risk"], "risk")
        _exact_keys(
            risk_raw,
            {
                "annualization_sessions",
                "drawdown_return_sessions",
                "liquidity_sessions",
                "buy_max_relative_volatility",
                "buy_minimum_drawdown",
                "buy_minimum_dollar_turnover",
                "buy_minimum_target_close",
            },
            "risk",
        )
        risk = ProductRiskConfig(
            annualization_sessions=_literal_int(risk_raw, "annualization_sessions", 252),
            drawdown_return_sessions=_literal_int(risk_raw, "drawdown_return_sessions", 252),
            liquidity_sessions=_literal_int(risk_raw, "liquidity_sessions", 20),
            buy_max_relative_volatility=_literal_float(
                risk_raw, "buy_max_relative_volatility", 2.0
            ),
            buy_minimum_drawdown=_literal_float(risk_raw, "buy_minimum_drawdown", -0.5),
            buy_minimum_dollar_turnover=_literal_float(
                risk_raw, "buy_minimum_dollar_turnover", 5_000_000.0
            ),
            buy_minimum_target_close=_literal_float(risk_raw, "buy_minimum_target_close", 10.0),
        )

        rounding_raw = _mapping(mapping["rounding"], "rounding")
        _exact_keys(
            rounding_raw,
            {"return_decimal_places", "price_decimal_places", "mode"},
            "rounding",
        )
        rounding = ProductRoundingConfig(
            return_decimal_places=_literal_int(rounding_raw, "return_decimal_places", 4),
            price_decimal_places=_literal_int(rounding_raw, "price_decimal_places", 6),
            mode=_literal_text(rounding_raw, "mode", "ROUND_HALF_EVEN"),
        )

        universe_raw = _mapping(mapping["universe"], "universe")
        _exact_keys(
            universe_raw,
            {
                "core_config",
                "maximum_saved_names",
                "benchmark_excluded_from_membership",
            },
            "universe",
        )
        universe = ProspectiveUniverseConfig(
            core_config=_literal_text(
                universe_raw,
                "core_config",
                "config/universes/us_liquid_starter_v1.yaml",
            ),
            maximum_saved_names=_literal_int(universe_raw, "maximum_saved_names", 20),
            benchmark_excluded_from_membership=_literal_bool(
                universe_raw, "benchmark_excluded_from_membership", True
            ),
        )

        replay_raw = _mapping(mapping["replay"], "replay")
        _exact_keys(
            replay_raw,
            {
                "evidence_label",
                "fixed_epoch",
                "development_end_exclusive",
                "validation_start",
                "validation_end_exclusive",
                "holdout_start",
                "holdout_complete_through",
                "anchor_spacing",
                "purge_partition_crossings",
                "comparators",
            },
            "replay",
        )
        replay = FrozenReplayConfig(
            evidence_label=_literal_text(
                replay_raw,
                "evidence_label",
                "current-universe_current-vintage_retrospective-math-replay",
            ),
            fixed_epoch=_literal_date(replay_raw, "fixed_epoch", date(2019, 9, 3)),
            development_end_exclusive=_literal_date(
                replay_raw, "development_end_exclusive", date(2024, 1, 1)
            ),
            validation_start=_literal_date(replay_raw, "validation_start", date(2024, 1, 1)),
            validation_end_exclusive=_literal_date(
                replay_raw, "validation_end_exclusive", date(2025, 1, 1)
            ),
            holdout_start=_literal_date(replay_raw, "holdout_start", date(2025, 1, 1)),
            holdout_complete_through=_literal_date(
                replay_raw, "holdout_complete_through", date(2026, 9, 11)
            ),
            anchor_spacing=_literal_text(
                replay_raw, "anchor_spacing", "evaluated_horizon_sessions"
            ),
            purge_partition_crossings=_literal_bool(replay_raw, "purge_partition_crossings", True),
            comparators=_text_tuple(replay_raw["comparators"], "replay.comparators"),
        )
        if replay.comparators != (
            "zero_log_drift_gaussian",
            "historical_log_drift_gaussian",
        ):
            raise ValueError("replay.comparators do not match the frozen protocol")

        return cls(
            schema_version=1,
            product_version=PRODUCT_VERSION,
            payload_schema=PRODUCT_PAYLOAD_SCHEMA,
            calendar="XNYS",
            currency="USD",
            price_provider="twelve_data",
            benchmark_subject="SPY",
            return_basis="split_adjusted_price_return",
            dividends_included=False,
            required_closes=757,
            momentum=momentum,
            simulation=simulation,
            risk=risk,
            rounding=rounding,
            universe=universe,
            replay=replay,
        )


def default_price_product_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "scoring" / _CONFIG_BASENAME


def load_price_product_config(path: Path | None = None) -> PriceProductConfig:
    config_path = path or default_price_product_config_path()
    try:
        payload = config_path.read_bytes()
        text = payload.decode("utf-8")
        tokens = list(yaml.scan(text))
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ValueError(_MALFORMED) from None
    if any(isinstance(token, (AnchorToken, AliasToken)) for token in tokens):
        raise ValueError(_YAML_FEATURES)
    if root is None:
        raise ValueError(_MALFORMED)
    _validate_yaml_nodes(root)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        raise ValueError(_MALFORMED) from None
    config = PriceProductConfig.from_mapping(data)
    if path is None:
        if hashlib.sha256(payload).hexdigest() != PRODUCT_CONFIG_FILE_SHA256:
            raise ValueError("research-product-v1 config bytes changed")
        if price_product_config_hash(config) != PRODUCT_EFFECTIVE_CONFIG_HASH:
            raise ValueError("research-product-v1 effective config changed")
    return config


def price_product_config_hash(config: PriceProductConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.isoformat() if isinstance(value, date) else str(value),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_yaml_nodes(node: Node) -> None:
    if isinstance(node, MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if isinstance(key_node, ScalarNode) and (
                key_node.value == "<<" or key_node.tag == "tag:yaml.org,2002:merge"
            ):
                raise ValueError(_YAML_FEATURES)
            if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                raise ValueError(_MALFORMED)
            if key_node.value in seen:
                raise ValueError(_DUPLICATE)
            seen.add(key_node.value)
            _validate_yaml_nodes(value_node)
        return
    if isinstance(node, SequenceNode):
        for child in node.value:
            _validate_yaml_nodes(child)
        return
    if isinstance(node, ScalarNode) and node.tag == "tag:yaml.org,2002:merge":
        raise ValueError(_YAML_FEATURES)


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys")
    return cast(dict[str, object], value)


def _exact_keys(mapping: dict[str, object], expected: set[str], path: str) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{path} keys do not match the reviewed schema")


def _literal_text(mapping: dict[str, object], key: str, expected: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{key} must be {expected!r}")
    return value


def _literal_int(mapping: dict[str, object], key: str, expected: int) -> int:
    value = mapping.get(key)
    if type(value) is not int or value != expected:
        raise ValueError(f"{key} must be integer {expected}")
    return value


def _literal_float(mapping: dict[str, object], key: str, expected: float) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized != expected:
        raise ValueError(f"{key} must be numeric {expected}")
    return normalized


def _literal_bool(mapping: dict[str, object], key: str, expected: bool) -> bool:
    value = mapping.get(key)
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{key} must be boolean {expected}")
    return value


def _literal_date(mapping: dict[str, object], key: str, expected: date) -> date:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must use YYYY-MM-DD text")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{key} must use YYYY-MM-DD text") from None
    if parsed != expected:
        raise ValueError(f"{key} must remain {expected.isoformat()}")
    return parsed


def _number_tuple(value: object, path: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{path} must contain finite numbers")
        normalized = float(item)
        if not math.isfinite(normalized):
            raise ValueError(f"{path} must contain finite numbers")
        result.append(normalized)
    return tuple(result)


def _text_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{path} must be a non-empty text list")
    return tuple(cast(list[str], value))
