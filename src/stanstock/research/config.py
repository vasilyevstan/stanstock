from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self, cast

import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

COMPONENTS = (
    "quality",
    "growth",
    "valuation",
    "momentum_technical",
    "risk_liquidity",
    "market_sector",
)
SCORE_HORIZONS = ("short", "medium", "long")
# Compatibility alias for external imports; internal scoring code uses the explicit name.
HORIZONS = SCORE_HORIZONS
V3_VERSION = "us-price-baseline-v3"
V3_EFFECTIVE_CONFIG_HASH = "aee8ae47cc82092f2167d778381e55bccc7c393913a997afd1ca391ca182585c"
_LEGACY_VERSIONS = {"default-v1", "us-price-baseline-v1", "us-price-baseline-v2"}
_V3_INSPECTION_MAX_NODES = 1024
_V3_INSPECTION_MAX_EDGES = 4096
_V3_INSPECTION_MAX_DEPTH = 64
_V3_INSPECTION_BUDGET_ERROR = "Scoring config structure exceeds the v3 inspection budget"


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    minimum_component_coverage: float
    score_penalty_rate: float
    score_penalty_floor: float
    confidence_floor: float
    confidence_cap: float


@dataclass(frozen=True, slots=True)
class FreshnessConfig:
    fresh_days: int
    stale_days: int
    stale_score_penalty: float
    stale_confidence_cap: float


@dataclass(frozen=True, slots=True)
class RiskConfig:
    low_max: float
    medium_max: float
    high_max: float


@dataclass(frozen=True, slots=True)
class FactorPolicyConfig:
    macd_indicator: str
    macd_score_low: float
    macd_score_high: float
    abnormal_volume_indicator: str
    liquidity_indicator: str
    liquidity_score_low: float
    liquidity_score_high: float
    strict_finite_inputs: bool


@dataclass(frozen=True, slots=True)
class TransformMapConfig:
    kind: str
    low: float | None = None
    high: float | None = None
    target: float | None = None
    slope: float | None = None
    nonpositive_score: float | None = None

    def as_dict(self) -> dict[str, float | str]:
        if self.kind in {"linear_higher", "linear_lower"}:
            assert self.low is not None and self.high is not None
            return {"kind": self.kind, "low": self.low, "high": self.high}
        assert (
            self.target is not None
            and self.slope is not None
            and self.nonpositive_score is not None
        )
        return {
            "kind": self.kind,
            "target": self.target,
            "slope": self.slope,
            "nonpositive_score": self.nonpositive_score,
        }


@dataclass(frozen=True, slots=True)
class RsiPolicy:
    convention: str
    window_sessions: int


@dataclass(frozen=True, slots=True)
class CommonRiskPolicy:
    sessions: int
    annualization_sessions: int


@dataclass(frozen=True, slots=True)
class BetaRoles:
    factor_score: str
    composite_risk: str


@dataclass(frozen=True, slots=True)
class ShortScoringConfig:
    schema_version: int
    rsi: RsiPolicy
    risk_window: CommonRiskPolicy
    beta_roles: BetaRoles
    factor_maps: Mapping[str, TransformMapConfig]
    risk_penalty_maps: Mapping[str, TransformMapConfig]


@dataclass(frozen=True, slots=True)
class RecommendationConfig:
    buy_min_score: float
    buy_max_risk: float
    buy_min_confidence: float
    buy_min_liquidity_20d: float
    buy_max_bear_downside: dict[str, float]
    avoid_max_score: float
    avoid_min_risk: float
    avoid_max_confidence: float

    @property
    def buy_min_avg_volume_20d(self) -> float:
        """Compatibility alias for historical v1 callers and tests."""
        return self.buy_min_liquidity_20d


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    short_horizon_days: int
    medium_horizon_days: int
    empirical_min_observations: dict[str, int]
    probability_min_samples: dict[str, int]
    long_term_years: int
    long_bear_growth_haircut: float
    long_bull_growth_uplift: float
    long_bear_multiple_compression: float
    long_bull_multiple_expansion: float


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    version: str
    analysis_mode: str
    overall_horizon: str
    supported_horizons: tuple[str, ...]
    windows: tuple[int, ...]
    horizon_weights: dict[str, dict[str, float]]
    component_factor_counts: dict[str, int]
    factor_policy: FactorPolicyConfig
    coverage: CoverageConfig
    freshness: FreshnessConfig
    risk: RiskConfig
    recommendation: RecommendationConfig
    scenarios: ScenarioConfig
    short_scoring: ShortScoringConfig | None
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> Self:
        version = mapping.get("version")
        if version == V3_VERSION or (
            "short_scoring" in mapping and version not in _LEGACY_VERSIONS
        ):
            return cast(Self, _v3_config_from_mapping(mapping))
        analysis_mode = str(mapping.get("analysis_mode") or "full")
        overall_horizon = str(mapping.get("overall_horizon") or "medium")
        if overall_horizon not in SCORE_HORIZONS:
            raise ValueError(f"Unsupported overall_horizon {overall_horizon!r}")
        raw_supported = mapping.get("supported_horizons", list(SCORE_HORIZONS))
        if not isinstance(raw_supported, list) or not raw_supported:
            raise ValueError("supported_horizons must be a non-empty list")
        supported_horizons = tuple(str(value) for value in raw_supported)
        if len(supported_horizons) != len(set(supported_horizons)):
            raise ValueError("supported_horizons contains duplicates")
        if any(horizon not in SCORE_HORIZONS for horizon in supported_horizons):
            raise ValueError(f"supported_horizons must contain only {SCORE_HORIZONS!r}")
        if overall_horizon not in supported_horizons:
            raise ValueError("overall_horizon must be included in supported_horizons")
        weights = cast(dict[str, dict[str, float]], mapping["horizon_weights"])
        for horizon in SCORE_HORIZONS:
            if horizon not in weights:
                raise ValueError(f"Missing horizon weights for {horizon}")
            total = sum(float(weights[horizon].get(component, 0.0)) for component in COMPONENTS)
            if abs(total - 1.0) > 0.0001:
                raise ValueError(f"Weights for {horizon} must sum to 1.0")
        coverage = cast(dict[str, Any], mapping["coverage"])
        freshness = cast(dict[str, Any], mapping["freshness"])
        risk = cast(dict[str, Any], mapping["risk"])
        recommendation = cast(dict[str, Any], mapping["recommendation"])
        scenarios = cast(dict[str, Any], mapping["scenarios"])
        raw_factor_policy = mapping.get("factor_policy", {})
        if not isinstance(raw_factor_policy, dict):
            raise ValueError("factor_policy must be a mapping")
        factor_policy = cast(dict[str, Any], raw_factor_policy)
        macd_indicator = str(factor_policy.get("macd_indicator") or "macd_histogram")
        abnormal_volume_indicator = str(
            factor_policy.get("abnormal_volume_indicator") or "abnormal_volume"
        )
        liquidity_indicator = str(factor_policy.get("liquidity_indicator") or "avg_volume_20d")
        if macd_indicator not in {"macd_histogram", "macd_histogram_pct"}:
            raise ValueError("factor_policy.macd_indicator is unsupported")
        if abnormal_volume_indicator not in {
            "abnormal_volume",
            "abnormal_volume_strict",
        }:
            raise ValueError("factor_policy.abnormal_volume_indicator is unsupported")
        if liquidity_indicator not in {"avg_volume_20d", "avg_dollar_volume_20d"}:
            raise ValueError("factor_policy.liquidity_indicator is unsupported")
        macd_score_low = float(factor_policy.get("macd_score_low", -2.0))
        macd_score_high = float(factor_policy.get("macd_score_high", 2.0))
        liquidity_score_low = float(factor_policy.get("liquidity_score_low", 50_000.0))
        liquidity_score_high = float(factor_policy.get("liquidity_score_high", 2_000_000.0))
        raw_strict_finite_inputs = factor_policy.get("strict_finite_inputs", False)
        if not isinstance(raw_strict_finite_inputs, bool):
            raise ValueError("factor_policy.strict_finite_inputs must be boolean")
        strict_finite_inputs = raw_strict_finite_inputs
        if not all(
            math.isfinite(value)
            for value in (
                macd_score_low,
                macd_score_high,
                liquidity_score_low,
                liquidity_score_high,
            )
        ):
            raise ValueError("factor_policy score bounds must be finite")
        if macd_score_low >= macd_score_high:
            raise ValueError("factor_policy MACD score bounds must increase")
        if liquidity_score_low < 0 or liquidity_score_low >= liquidity_score_high:
            raise ValueError("factor_policy liquidity score bounds must be nonnegative")
        raw_buy_min_liquidity = recommendation.get(
            "buy_min_liquidity_20d",
            recommendation.get("buy_min_avg_volume_20d"),
        )
        if raw_buy_min_liquidity is None:
            raise ValueError(
                "recommendation requires buy_min_liquidity_20d or buy_min_avg_volume_20d"
            )
        buy_min_liquidity = float(raw_buy_min_liquidity)
        if not math.isfinite(buy_min_liquidity) or buy_min_liquidity < 0:
            raise ValueError("recommendation liquidity floor must be finite and nonnegative")
        component_factor_counts = {
            str(key): int(value)
            for key, value in cast(dict[str, int], mapping["component_factor_counts"]).items()
        }
        if any(count <= 0 for count in component_factor_counts.values()):
            raise ValueError("component_factor_counts values must be positive")
        active_components = {
            component
            for horizon in supported_horizons
            for component, weight in weights[horizon].items()
            if float(weight) > 0
        }
        if set(component_factor_counts) != active_components:
            raise ValueError(
                "component_factor_counts must exactly match components with positive "
                "weights in supported_horizons"
            )
        buy_max_bear_downside = {
            str(key): float(value)
            for key, value in cast(
                dict[str, float], recommendation["buy_max_bear_downside"]
            ).items()
        }
        if set(buy_max_bear_downside) != set(supported_horizons):
            raise ValueError("buy_max_bear_downside must exactly match supported_horizons")
        return cls(
            version=str(mapping["version"]),
            analysis_mode=analysis_mode,
            overall_horizon=overall_horizon,
            supported_horizons=supported_horizons,
            windows=tuple(int(window) for window in cast(list[int], mapping["windows"])),
            horizon_weights={
                horizon: {component: float(weights[horizon][component]) for component in COMPONENTS}
                for horizon in SCORE_HORIZONS
            },
            component_factor_counts=component_factor_counts,
            factor_policy=FactorPolicyConfig(
                macd_indicator=macd_indicator,
                macd_score_low=macd_score_low,
                macd_score_high=macd_score_high,
                abnormal_volume_indicator=abnormal_volume_indicator,
                liquidity_indicator=liquidity_indicator,
                liquidity_score_low=liquidity_score_low,
                liquidity_score_high=liquidity_score_high,
                strict_finite_inputs=strict_finite_inputs,
            ),
            coverage=CoverageConfig(
                minimum_component_coverage=float(coverage["minimum_component_coverage"]),
                score_penalty_rate=float(coverage["score_penalty_rate"]),
                score_penalty_floor=float(coverage["score_penalty_floor"]),
                confidence_floor=float(coverage["confidence_floor"]),
                confidence_cap=float(coverage["confidence_cap"]),
            ),
            freshness=FreshnessConfig(
                fresh_days=int(freshness["fresh_days"]),
                stale_days=int(freshness["stale_days"]),
                stale_score_penalty=float(freshness["stale_score_penalty"]),
                stale_confidence_cap=float(freshness["stale_confidence_cap"]),
            ),
            risk=RiskConfig(
                low_max=float(risk["low_max"]),
                medium_max=float(risk["medium_max"]),
                high_max=float(risk["high_max"]),
            ),
            recommendation=RecommendationConfig(
                buy_min_score=float(recommendation["buy_min_score"]),
                buy_max_risk=float(recommendation["buy_max_risk"]),
                buy_min_confidence=float(recommendation["buy_min_confidence"]),
                buy_min_liquidity_20d=buy_min_liquidity,
                buy_max_bear_downside=buy_max_bear_downside,
                avoid_max_score=float(recommendation["avoid_max_score"]),
                avoid_min_risk=float(recommendation["avoid_min_risk"]),
                avoid_max_confidence=float(recommendation["avoid_max_confidence"]),
            ),
            scenarios=ScenarioConfig(
                short_horizon_days=int(scenarios["short_horizon_days"]),
                medium_horizon_days=int(scenarios["medium_horizon_days"]),
                empirical_min_observations={
                    str(key): int(value)
                    for key, value in cast(
                        dict[str, int], scenarios["empirical_min_observations"]
                    ).items()
                },
                probability_min_samples={
                    str(key): int(value)
                    for key, value in cast(
                        dict[str, int], scenarios["probability_min_samples"]
                    ).items()
                },
                long_term_years=int(scenarios["long_term_years"]),
                long_bear_growth_haircut=float(scenarios["long_bear_growth_haircut"]),
                long_bull_growth_uplift=float(scenarios["long_bull_growth_uplift"]),
                long_bear_multiple_compression=float(scenarios["long_bear_multiple_compression"]),
                long_bull_multiple_expansion=float(scenarios["long_bull_multiple_expansion"]),
            ),
            short_scoring=None,
            raw=mapping,
        )


_V3_TOP_LEVEL_KEYS = {
    "version",
    "analysis_mode",
    "overall_horizon",
    "supported_horizons",
    "windows",
    "factor_policy",
    "horizon_weights",
    "component_factor_counts",
    "coverage",
    "freshness",
    "risk",
    "recommendation",
    "scenarios",
    "short_scoring",
}
_V3_FACTOR_KINDS = {
    "momentum.return_20d": "linear_higher",
    "momentum.return_63d": "linear_higher",
    "momentum.return_126d": "linear_higher",
    "momentum.sma_50": "linear_higher",
    "momentum.sma_200": "linear_higher",
    "momentum.rsi": "linear_higher",
    "momentum.macd": "linear_higher",
    "momentum.52w": "linear_higher",
    "risk.annualized_volatility": "linear_lower",
    "risk.downside_volatility": "linear_lower",
    "risk.max_drawdown": "linear_higher",
    "risk.abnormal_volume": "positive_target_penalty",
    "risk.avg_volume": "linear_higher",
    "market.relative_20d": "linear_higher",
    "market.relative_63d": "linear_higher",
    "market.relative_252d": "linear_higher",
}
_V3_RISK_KINDS = {
    "annualized_volatility": "linear_higher",
    "downside_volatility": "linear_higher",
    "max_drawdown_magnitude": "linear_higher",
    "absolute_beta": "linear_higher",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate keys at every mapping level."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._flattening_mappings: set[int] = set()

    def flatten_mapping(self, node: MappingNode) -> None:
        marker = id(node)
        if marker in self._flattening_mappings:
            raise ValueError("Cyclic YAML merge in scoring config")
        self._flattening_mappings.add(marker)
        try:
            super().flatten_mapping(node)
        finally:
            self._flattening_mappings.remove(marker)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("Scoring config mapping keys must be scalar") from error
        if duplicate:
            raise ValueError(f"Duplicate scoring config key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"
_V3_MERGE_ERROR = "YAML merge keys are not supported for us-price-baseline-v3"


def _yaml_children(node: ScalarNode | SequenceNode | MappingNode) -> list[Any]:
    if isinstance(node, MappingNode):
        return [child for pair in node.value for child in pair]
    if isinstance(node, SequenceNode):
        return list(node.value)
    return []


def _root_mapping_claims_v3(node: MappingNode) -> bool:
    """Inspect only the root mapping and its merge closure, without flattening."""
    pending = [node]
    inspected: set[int] = set()
    while pending:
        mapping = pending.pop()
        marker = id(mapping)
        if marker in inspected:
            continue
        inspected.add(marker)
        for key, value in mapping.value:
            if (
                isinstance(key, ScalarNode)
                and key.value == "version"
                and isinstance(value, ScalarNode)
                and value.value == V3_VERSION
            ):
                return True
            if key.tag != _YAML_MERGE_TAG:
                continue
            if isinstance(value, MappingNode):
                pending.append(value)
            elif isinstance(value, SequenceNode):
                pending.extend(
                    item for item in reversed(value.value) if isinstance(item, MappingNode)
                )
    return False


def _yaml_graph_max_depth(
    root_marker: int,
    adjacency: dict[int, tuple[int, ...]],
) -> int:
    """Return a cycle-safe upper bound that is exact for acyclic YAML graphs."""
    finished: list[int] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, 0)]
        while stack:
            marker, child_index = stack[-1]
            children = adjacency[marker]
            if child_index == len(children):
                finished.append(marker)
                stack.pop()
                continue
            child = children[child_index]
            stack[-1] = (marker, child_index + 1)
            if child not in visited:
                visited.add(child)
                stack.append((child, 0))

    reverse_adjacency: dict[int, list[int]] = {marker: [] for marker in adjacency}
    for marker, children in adjacency.items():
        for child in children:
            reverse_adjacency[child].append(marker)

    component_by_node: dict[int, int] = {}
    component_sizes: list[int] = []
    for start in reversed(finished):
        if start in component_by_node:
            continue
        component = len(component_sizes)
        component_by_node[start] = component
        size = 0
        component_stack = [start]
        while component_stack:
            marker = component_stack.pop()
            size += 1
            for parent in reverse_adjacency[marker]:
                if parent not in component_by_node:
                    component_by_node[parent] = component
                    component_stack.append(parent)
        component_sizes.append(size)

    component_children: list[set[int]] = [set() for _component in component_sizes]
    component_indegree = [0 for _component in component_sizes]
    for marker, children in adjacency.items():
        source = component_by_node[marker]
        for child in children:
            target = component_by_node[child]
            if source == target or target in component_children[source]:
                continue
            component_children[source].add(target)
            component_indegree[target] += 1

    root_component = component_by_node[root_marker]
    depths = [0 for _component in component_sizes]
    depths[root_component] = component_sizes[root_component]
    ready = [component for component, indegree in enumerate(component_indegree) if indegree == 0]
    next_ready = 0
    maximum = depths[root_component]
    while next_ready < len(ready):
        component = ready[next_ready]
        next_ready += 1
        for child in component_children[component]:
            if depths[component]:
                depths[child] = max(
                    depths[child],
                    depths[component] + component_sizes[child],
                )
                maximum = max(maximum, depths[child])
            component_indegree[child] -= 1
            if component_indegree[child] == 0:
                ready.append(child)
    return maximum


def _inspect_yaml_structure(text: str) -> tuple[bool, bool]:
    """Compose and bound the YAML node graph before strict construction."""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except RecursionError:
        raise ValueError(_V3_INSPECTION_BUDGET_ERROR) from None
    if root is None:
        return (False, False)

    pending = [root]
    inspected: set[int] = set()
    adjacency: dict[int, tuple[int, ...]] = {}
    node_count = 0
    edge_count = 0
    has_merge_key = False
    while pending:
        node = pending.pop()
        marker = id(node)
        if marker in inspected:
            continue
        inspected.add(marker)
        node_count += 1
        if node_count > _V3_INSPECTION_MAX_NODES:
            raise ValueError(_V3_INSPECTION_BUDGET_ERROR)

        children = _yaml_children(node)
        edge_count += len(children)
        if edge_count > _V3_INSPECTION_MAX_EDGES:
            raise ValueError(_V3_INSPECTION_BUDGET_ERROR)
        adjacency[marker] = tuple(id(child) for child in children)
        if isinstance(node, MappingNode):
            has_merge_key = has_merge_key or any(
                key.tag == _YAML_MERGE_TAG for key, _value in node.value
            )
        pending.extend(reversed(children))

    if _yaml_graph_max_depth(id(root), adjacency) > _V3_INSPECTION_MAX_DEPTH:
        raise ValueError(_V3_INSPECTION_BUDGET_ERROR)

    claims_v3 = isinstance(root, MappingNode) and _root_mapping_claims_v3(root)
    return (claims_v3, has_merge_key)


def _v3_config_from_mapping(raw: dict[str, Any]) -> ScoringConfig:
    mapping = _strict_mapping(raw, "config")
    _require_keys(mapping, _V3_TOP_LEVEL_KEYS, "config")
    _literal(mapping["version"], V3_VERSION, "version")
    _literal(mapping["analysis_mode"], "price_only_baseline", "analysis_mode")
    _literal(mapping["overall_horizon"], "short", "overall_horizon")
    _literal_list(mapping["supported_horizons"], ("short",), "supported_horizons")
    _literal_list(
        mapping["windows"],
        (1, 5, 10, 20, 63, 126, 252, 756, 1260),
        "windows",
        integers=True,
    )

    factor_policy_raw = _strict_mapping(mapping["factor_policy"], "factor_policy")
    _require_keys(
        factor_policy_raw,
        {
            "macd_indicator",
            "abnormal_volume_indicator",
            "liquidity_indicator",
            "strict_finite_inputs",
        },
        "factor_policy",
    )
    _literal(
        factor_policy_raw["macd_indicator"],
        "macd_histogram_pct",
        "factor_policy.macd_indicator",
    )
    _literal(
        factor_policy_raw["abnormal_volume_indicator"],
        "abnormal_volume_strict",
        "factor_policy.abnormal_volume_indicator",
    )
    _literal(
        factor_policy_raw["liquidity_indicator"],
        "avg_dollar_volume_20d",
        "factor_policy.liquidity_indicator",
    )
    if factor_policy_raw["strict_finite_inputs"] is not True:
        raise ValueError("factor_policy.strict_finite_inputs must be true")

    weights_raw = _strict_mapping(mapping["horizon_weights"], "horizon_weights")
    _require_keys(weights_raw, set(SCORE_HORIZONS), "horizon_weights")
    weights: dict[str, dict[str, float]] = {}
    for horizon in SCORE_HORIZONS:
        horizon_raw = _strict_mapping(
            weights_raw[horizon],
            f"horizon_weights.{horizon}",
        )
        _require_keys(horizon_raw, set(COMPONENTS), f"horizon_weights.{horizon}")
        parsed_weights = {
            component: _finite_number(
                horizon_raw[component],
                f"horizon_weights.{horizon}.{component}",
                low=0.0,
                high=1.0,
            )
            for component in COMPONENTS
        }
        if not math.isclose(sum(parsed_weights.values()), 1.0, abs_tol=0.0001):
            raise ValueError(f"Weights for {horizon} must sum to 1.0")
        weights[horizon] = parsed_weights

    counts_raw = _strict_mapping(
        mapping["component_factor_counts"],
        "component_factor_counts",
    )
    expected_counts = {
        "momentum_technical": 8,
        "risk_liquidity": 5,
        "market_sector": 3,
    }
    _require_keys(counts_raw, set(expected_counts), "component_factor_counts")
    for key, expected in expected_counts.items():
        _literal_integer(
            counts_raw[key],
            expected,
            f"component_factor_counts.{key}",
        )

    coverage_raw = _strict_mapping(mapping["coverage"], "coverage")
    _require_keys(
        coverage_raw,
        {
            "minimum_component_coverage",
            "score_penalty_rate",
            "score_penalty_floor",
            "confidence_floor",
            "confidence_cap",
        },
        "coverage",
    )
    coverage = CoverageConfig(
        minimum_component_coverage=_finite_number(
            coverage_raw["minimum_component_coverage"],
            "coverage.minimum_component_coverage",
            low=0.0,
            high=1.0,
        ),
        score_penalty_rate=_finite_number(
            coverage_raw["score_penalty_rate"],
            "coverage.score_penalty_rate",
            low=0.0,
            high=1.0,
        ),
        score_penalty_floor=_finite_number(
            coverage_raw["score_penalty_floor"],
            "coverage.score_penalty_floor",
            low=0.0,
            high=1.0,
        ),
        confidence_floor=_finite_number(
            coverage_raw["confidence_floor"],
            "coverage.confidence_floor",
            low=0.0,
            high=100.0,
        ),
        confidence_cap=_finite_number(
            coverage_raw["confidence_cap"],
            "coverage.confidence_cap",
            low=0.0,
            high=100.0,
        ),
    )
    if coverage.confidence_floor > coverage.confidence_cap:
        raise ValueError("coverage confidence floor cannot exceed its cap")

    freshness_raw = _strict_mapping(mapping["freshness"], "freshness")
    _require_keys(
        freshness_raw,
        {"fresh_days", "stale_days", "stale_score_penalty", "stale_confidence_cap"},
        "freshness",
    )
    freshness = FreshnessConfig(
        fresh_days=_positive_integer(freshness_raw["fresh_days"], "freshness.fresh_days"),
        stale_days=_positive_integer(freshness_raw["stale_days"], "freshness.stale_days"),
        stale_score_penalty=_finite_number(
            freshness_raw["stale_score_penalty"],
            "freshness.stale_score_penalty",
            low=0.0,
            high=1.0,
        ),
        stale_confidence_cap=_finite_number(
            freshness_raw["stale_confidence_cap"],
            "freshness.stale_confidence_cap",
            low=0.0,
            high=100.0,
        ),
    )
    if freshness.fresh_days >= freshness.stale_days:
        raise ValueError("freshness.fresh_days must be less than stale_days")

    risk_raw = _strict_mapping(mapping["risk"], "risk")
    _require_keys(risk_raw, {"low_max", "medium_max", "high_max"}, "risk")
    risk = RiskConfig(
        low_max=_finite_number(risk_raw["low_max"], "risk.low_max", low=0.0, high=100.0),
        medium_max=_finite_number(risk_raw["medium_max"], "risk.medium_max", low=0.0, high=100.0),
        high_max=_finite_number(risk_raw["high_max"], "risk.high_max", low=0.0, high=100.0),
    )
    if not risk.low_max < risk.medium_max < risk.high_max:
        raise ValueError("risk class bounds must increase")

    recommendation_raw = _strict_mapping(mapping["recommendation"], "recommendation")
    _require_keys(
        recommendation_raw,
        {
            "buy_min_score",
            "buy_max_risk",
            "buy_min_confidence",
            "buy_min_liquidity_20d",
            "buy_max_bear_downside",
            "avoid_max_score",
            "avoid_min_risk",
            "avoid_max_confidence",
        },
        "recommendation",
    )
    bear_raw = _strict_mapping(
        recommendation_raw["buy_max_bear_downside"],
        "recommendation.buy_max_bear_downside",
    )
    _require_keys(bear_raw, {"short"}, "recommendation.buy_max_bear_downside")
    recommendation = RecommendationConfig(
        buy_min_score=_score_number(recommendation_raw["buy_min_score"], "buy_min_score"),
        buy_max_risk=_score_number(recommendation_raw["buy_max_risk"], "buy_max_risk"),
        buy_min_confidence=_score_number(
            recommendation_raw["buy_min_confidence"],
            "buy_min_confidence",
        ),
        buy_min_liquidity_20d=_finite_number(
            recommendation_raw["buy_min_liquidity_20d"],
            "recommendation.buy_min_liquidity_20d",
            low=0.0,
        ),
        buy_max_bear_downside={
            "short": _finite_number(
                bear_raw["short"],
                "recommendation.buy_max_bear_downside.short",
                low=-1.0,
                high=0.0,
            )
        },
        avoid_max_score=_score_number(
            recommendation_raw["avoid_max_score"],
            "avoid_max_score",
        ),
        avoid_min_risk=_score_number(
            recommendation_raw["avoid_min_risk"],
            "avoid_min_risk",
        ),
        avoid_max_confidence=_score_number(
            recommendation_raw["avoid_max_confidence"],
            "avoid_max_confidence",
        ),
    )

    scenarios_raw = _strict_mapping(mapping["scenarios"], "scenarios")
    _require_keys(
        scenarios_raw,
        {
            "short_horizon_days",
            "medium_horizon_days",
            "empirical_min_observations",
            "probability_min_samples",
            "long_term_years",
            "long_bear_growth_haircut",
            "long_bull_growth_uplift",
            "long_bear_multiple_compression",
            "long_bull_multiple_expansion",
        },
        "scenarios",
    )
    empirical = _integer_mapping(
        scenarios_raw["empirical_min_observations"],
        "scenarios.empirical_min_observations",
        ("short", "medium"),
    )
    probability = _integer_mapping(
        scenarios_raw["probability_min_samples"],
        "scenarios.probability_min_samples",
        SCORE_HORIZONS,
    )
    scenarios = ScenarioConfig(
        short_horizon_days=_positive_integer(
            scenarios_raw["short_horizon_days"],
            "scenarios.short_horizon_days",
        ),
        medium_horizon_days=_positive_integer(
            scenarios_raw["medium_horizon_days"],
            "scenarios.medium_horizon_days",
        ),
        empirical_min_observations=empirical,
        probability_min_samples=probability,
        long_term_years=_positive_integer(
            scenarios_raw["long_term_years"],
            "scenarios.long_term_years",
        ),
        long_bear_growth_haircut=_finite_number(
            scenarios_raw["long_bear_growth_haircut"],
            "scenarios.long_bear_growth_haircut",
        ),
        long_bull_growth_uplift=_finite_number(
            scenarios_raw["long_bull_growth_uplift"],
            "scenarios.long_bull_growth_uplift",
        ),
        long_bear_multiple_compression=_finite_number(
            scenarios_raw["long_bear_multiple_compression"],
            "scenarios.long_bear_multiple_compression",
        ),
        long_bull_multiple_expansion=_finite_number(
            scenarios_raw["long_bull_multiple_expansion"],
            "scenarios.long_bull_multiple_expansion",
        ),
    )

    short_raw = _strict_mapping(mapping["short_scoring"], "short_scoring")
    _require_keys(
        short_raw,
        {
            "schema_version",
            "rsi",
            "risk_window",
            "beta_roles",
            "factor_maps",
            "risk_penalty_maps",
        },
        "short_scoring",
    )
    _literal_integer(short_raw["schema_version"], 1, "short_scoring.schema_version")
    rsi_raw = _strict_mapping(short_raw["rsi"], "short_scoring.rsi")
    _require_keys(rsi_raw, {"convention", "window_sessions"}, "short_scoring.rsi")
    _literal(rsi_raw["convention"], "cutler_sma", "short_scoring.rsi.convention")
    _literal_integer(rsi_raw["window_sessions"], 14, "short_scoring.rsi.window_sessions")
    risk_window_raw = _strict_mapping(
        short_raw["risk_window"],
        "short_scoring.risk_window",
    )
    _require_keys(
        risk_window_raw,
        {"sessions", "annualization_sessions"},
        "short_scoring.risk_window",
    )
    _literal_integer(risk_window_raw["sessions"], 252, "short_scoring.risk_window.sessions")
    _literal_integer(
        risk_window_raw["annualization_sessions"],
        252,
        "short_scoring.risk_window.annualization_sessions",
    )
    beta_raw = _strict_mapping(short_raw["beta_roles"], "short_scoring.beta_roles")
    _require_keys(
        beta_raw,
        {"factor_score", "composite_risk"},
        "short_scoring.beta_roles",
    )
    _literal(beta_raw["factor_score"], "excluded", "short_scoring.beta_roles.factor_score")
    _literal(
        beta_raw["composite_risk"],
        "absolute_exposure_penalty",
        "short_scoring.beta_roles.composite_risk",
    )
    factor_maps = _transform_mapping(
        short_raw["factor_maps"],
        "short_scoring.factor_maps",
        _V3_FACTOR_KINDS,
    )
    risk_maps = _transform_mapping(
        short_raw["risk_penalty_maps"],
        "short_scoring.risk_penalty_maps",
        _V3_RISK_KINDS,
    )
    short_scoring = ShortScoringConfig(
        schema_version=1,
        rsi=RsiPolicy(convention="cutler_sma", window_sessions=14),
        risk_window=CommonRiskPolicy(sessions=252, annualization_sessions=252),
        beta_roles=BetaRoles(
            factor_score="excluded",
            composite_risk="absolute_exposure_penalty",
        ),
        factor_maps=MappingProxyType(factor_maps),
        risk_penalty_maps=MappingProxyType(risk_maps),
    )
    macd_map = factor_maps["momentum.macd"]
    liquidity_map = factor_maps["risk.avg_volume"]
    assert (
        macd_map.low is not None
        and macd_map.high is not None
        and liquidity_map.low is not None
        and liquidity_map.high is not None
    )
    return ScoringConfig(
        version=V3_VERSION,
        analysis_mode="price_only_baseline",
        overall_horizon="short",
        supported_horizons=("short",),
        windows=(1, 5, 10, 20, 63, 126, 252, 756, 1260),
        horizon_weights=weights,
        component_factor_counts=expected_counts,
        factor_policy=FactorPolicyConfig(
            macd_indicator="macd_histogram_pct",
            macd_score_low=macd_map.low,
            macd_score_high=macd_map.high,
            abnormal_volume_indicator="abnormal_volume_strict",
            liquidity_indicator="avg_dollar_volume_20d",
            liquidity_score_low=liquidity_map.low,
            liquidity_score_high=liquidity_map.high,
            strict_finite_inputs=True,
        ),
        coverage=coverage,
        freshness=freshness,
        risk=risk,
        recommendation=recommendation,
        scenarios=scenarios,
        short_scoring=short_scoring,
        raw=raw,
    )


def _strict_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys")
    return cast(dict[str, Any], value)


def _require_keys(mapping: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{path} is missing keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown keys: {', '.join(sorted(unknown))}")


def _literal(value: Any, expected: str, path: str) -> None:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{path} must be {expected!r}")


def _literal_integer(value: Any, expected: int, path: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{path} must be integer {expected}")


def _literal_list(
    value: Any,
    expected: tuple[Any, ...],
    path: str,
    *,
    integers: bool = False,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    if integers and any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{path} must contain positive integers")
    if tuple(value) != expected:
        raise ValueError(f"{path} must be exactly {list(expected)!r}")


def _finite_number(
    value: Any,
    path: str,
    *,
    low: float | None = None,
    high: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if low is not None and result < low:
        raise ValueError(f"{path} must be at least {low}")
    if high is not None and result > high:
        raise ValueError(f"{path} must be at most {high}")
    return result


def _score_number(value: Any, leaf: str) -> float:
    return _finite_number(value, f"recommendation.{leaf}", low=0.0, high=100.0)


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _integer_mapping(value: Any, path: str, keys: tuple[str, ...]) -> dict[str, int]:
    mapping = _strict_mapping(value, path)
    _require_keys(mapping, set(keys), path)
    return {key: _positive_integer(mapping[key], f"{path}.{key}") for key in keys}


def _transform_mapping(
    value: Any,
    path: str,
    expected_kinds: Mapping[str, str],
) -> dict[str, TransformMapConfig]:
    mapping = _strict_mapping(value, path)
    _require_keys(mapping, set(expected_kinds), path)
    result: dict[str, TransformMapConfig] = {}
    for name, expected_kind in expected_kinds.items():
        item_path = f"{path}.{name}"
        item = _strict_mapping(mapping[name], item_path)
        if expected_kind in {"linear_higher", "linear_lower"}:
            _require_keys(item, {"kind", "low", "high"}, item_path)
            _literal(item["kind"], expected_kind, f"{item_path}.kind")
            low = _finite_number(item["low"], f"{item_path}.low")
            high = _finite_number(item["high"], f"{item_path}.high")
            if low >= high:
                raise ValueError(f"{item_path}.low must be less than high")
            result[name] = TransformMapConfig(kind=expected_kind, low=low, high=high)
        else:
            _require_keys(
                item,
                {"kind", "target", "slope", "nonpositive_score"},
                item_path,
            )
            _literal(item["kind"], "positive_target_penalty", f"{item_path}.kind")
            target = _finite_number(item["target"], f"{item_path}.target")
            slope = _finite_number(item["slope"], f"{item_path}.slope")
            if target <= 0 or slope <= 0:
                raise ValueError(f"{item_path} target and slope must be positive")
            result[name] = TransformMapConfig(
                kind="positive_target_penalty",
                target=target,
                slope=slope,
                nonpositive_score=_finite_number(
                    item["nonpositive_score"],
                    f"{item_path}.nonpositive_score",
                    low=0.0,
                    high=100.0,
                ),
            )
    return result


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "scoring" / "default-v1.yml"


def load_scoring_config(path: Path | None = None) -> ScoringConfig:
    config_path = path or default_config_path()
    text = config_path.read_text(encoding="utf-8")
    is_canonical_v3_path = config_path.name == "us-price-baseline-v3.yml"
    claims_v3, has_merge_key = _inspect_yaml_structure(text)
    if is_canonical_v3_path or claims_v3:
        if has_merge_key:
            raise ValueError(_V3_MERGE_ERROR)
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Scoring config must be a mapping: {config_path}")
    if is_canonical_v3_path:
        return _v3_config_from_mapping(cast(dict[str, Any], data))
    return ScoringConfig.from_mapping(cast(dict[str, Any], data))


def config_hash(config: ScoringConfig) -> str:
    payload = json.dumps(config.raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_revision() -> str:
    revision = os.getenv("STANSTOCK_CODE_REVISION")
    if revision:
        return revision[:64]
    return "working-tree"
