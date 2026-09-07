from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import yaml

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
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> Self:
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
            raw=mapping,
        )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "scoring" / "default-v1.yml"


def load_scoring_config(path: Path | None = None) -> ScoringConfig:
    config_path = path or default_config_path()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Scoring config must be a mapping: {config_path}")
    return ScoringConfig.from_mapping(cast(dict[str, Any], data))


def config_hash(config: ScoringConfig) -> str:
    payload = json.dumps(config.raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_revision() -> str:
    revision = os.getenv("STANSTOCK_CODE_REVISION")
    if revision:
        return revision[:64]
    return "working-tree"
