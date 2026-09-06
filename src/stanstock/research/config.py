from __future__ import annotations

import hashlib
import json
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
HORIZONS = ("short", "medium", "long")


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
class RecommendationConfig:
    buy_min_score: float
    buy_max_risk: float
    buy_min_confidence: float
    buy_min_avg_volume_20d: float
    buy_max_bear_downside: dict[str, float]
    avoid_max_score: float
    avoid_min_risk: float
    avoid_max_confidence: float


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
        if overall_horizon not in HORIZONS:
            raise ValueError(f"Unsupported overall_horizon {overall_horizon!r}")
        raw_supported = mapping.get("supported_horizons", list(HORIZONS))
        if not isinstance(raw_supported, list) or not raw_supported:
            raise ValueError("supported_horizons must be a non-empty list")
        supported_horizons = tuple(str(value) for value in raw_supported)
        if len(supported_horizons) != len(set(supported_horizons)):
            raise ValueError("supported_horizons contains duplicates")
        if any(horizon not in HORIZONS for horizon in supported_horizons):
            raise ValueError(f"supported_horizons must contain only {HORIZONS!r}")
        if overall_horizon not in supported_horizons:
            raise ValueError("overall_horizon must be included in supported_horizons")
        weights = cast(dict[str, dict[str, float]], mapping["horizon_weights"])
        for horizon in HORIZONS:
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
                for horizon in HORIZONS
            },
            component_factor_counts=component_factor_counts,
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
                buy_min_avg_volume_20d=float(recommendation["buy_min_avg_volume_20d"]),
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
