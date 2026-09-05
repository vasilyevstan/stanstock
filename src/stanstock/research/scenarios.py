from __future__ import annotations

import numpy as np
import polars as pl

from stanstock.research.config import ScoringConfig
from stanstock.research.types import AggregateScore, IndicatorResult, ResearchValues, Scenario


def build_scenarios(
    price_frame: pl.DataFrame,
    indicators: IndicatorResult,
    fundamentals: ResearchValues,
    score: AggregateScore,
    config: ScoringConfig,
    *,
    sample_support: dict[str, int] | None = None,
) -> dict[str, Scenario]:
    support = sample_support or {}
    return {
        "short": _short_scenario(price_frame, indicators, score, config, support.get("short", 0)),
        "medium": _medium_scenario(
            price_frame, indicators, fundamentals, score, config, support.get("medium", 0)
        ),
        "long": _long_scenario(fundamentals, score, config, support.get("long", 0)),
    }


def _short_scenario(
    frame: pl.DataFrame,
    indicators: IndicatorResult,
    score: AggregateScore,
    config: ScoringConfig,
    support: int,
) -> Scenario:
    horizon = config.scenarios.short_horizon_days
    returns = _rolling_returns(frame, horizon)
    if len(returns) >= config.scenarios.empirical_min_observations["short"]:
        bear, base, bull = _quantiles(returns)
        method = "empirical_rolling_10d"
    else:
        momentum = indicators.get("momentum_20d")
        volatility = indicators.get("annualized_volatility")
        missing = _missing_names(
            {
                "momentum_20d": momentum,
                "annualized_volatility": volatility,
            }
        )
        if missing:
            return _missing_scenario(
                score.confidence,
                "rule_volatility_momentum_10d",
                f"Insufficient short-horizon scenario inputs: missing {', '.join(missing)}",
            )
        assert momentum is not None
        assert volatility is not None
        base = 0.25 * momentum
        spread = max(0.02, volatility / np.sqrt(252.0) * np.sqrt(float(horizon)) * 1.3)
        bear, bull = base - spread, base + spread
        method = "rule_volatility_momentum_10d"
    probability, reason = _probability(
        returns, config.scenarios.probability_min_samples["short"], support
    )
    return _scenario(bear, base, bull, probability, score.confidence, reason, method)


def _medium_scenario(
    frame: pl.DataFrame,
    indicators: IndicatorResult,
    fundamentals: ResearchValues,
    score: AggregateScore,
    config: ScoringConfig,
    support: int,
) -> Scenario:
    horizon = config.scenarios.medium_horizon_days
    returns = _rolling_returns(frame, horizon)
    if len(returns) >= config.scenarios.empirical_min_observations["medium"]:
        bear, base, bull = _quantiles(returns)
        method = "empirical_rolling_6m"
    else:
        momentum = indicators.get("momentum_126d") or indicators.get("momentum_63d")
        growth = _average_present(
            fundamentals,
            ("revenue_growth", "net_income_growth", "free_cash_flow_growth"),
        )
        valuation_component = score.component_scores.components.get("valuation")
        volatility = indicators.get("annualized_volatility")
        missing = _missing_names(
            {
                "momentum_126d_or_63d": momentum,
                "fundamental_growth": growth,
                "valuation_component": valuation_component,
                "annualized_volatility": volatility,
            }
        )
        if missing:
            return _missing_scenario(
                score.confidence,
                "rule_growth_valuation_momentum_6m",
                f"Insufficient medium-horizon scenario inputs: missing {', '.join(missing)}",
            )
        assert momentum is not None
        assert growth is not None
        assert valuation_component is not None
        assert volatility is not None
        valuation = valuation_component / 100.0 - 0.5
        base = 0.35 * momentum + 0.45 * growth + 0.12 * valuation
        spread = max(0.10, volatility * np.sqrt(float(horizon) / 252.0) * 0.9)
        bear, bull = base - spread, base + spread
        method = "rule_growth_valuation_momentum_6m"
    probability, reason = _probability(
        returns, config.scenarios.probability_min_samples["medium"], support
    )
    return _scenario(bear, base, bull, probability, score.confidence, reason, method)


def _long_scenario(
    fundamentals: ResearchValues,
    score: AggregateScore,
    config: ScoringConfig,
    support: int,
) -> Scenario:
    years = config.scenarios.long_term_years
    growth = _average_present(
        fundamentals,
        ("revenue_growth", "net_income_growth", "free_cash_flow_growth"),
    )
    fcf_yield = fundamentals.get("free_cash_flow_yield")
    quality_component = score.component_scores.components.get("quality")
    missing = _missing_names(
        {
            "fundamental_growth": growth,
            "free_cash_flow_yield": fcf_yield,
            "quality_component": quality_component,
        }
    )
    if missing:
        reason = (
            f"Insufficient long-horizon scenario inputs: missing {', '.join(missing)}; "
            "long-term probability also requires calibrated comparable outcomes"
        )
        return _missing_scenario(score.confidence, "explicit_fundamental_3y", reason)
    assert growth is not None
    assert fcf_yield is not None
    assert quality_component is not None
    quality = quality_component / 100.0 - 0.5
    base_annual = growth * 0.65 + fcf_yield * 0.50 + quality * 0.04
    bear_annual = (
        base_annual * config.scenarios.long_bear_growth_haircut
        + config.scenarios.long_bear_multiple_compression / years
    )
    bull_annual = (
        base_annual * config.scenarios.long_bull_growth_uplift
        + config.scenarios.long_bull_multiple_expansion / years
    )
    bear = (1.0 + bear_annual) ** years - 1.0
    base = (1.0 + base_annual) ** years - 1.0
    bull = (1.0 + bull_annual) ** years - 1.0
    reason = (
        "Long-term probability requires calibrated comparable outcomes; "
        f"rules-only v1 has support count {support} but no calibrated outcome sample"
    )
    return _scenario(bear, base, bull, None, score.confidence, reason, "explicit_fundamental_3y")


def _rolling_returns(
    frame: pl.DataFrame, horizon: int
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    if "close" not in frame.columns or frame.height <= horizon:
        return np.asarray([], dtype=np.float64)
    clean = frame.select(pl.col("close").cast(pl.Float64, strict=False)).filter(
        pl.col("close").is_not_null() & (pl.col("close") > 0)
    )
    closes = np.asarray(clean["close"].to_numpy(), dtype=np.float64)
    if len(closes) <= horizon:
        return np.asarray([], dtype=np.float64)
    return closes[horizon:] / closes[:-horizon] - 1.0


def _quantiles(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> tuple[float, float, float]:
    quantiles = np.quantile(values, [0.2, 0.5, 0.8])
    return float(quantiles[0]), float(quantiles[1]), float(quantiles[2])


def _probability(
    returns: np.ndarray[tuple[int], np.dtype[np.float64]],
    minimum: int,
    support: int,
) -> tuple[float | None, str]:
    usable = min(len(returns), support)
    if usable < minimum:
        return None, _insufficient_reason(minimum, usable)
    recent = returns[-usable:]
    return float(np.mean(recent > 0.0)), ""


def _insufficient_reason(minimum: int, actual: int) -> str:
    if actual >= minimum:
        return ""
    return f"Insufficient comparable observations for probability: {actual}/{minimum}"


def _scenario(
    bear: float,
    base: float,
    bull: float,
    probability: float | None,
    confidence: float,
    reason: str,
    method: str,
) -> Scenario:
    ordered = sorted([max(-1.0, float(value)) for value in (bear, base, bull)])
    return Scenario(
        bear=ordered[0],
        base=ordered[1],
        bull=ordered[2],
        probability_positive=probability,
        confidence=confidence,
        confidence_status="heuristic",
        insufficiency_reason=reason,
        method=method,
    )


def _missing_scenario(confidence: float, method: str, reason: str) -> Scenario:
    return Scenario(
        bear=None,
        base=None,
        bull=None,
        probability_positive=None,
        confidence=confidence,
        confidence_status="heuristic",
        insufficiency_reason=reason,
        method=method,
    )


def _missing_names(values: dict[str, float | None]) -> list[str]:
    return [key for key, value in values.items() if value is None]


def _average_present(values: ResearchValues, keys: tuple[str, ...]) -> float | None:
    present = [value for key in keys if (value := values.get(key)) is not None]
    if not present:
        return None
    return sum(present) / len(present)
