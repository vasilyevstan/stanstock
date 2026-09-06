from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date

from stanstock.research.config import COMPONENTS, HORIZONS, ScoringConfig
from stanstock.research.indicators import last_observation_date
from stanstock.research.models import Recommendation, RiskClass
from stanstock.research.types import (
    AggregateScore,
    ComponentScores,
    IndicatorResult,
    RecommendationDecision,
    ResearchValues,
    RiskAssessment,
    Scenario,
)


def score_components(
    indicators: IndicatorResult,
    fundamentals: ResearchValues,
    config: ScoringConfig,
) -> ComponentScores:
    factors: dict[str, float] = {}
    missing: dict[str, str] = {}

    _add(
        factors, missing, "quality.net_margin", fundamentals, "net_margin", _score_higher(0.0, 0.25)
    )
    _add(
        factors,
        missing,
        "quality.operating_margin",
        fundamentals,
        "operating_margin",
        _score_higher(0.0, 0.30),
    )
    _add(
        factors,
        missing,
        "quality.fcf_margin",
        fundamentals,
        "free_cash_flow_margin",
        _score_higher(0.0, 0.20),
    )
    _add(
        factors,
        missing,
        "quality.cash_to_debt",
        fundamentals,
        "cash_to_debt",
        _score_higher(0.0, 2.0),
    )
    _add(
        factors,
        missing,
        "quality.debt_to_equity",
        fundamentals,
        "debt_to_equity",
        _score_lower(0.0, 2.0),
    )
    _add(
        factors,
        missing,
        "quality.interest_coverage",
        fundamentals,
        "interest_coverage",
        _score_higher(1.0, 8.0),
    )
    _add(
        factors,
        missing,
        "quality.fcf_consistency",
        fundamentals,
        "free_cash_flow_consistency",
        _score_higher(0.0, 1.0),
    )
    _add(factors, missing, "quality.roe", fundamentals, "roe", _score_higher(0.0, 0.25))
    _add(factors, missing, "quality.roic", fundamentals, "roic", _score_higher(0.0, 0.20))
    _add(
        factors,
        missing,
        "quality.current_ratio",
        fundamentals,
        "current_ratio",
        _score_higher(0.8, 2.5),
    )

    _add(
        factors, missing, "growth.revenue", fundamentals, "revenue_growth", _score_centered_growth()
    )
    _add(
        factors,
        missing,
        "growth.net_income",
        fundamentals,
        "net_income_growth",
        _score_centered_growth(),
    )
    _add(
        factors,
        missing,
        "growth.fcf",
        fundamentals,
        "free_cash_flow_growth",
        _score_centered_growth(),
    )

    _add(factors, missing, "valuation.pe", fundamentals, "pe_ratio", _score_lower(8.0, 35.0))
    _add(factors, missing, "valuation.ps", fundamentals, "ps_ratio", _score_lower(1.0, 12.0))
    _add(factors, missing, "valuation.pb", fundamentals, "pb_ratio", _score_lower(0.8, 8.0))
    _add(
        factors, missing, "valuation.ev_sales", fundamentals, "ev_to_sales", _score_lower(1.0, 10.0)
    )
    _add(
        factors,
        missing,
        "valuation.ev_ebitda",
        fundamentals,
        "ev_to_ebitda",
        _score_lower(5.0, 25.0),
    )
    _add(factors, missing, "valuation.ev_ebit", fundamentals, "ev_to_ebit", _score_lower(6.0, 30.0))
    _add(
        factors,
        missing,
        "valuation.price_fcf",
        fundamentals,
        "price_to_fcf",
        _score_lower(8.0, 35.0),
    )
    _add(
        factors,
        missing,
        "valuation.fcf_yield",
        fundamentals,
        "free_cash_flow_yield",
        _score_higher(0.0, 0.10),
    )

    _add(
        factors,
        missing,
        "momentum.return_20d",
        indicators,
        "return_20d",
        _score_higher(-0.10, 0.15),
    )
    _add(
        factors,
        missing,
        "momentum.return_63d",
        indicators,
        "return_63d",
        _score_higher(-0.20, 0.30),
    )
    _add(
        factors,
        missing,
        "momentum.return_126d",
        indicators,
        "return_126d",
        _score_higher(-0.30, 0.45),
    )
    _add(
        factors,
        missing,
        "momentum.sma_50",
        indicators,
        "close_vs_sma_50",
        _score_higher(-0.10, 0.10),
    )
    _add(
        factors,
        missing,
        "momentum.sma_200",
        indicators,
        "close_vs_sma_200",
        _score_higher(-0.15, 0.20),
    )
    _add(factors, missing, "momentum.rsi", indicators, "rsi_14", _score_rsi)
    _add(
        factors,
        missing,
        "momentum.macd",
        indicators,
        config.factor_policy.macd_indicator,
        _score_higher(
            config.factor_policy.macd_score_low,
            config.factor_policy.macd_score_high,
        ),
        require_finite=config.factor_policy.strict_finite_inputs,
    )
    _add(factors, missing, "momentum.52w", indicators, "52w_position", _score_higher(0.15, 0.95))

    _add(
        factors,
        missing,
        "risk.annualized_volatility",
        indicators,
        "annualized_volatility",
        _score_lower(0.12, 0.65),
    )
    _add(
        factors,
        missing,
        "risk.downside_volatility",
        indicators,
        "downside_volatility",
        _score_lower(0.08, 0.50),
    )
    _add(
        factors,
        missing,
        "risk.max_drawdown",
        indicators,
        "max_drawdown",
        lambda value: _score_higher(-0.60, -0.05)(value),
    )
    _add(factors, missing, "risk.beta", indicators, "beta", _score_beta)
    _add(
        factors,
        missing,
        "risk.abnormal_volume",
        indicators,
        config.factor_policy.abnormal_volume_indicator,
        _score_volume_abnormality,
        require_finite=config.factor_policy.strict_finite_inputs,
    )
    _add(
        factors,
        missing,
        "risk.avg_volume",
        indicators,
        config.factor_policy.liquidity_indicator,
        _score_higher(
            config.factor_policy.liquidity_score_low,
            config.factor_policy.liquidity_score_high,
        ),
        require_finite=config.factor_policy.strict_finite_inputs,
    )

    _add(
        factors,
        missing,
        "market.relative_20d",
        indicators,
        "relative_return_20d",
        _score_higher(-0.08, 0.08),
    )
    _add(
        factors,
        missing,
        "market.relative_63d",
        indicators,
        "relative_return_63d",
        _score_higher(-0.15, 0.15),
    )
    _add(
        factors,
        missing,
        "market.relative_252d",
        indicators,
        "relative_return_252d",
        _score_higher(-0.25, 0.25),
    )

    components = _component_averages(factors)
    expected = sum(config.component_factor_counts.values())
    coverage = min(1.0, len(factors) / expected) if expected > 0 else 0.0
    return ComponentScores(
        components=components,
        factor_scores=factors,
        missing=missing,
        coverage=coverage,
    )


def aggregate_score(
    component_scores: ComponentScores,
    config: ScoringConfig,
    *,
    decision_date: date | None = None,
    indicators: IndicatorResult | None = None,
) -> AggregateScore:
    horizon_scores = {
        horizon: _weighted_score(component_scores.components, config.horizon_weights[horizon])
        for horizon in HORIZONS
    }
    raw = horizon_scores[config.overall_horizon]
    missingness_penalty = max(
        config.coverage.score_penalty_floor,
        1.0 - config.coverage.score_penalty_rate * (1.0 - component_scores.coverage),
    )
    freshness_penalty = _freshness_penalty(config, decision_date, indicators)
    overall = _clamp(raw * missingness_penalty * freshness_penalty)
    confidence = _confidence(component_scores.coverage, config, freshness_penalty)
    return AggregateScore(
        overall=overall,
        horizon_scores={key: _clamp(value) for key, value in horizon_scores.items()},
        confidence=confidence,
        confidence_status="heuristic",
        component_scores=component_scores,
        missingness_penalty=missingness_penalty,
        freshness_penalty=freshness_penalty,
    )


def assess_risk(
    indicators: IndicatorResult, fundamentals: ResearchValues, config: ScoringConfig
) -> RiskAssessment:
    risk_inputs: list[float] = []
    volatility = indicators.get("annualized_volatility")
    if volatility is not None:
        risk_inputs.append(_score_higher(0.10, 0.75)(volatility))
    downside = indicators.get("downside_volatility")
    if downside is not None:
        risk_inputs.append(_score_higher(0.05, 0.55)(downside))
    drawdown = indicators.get("max_drawdown")
    if drawdown is not None:
        risk_inputs.append(_score_lower(-0.60, -0.05)(drawdown))
    beta = indicators.get("beta")
    if beta is not None:
        risk_inputs.append(_clamp(abs(beta - 1.0) * 50.0))
    leverage = fundamentals.get("debt_to_equity")
    if leverage is not None:
        risk_inputs.append(_score_higher(0.0, 3.0)(leverage))
    fcf_consistency = fundamentals.get("free_cash_flow_consistency")
    if fcf_consistency is not None:
        risk_inputs.append(100.0 - _score_higher(0.0, 1.0)(fcf_consistency))
    if not risk_inputs:
        return RiskAssessment(
            score=None,
            risk_class=RiskClass.INSUFFICIENT.value,
            insufficiency_reason="No supported risk inputs are available",
        )
    score = _clamp(sum(risk_inputs) / len(risk_inputs))
    if score <= config.risk.low_max:
        risk_class = RiskClass.LOW.value
    elif score <= config.risk.medium_max:
        risk_class = RiskClass.MEDIUM.value
    elif score <= config.risk.high_max:
        risk_class = RiskClass.HIGH.value
    else:
        risk_class = RiskClass.VERY_HIGH.value
    return RiskAssessment(score=score, risk_class=risk_class)


def decide_recommendation(
    score: float,
    risk: RiskAssessment,
    confidence: float,
    config: ScoringConfig,
    *,
    scenarios: dict[str, Scenario],
    indicators: IndicatorResult,
) -> RecommendationDecision:
    """Apply deterministic recommendation gates.

    BUY requires affirmative evidence across score, risk, confidence,
    scenario downside, and liquidity. Missing evidence blocks BUY. AVOID
    remains conservative: any configured danger gate is sufficient; otherwise
    the result is HOLD.
    """

    buy_gates = {
        "score": score >= config.recommendation.buy_min_score,
        "risk_present": risk.score is not None,
        "risk": risk.score is not None and risk.score <= config.recommendation.buy_max_risk,
        "confidence": confidence >= config.recommendation.buy_min_confidence,
        "liquidity_present": (indicators.get(config.factor_policy.liquidity_indicator) is not None),
        "liquidity": _passes_liquidity(indicators, config),
    }
    for horizon, max_bear_downside in config.recommendation.buy_max_bear_downside.items():
        scenario = scenarios.get(horizon)
        present = _scenario_has_range(scenario)
        buy_gates[f"{horizon}_scenario_present"] = present
        buy_gates[f"{horizon}_bear_downside"] = (
            scenario is not None
            and scenario.bear is not None
            and scenario.bear >= max_bear_downside
        )
    avoid_gates = {
        "score": score <= config.recommendation.avoid_max_score,
        "risk": risk.score is not None and risk.score >= config.recommendation.avoid_min_risk,
        "confidence": confidence <= config.recommendation.avoid_max_confidence,
    }
    if all(buy_gates.values()):
        recommendation = Recommendation.BUY.value
    elif any(avoid_gates.values()):
        recommendation = Recommendation.AVOID.value
    else:
        recommendation = Recommendation.HOLD.value
    return RecommendationDecision(
        recommendation=recommendation,
        gates={
            **{f"buy_{key}": value for key, value in buy_gates.items()},
            **{f"avoid_{key}": value for key, value in avoid_gates.items()},
        },
    )


def _passes_liquidity(indicators: IndicatorResult, config: ScoringConfig) -> bool:
    average_volume = indicators.get(config.factor_policy.liquidity_indicator)
    return (
        average_volume is not None and average_volume >= config.recommendation.buy_min_liquidity_20d
    )


def _scenario_has_range(scenario: Scenario | None) -> bool:
    return (
        scenario is not None
        and scenario.bear is not None
        and scenario.base is not None
        and scenario.bull is not None
    )


def _component_averages(factors: dict[str, float]) -> dict[str, float]:
    components: dict[str, list[float]] = {component: [] for component in COMPONENTS}
    for name, value in factors.items():
        components[_component_name(name)].append(value)
    return {
        component: _clamp(sum(values) / len(values))
        for component, values in components.items()
        if values
    }


def _component_name(factor_name: str) -> str:
    prefix = factor_name.split(".", 1)[0]
    return {
        "momentum": "momentum_technical",
        "risk": "risk_liquidity",
        "market": "market_sector",
    }.get(prefix, prefix)


def _weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    present_weight = sum(weight for component, weight in weights.items() if component in components)
    if present_weight <= 0:
        return 0.0
    return (
        sum(
            components[component] * weight
            for component, weight in weights.items()
            if component in components
        )
        / present_weight
    )


def _add(
    factors: dict[str, float],
    missing: dict[str, str],
    score_name: str,
    source: ResearchValues,
    value_name: str,
    scorer: Callable[[float], float],
    *,
    require_finite: bool = False,
) -> None:
    value = source.get(value_name)
    if value is None:
        missing[score_name] = source.missing.get(value_name, "Missing input")
        return
    if require_finite and not math.isfinite(value):
        missing[score_name] = "Input must be finite"
        return
    scored = float(scorer(value))
    if require_finite and not math.isfinite(scored):
        missing[score_name] = "Score must be finite"
        return
    factors[score_name] = _clamp(scored)


def _score_higher(low: float, high: float) -> Callable[[float], float]:
    def score(value: float) -> float:
        if high == low:
            return 50.0
        return (value - low) / (high - low) * 100.0

    return score


def _score_lower(low: float, high: float) -> Callable[[float], float]:
    def score(value: float) -> float:
        if high == low:
            return 50.0
        return (high - value) / (high - low) * 100.0

    return score


def _score_centered_growth() -> Callable[[float], float]:
    def score(value: float) -> float:
        if value < 0:
            return 50.0 + value / 0.30 * 50.0
        return 50.0 + min(value, 0.30) / 0.30 * 50.0

    return score


def _score_rsi(value: float) -> float:
    if value < 30.0:
        return 45.0 + (30.0 - value) * 0.5
    if value > 75.0:
        return max(0.0, 100.0 - (value - 75.0) * 4.0)
    return 100.0 - abs(value - 55.0) * 1.2


def _score_beta(value: float) -> float:
    return 100.0 - min(100.0, abs(value - 1.0) * 65.0)


def _score_volume_abnormality(value: float) -> float:
    if value <= 0:
        return 0.0
    if value <= 2.0:
        return 100.0 - abs(value - 1.0) * 20.0
    return max(0.0, 80.0 - (value - 2.0) * 20.0)


def _freshness_penalty(
    config: ScoringConfig,
    decision_date: date | None,
    indicators: IndicatorResult | None,
) -> float:
    if decision_date is None or indicators is None:
        return 1.0
    last_date = last_observation_date(indicators)
    if last_date is None:
        return 1.0 - config.freshness.stale_score_penalty
    age = (decision_date - last_date).days
    if age <= config.freshness.fresh_days:
        return 1.0
    if age >= config.freshness.stale_days:
        return 1.0 - config.freshness.stale_score_penalty
    scale = (age - config.freshness.fresh_days) / (
        config.freshness.stale_days - config.freshness.fresh_days
    )
    return 1.0 - config.freshness.stale_score_penalty * scale


def _confidence(coverage: float, config: ScoringConfig, freshness_penalty: float) -> float:
    raw = coverage * 100.0
    cap = config.coverage.confidence_cap
    if coverage < config.coverage.minimum_component_coverage:
        cap = min(cap, 35.0)
    if freshness_penalty < 1.0:
        cap = min(cap, config.freshness.stale_confidence_cap + freshness_penalty * 10.0)
    return _clamp(max(config.coverage.confidence_floor, min(raw, cap)))


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
