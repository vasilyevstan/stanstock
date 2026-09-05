from __future__ import annotations

from datetime import date, timedelta

from stanstock.research.config import HORIZONS, load_scoring_config
from stanstock.research.scoring import (
    aggregate_score,
    assess_risk,
    decide_recommendation,
    score_components,
)
from stanstock.research.types import IndicatorResult, ResearchValues, RiskAssessment, Scenario


def _rich_indicators() -> IndicatorResult:
    values = {
        "return_20d": 0.08,
        "return_63d": 0.18,
        "return_126d": 0.24,
        "close_vs_sma_50": 0.06,
        "close_vs_sma_200": 0.12,
        "rsi_14": 58.0,
        "macd_histogram": 1.0,
        "52w_position": 0.85,
        "annualized_volatility": 0.22,
        "downside_volatility": 0.16,
        "max_drawdown": -0.12,
        "beta": 1.05,
        "abnormal_volume": 1.1,
        "avg_volume_20d": 1_500_000,
        "relative_return_20d": 0.02,
        "relative_return_63d": 0.07,
        "relative_return_252d": 0.11,
    }
    return IndicatorResult(values=values, observation_count=260, last_date=date(2026, 9, 4))


def _rich_fundamentals() -> ResearchValues:
    return ResearchValues(
        values={
            "net_margin": 0.18,
            "operating_margin": 0.22,
            "free_cash_flow_margin": 0.16,
            "cash_to_debt": 1.8,
            "debt_to_equity": 0.35,
            "interest_coverage": 9.0,
            "free_cash_flow_consistency": 1.0,
            "revenue_growth": 0.18,
            "net_income_growth": 0.22,
            "free_cash_flow_growth": 0.16,
            "pe_ratio": 14.0,
            "ps_ratio": 2.0,
            "pb_ratio": 2.0,
            "ev_to_sales": 2.2,
            "ev_to_ebitda": 8.0,
            "free_cash_flow_yield": 0.08,
        }
    )


def _evidenced_scenarios() -> dict[str, Scenario]:
    return {
        "short": Scenario(
            bear=-0.02,
            base=0.03,
            bull=0.08,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
        "medium": Scenario(
            bear=-0.12,
            base=0.10,
            bull=0.25,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
        "long": Scenario(
            bear=-0.20,
            base=0.35,
            bull=0.80,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
    }


def test_config_horizon_weights_are_versioned_and_sum_to_one() -> None:
    config = load_scoring_config()

    assert config.version == "default-v1"
    assert config.horizon_weights == {
        "short": {
            "quality": 0.05,
            "growth": 0.05,
            "valuation": 0.05,
            "momentum_technical": 0.40,
            "risk_liquidity": 0.25,
            "market_sector": 0.20,
        },
        "medium": {
            "quality": 0.20,
            "growth": 0.20,
            "valuation": 0.20,
            "momentum_technical": 0.15,
            "risk_liquidity": 0.15,
            "market_sector": 0.10,
        },
        "long": {
            "quality": 0.30,
            "growth": 0.25,
            "valuation": 0.20,
            "momentum_technical": 0.05,
            "risk_liquidity": 0.15,
            "market_sector": 0.05,
        },
    }
    assert config.recommendation.buy_min_avg_volume_20d == 100_000
    assert config.recommendation.buy_max_bear_downside == {
        "short": -0.08,
        "medium": -0.30,
        "long": -0.50,
    }
    for horizon in HORIZONS:
        assert sum(config.horizon_weights[horizon].values()) == 1.0
    assert (
        config.horizon_weights["short"]["momentum_technical"]
        > config.horizon_weights["long"]["momentum_technical"]
    )
    assert config.horizon_weights["long"]["quality"] > config.horizon_weights["short"]["quality"]


def test_score_bounds_confidence_cap_and_freshness_penalty() -> None:
    config = load_scoring_config()
    components = score_components(_rich_indicators(), _rich_fundamentals(), config)
    fresh = aggregate_score(
        components,
        config,
        decision_date=date(2026, 9, 5),
        indicators=_rich_indicators(),
    )
    stale_indicators = IndicatorResult(
        values=_rich_indicators().values,
        observation_count=260,
        last_date=date(2026, 7, 1),
    )
    stale = aggregate_score(
        components,
        config,
        decision_date=date(2026, 9, 5),
        indicators=stale_indicators,
    )

    assert all(0 <= value <= 100 for value in components.components.values())
    assert all(0 <= value <= 100 for value in fresh.horizon_scores.values())
    assert 0 <= fresh.overall <= 100
    assert fresh.confidence <= config.coverage.confidence_cap
    assert stale.overall < fresh.overall
    assert stale.confidence <= config.freshness.stale_confidence_cap + 10


def test_risk_classes_and_recommendation_gates_are_deterministic() -> None:
    config = load_scoring_config()
    indicators = _rich_indicators()
    low_risk = assess_risk(_rich_indicators(), _rich_fundamentals(), config)
    buy = decide_recommendation(
        80, low_risk, 70, config, scenarios=_evidenced_scenarios(), indicators=indicators
    )
    avoid = decide_recommendation(
        30, low_risk, 70, config, scenarios=_evidenced_scenarios(), indicators=indicators
    )
    insufficient_risk = assess_risk(
        IndicatorResult(values={}),
        ResearchValues(values={}),
        config,
    )
    insufficient_decision = decide_recommendation(
        90,
        insufficient_risk,
        70,
        config,
        scenarios=_evidenced_scenarios(),
        indicators=indicators,
    )

    assert low_risk.risk_class in {"low", "medium"}
    assert buy.recommendation == "buy"
    assert all(value for key, value in buy.gates.items() if key.startswith("buy_"))
    assert avoid.recommendation == "avoid"
    assert insufficient_risk.score is None
    assert insufficient_risk.risk_class == "insufficient"
    assert insufficient_risk.insufficiency_reason
    assert insufficient_decision.recommendation == "hold"
    assert insufficient_decision.gates["buy_risk_present"] is False


def test_buy_requires_scenario_evidence_and_liquidity() -> None:
    config = load_scoring_config()
    low_risk = RiskAssessment(score=20, risk_class="low")
    missing_short = {
        **_evidenced_scenarios(),
        "short": Scenario(
            bear=None,
            base=None,
            bull=None,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="missing",
            method="test",
        ),
    }
    no_scenario = decide_recommendation(
        90, low_risk, 80, config, scenarios=missing_short, indicators=_rich_indicators()
    )
    no_liquidity = decide_recommendation(
        90,
        low_risk,
        80,
        config,
        scenarios=_evidenced_scenarios(),
        indicators=IndicatorResult(values={}),
    )
    thin_liquidity = decide_recommendation(
        90,
        low_risk,
        80,
        config,
        scenarios=_evidenced_scenarios(),
        indicators=IndicatorResult(values={"avg_volume_20d": 10}),
    )

    assert no_scenario.recommendation == "hold"
    assert no_scenario.gates["buy_short_scenario_present"] is False
    assert no_liquidity.recommendation == "hold"
    assert no_liquidity.gates["buy_liquidity_present"] is False
    assert no_liquidity.gates["buy_liquidity"] is False
    assert thin_liquidity.recommendation == "hold"
    assert thin_liquidity.gates["buy_liquidity_present"] is True
    assert thin_liquidity.gates["buy_liquidity"] is False


def test_buy_requires_configured_bear_downside_by_horizon() -> None:
    config = load_scoring_config()
    adverse = {
        **_evidenced_scenarios(),
        "medium": Scenario(
            bear=-0.31,
            base=0.10,
            bull=0.20,
            probability_positive=None,
            confidence=60,
            confidence_status="heuristic",
            insufficiency_reason="",
            method="test",
        ),
    }

    decision = decide_recommendation(
        90,
        RiskAssessment(score=20, risk_class="low"),
        80,
        config,
        scenarios=adverse,
        indicators=_rich_indicators(),
    )

    assert decision.recommendation == "hold"
    assert decision.gates["buy_medium_scenario_present"] is True
    assert decision.gates["buy_medium_bear_downside"] is False


def test_missingness_penalty_lowers_sparse_scores_without_zero_filling() -> None:
    config = load_scoring_config()
    sparse = score_components(
        IndicatorResult(values={"return_20d": 0.20}, last_date=date.today() - timedelta(days=1)),
        ResearchValues(values={}),
        config,
    )
    aggregate = aggregate_score(sparse, config)

    assert sparse.coverage < config.coverage.minimum_component_coverage
    assert aggregate.missingness_penalty < 1
    assert aggregate.confidence <= 35
