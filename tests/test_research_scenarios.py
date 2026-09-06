from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from stanstock.research.config import load_scoring_config
from stanstock.research.scenarios import build_scenarios
from stanstock.research.types import (
    AggregateScore,
    ComponentScores,
    IndicatorResult,
    ResearchValues,
)


def _aggregate() -> AggregateScore:
    return AggregateScore(
        overall=70,
        horizon_scores={"short": 68, "medium": 70, "long": 72},
        confidence=60,
        confidence_status="heuristic",
        component_scores=ComponentScores(
            components={"quality": 70, "valuation": 65, "growth": 75},
            factor_scores={},
            missing={},
            coverage=0.8,
        ),
        missingness_penalty=1.0,
        freshness_penalty=1.0,
    )


def _frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2024, 1, 1) + timedelta(days=index) for index in range(rows)],
            "close": [100 + index * 0.15 + (index % 7) * 0.03 for index in range(rows)],
        }
    )


def test_scenarios_are_null_with_explicit_reason_when_inputs_are_insufficient() -> None:
    config = load_scoring_config()
    scenarios = build_scenarios(
        _frame(20),
        IndicatorResult(values={}),
        ResearchValues(values={}),
        _aggregate(),
        config,
    )

    for horizon, scenario in scenarios.items():
        assert scenario.bear is None
        assert scenario.base is None
        assert scenario.bull is None
        assert scenario.probability_positive is None
        if horizon == "long":
            assert "Insufficient long-horizon scenario inputs" in scenario.insufficiency_reason
        else:
            assert (
                f"Insufficient {horizon}-horizon scenario inputs" in scenario.insufficiency_reason
            )
        assert scenario.confidence_status == "heuristic"


def test_rule_scenarios_require_real_inputs_and_then_produce_ordered_ranges() -> None:
    config = load_scoring_config()
    scenarios = build_scenarios(
        _frame(90),
        IndicatorResult(
            values={
                "momentum_20d": 0.05,
                "momentum_63d": 0.08,
                "annualized_volatility": 0.25,
            }
        ),
        ResearchValues(values={"revenue_growth": 0.08, "free_cash_flow_yield": 0.04}),
        _aggregate(),
        config,
    )

    assert scenarios["short"].bear is not None
    assert scenarios["medium"].bear is not None
    assert scenarios["long"].bear is not None
    for scenario in scenarios.values():
        assert scenario.bear is not None
        assert scenario.base is not None
        assert scenario.bull is not None
        assert scenario.bear <= scenario.base <= scenario.bull
        assert scenario.probability_positive is None


def test_price_only_baseline_withholds_unsupported_horizons() -> None:
    config = load_scoring_config(
        Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v1.yml"
    )

    scenarios = build_scenarios(
        _frame(90),
        IndicatorResult(
            values={
                "momentum_20d": 0.05,
                "momentum_63d": 0.08,
                "annualized_volatility": 0.25,
            }
        ),
        ResearchValues(values={"revenue_growth": 0.08, "free_cash_flow_yield": 0.04}),
        _aggregate(),
        config,
    )

    assert scenarios["short"].bear is not None
    for horizon in ("medium", "long"):
        assert scenarios[horizon].bear is None
        assert scenarios[horizon].base is None
        assert scenarios[horizon].bull is None
        assert scenarios[horizon].method == "unsupported_by_model"
        assert "not supported by 'price_only_baseline'" in scenarios[horizon].insufficiency_reason


def test_scenario_total_returns_are_clamped_at_negative_one() -> None:
    config = load_scoring_config()
    scenarios = build_scenarios(
        _frame(90),
        IndicatorResult(
            values={
                "momentum_20d": -10.0,
                "momentum_63d": -10.0,
                "annualized_volatility": 5.0,
            }
        ),
        ResearchValues(
            values={
                "revenue_growth": -5.0,
                "net_income_growth": -5.0,
                "free_cash_flow_yield": -2.0,
            }
        ),
        _aggregate(),
        config,
    )

    for scenario in scenarios.values():
        assert scenario.bear is not None
        assert scenario.base is not None
        assert scenario.bull is not None
        assert scenario.bear >= -1.0
        assert scenario.base >= -1.0
        assert scenario.bull >= -1.0
        assert scenario.bear <= scenario.base <= scenario.bull


def test_probability_appears_only_when_configured_sample_support_is_sufficient() -> None:
    config = load_scoring_config()
    scenarios = build_scenarios(
        _frame(400),
        IndicatorResult(values={"momentum_20d": 0.04, "annualized_volatility": 0.20}),
        ResearchValues(values={"revenue_growth": 0.06, "free_cash_flow_yield": 0.03}),
        _aggregate(),
        config,
        sample_support={"short": 80, "medium": 80, "long": 40},
    )

    assert scenarios["short"].probability_positive is not None
    assert scenarios["medium"].probability_positive is not None
    assert scenarios["long"].probability_positive is None
    assert "calibrated comparable outcomes" in scenarios["long"].insufficiency_reason
    assert scenarios["short"].insufficiency_reason == ""
