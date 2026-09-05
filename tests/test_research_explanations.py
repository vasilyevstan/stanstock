from __future__ import annotations

from datetime import date

from stanstock.research.config import load_scoring_config
from stanstock.research.explanations import generate_reasons, generate_risks
from stanstock.research.scoring import aggregate_score, assess_risk, score_components
from stanstock.research.types import IndicatorResult, ResearchValues


def test_explanations_are_deterministic_and_only_use_calculated_factors() -> None:
    config = load_scoring_config()
    indicators = IndicatorResult(
        values={
            "return_20d": 0.03,
            "return_63d": 0.12,
            "annualized_volatility": 0.42,
            "max_drawdown": -0.35,
        },
        last_date=date(2026, 9, 4),
    )
    fundamentals = ResearchValues(
        values={
            "net_margin": 0.22,
            "operating_margin": 0.25,
            "free_cash_flow_yield": 0.07,
            "debt_to_equity": 2.1,
        }
    )
    components = score_components(indicators, fundamentals, config)
    aggregate = aggregate_score(components, config)
    risk = assess_risk(indicators, fundamentals, config)

    first_reasons = generate_reasons(indicators, fundamentals, aggregate)
    second_reasons = generate_reasons(indicators, fundamentals, aggregate)
    risks = generate_risks(indicators, fundamentals, risk, aggregate)

    assert first_reasons == second_reasons
    assert any("Three-month price momentum" in reason for reason in first_reasons)
    assert any("Free cash flow yield" in reason for reason in first_reasons)
    assert not any("benchmark" in reason.lower() for reason in first_reasons)
    assert any("volatility" in item for item in risks)
    assert any("Debt to equity" in item for item in risks)
