from __future__ import annotations

from stanstock.research.types import AggregateScore, IndicatorResult, ResearchValues, RiskAssessment


def generate_reasons(
    indicators: IndicatorResult,
    fundamentals: ResearchValues,
    score: AggregateScore,
) -> list[str]:
    candidates: list[tuple[str, str]] = []
    quality = score.component_scores.components.get("quality")
    if quality is not None and quality >= 70:
        candidates.append(("quality", f"Quality score is strong at {quality:.0f}/100."))
    growth = score.component_scores.components.get("growth")
    if growth is not None and growth >= 70:
        candidates.append(("growth", f"Growth factors score {growth:.0f}/100."))
    valuation = score.component_scores.components.get("valuation")
    if valuation is not None and valuation >= 70:
        candidates.append(
            ("valuation", f"Valuation inputs screen attractively at {valuation:.0f}/100.")
        )
    momentum = indicators.get("momentum_63d")
    if momentum is None:
        momentum = indicators.get("return_63d")
    if momentum is not None and momentum > 0.05:
        candidates.append(
            ("momentum", f"Three-month price momentum is positive at {momentum:.1%}.")
        )
    fcf_yield = fundamentals.get("free_cash_flow_yield")
    if fcf_yield is not None and fcf_yield > 0.04:
        candidates.append(("fcf", f"Free cash flow yield is {fcf_yield:.1%}."))
    relative = indicators.get("relative_return_63d")
    if relative is not None and relative > 0.03:
        candidates.append(
            (
                "relative",
                f"The listing outperformed its benchmark by {relative:.1%} over 63 trading days.",
            )
        )
    return [message for _, message in candidates[:5]]


def generate_risks(
    indicators: IndicatorResult,
    fundamentals: ResearchValues,
    risk: RiskAssessment,
    score: AggregateScore,
) -> list[str]:
    candidates: list[tuple[str, str]] = []
    volatility = indicators.get("annualized_volatility")
    if volatility is not None and volatility > 0.35:
        candidates.append(("volatility", f"Annualized volatility is elevated at {volatility:.1%}."))
    drawdown = indicators.get("max_drawdown")
    if drawdown is not None and drawdown < -0.30:
        candidates.append(("drawdown", f"Observed max drawdown is {drawdown:.1%}."))
    leverage = fundamentals.get("debt_to_equity")
    if leverage is not None and leverage > 1.5:
        candidates.append(("leverage", f"Debt to equity is high at {leverage:.2f}x."))
    coverage = fundamentals.get("interest_coverage")
    if coverage is not None and coverage < 2.0:
        candidates.append(("interest", f"Interest coverage is thin at {coverage:.1f}x."))
    if score.component_scores.coverage < 0.5:
        candidates.append(
            ("missing", f"Input coverage is limited at {score.component_scores.coverage:.0%}.")
        )
    if risk.score is None:
        candidates.append(
            ("risk-missing", "Composite risk is unavailable because supported inputs are missing.")
        )
    elif risk.score >= 70:
        candidates.append(("risk", f"Composite risk score is high at {risk.score:.0f}/100."))
    return [message for _, message in candidates[:5]]
