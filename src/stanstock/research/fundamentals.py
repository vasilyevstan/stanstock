from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Protocol

from stanstock.research.types import FundamentalInputs, ResearchValues

FactValue = float | int | Decimal | str | None


class FactLike(Protocol):
    concept: str
    value: Decimal
    period_end: date


GROWTH_PAIRS = {
    "revenue_growth": "revenue",
    "net_income_growth": "net_income",
    "free_cash_flow_growth": "free_cash_flow",
}

CANONICAL_CONCEPTS = {
    "revenue": "revenue",
    "revenues": "revenue",
    "salesrevenuenet": "revenue",
    "grossprofit": "gross_profit",
    "operatingincome": "operating_income",
    "operatingincomeloss": "operating_income",
    "netincome": "net_income",
    "netincomeloss": "net_income",
    "freecashflow": "free_cash_flow",
    "cash": "cash",
    "cashandcashequivalentsatcarryingvalue": "cash_and_equivalents",
    "cashandcashequivalents": "cash_and_equivalents",
    "cashandshortterminvestments": "cash_and_short_term_investments",
    "debt": "debt",
    "longtermdebtandfinancingleaseobligationscurrent": "short_term_debt_and_long_term_debt",
    "longtermdebtcurrent": "short_term_debt_and_long_term_debt",
    "longtermdebtnoncurrent": "total_debt",
    "stockholdersequity": "equity",
    "stockholdersequityincludingportionattributabletononcontrollinginterest": "equity",
    "shareholdersequity": "shareholders_equity",
    "assets": "assets",
    "currentassets": "current_assets",
    "assetscurrent": "current_assets",
    "currentliabilities": "current_liabilities",
    "liabilitiescurrent": "current_liabilities",
    "interestexpense": "interest_expense",
    "interestanddebtexpense": "interest_and_debt_expense",
    "weightedaveragenumberofsharesoutstandingbasic": "weighted_average_shares",
    "commonstocksharesoutstanding": "shares_outstanding",
    "ebitda": "ebitda",
}


def inputs_from_fact_maps(
    current: Mapping[str, FactValue],
    previous: Mapping[str, FactValue] | None = None,
    history: Iterable[Mapping[str, FactValue]] = (),
) -> FundamentalInputs:
    return FundamentalInputs(
        current={
            key: value for key, raw in current.items() if (value := _to_float(raw)) is not None
        },
        previous={
            key: value
            for key, raw in (previous or {}).items()
            if (value := _to_float(raw)) is not None
        },
        history=tuple(
            {key: value for key, raw in row.items() if (value := _to_float(raw)) is not None}
            for row in history
        ),
    )


def inputs_from_facts(facts: Iterable[FactLike]) -> FundamentalInputs:
    selected: dict[tuple[str, date], tuple[tuple[float, int], float]] = {}
    for index, fact in enumerate(facts):
        value = _to_float(fact.value)
        if value is None:
            continue
        concept = _canonical_concept(fact.concept)
        key = (concept, fact.period_end)
        rank = (_availability_timestamp(fact), index)
        if key not in selected or rank >= selected[key][0]:
            selected[key] = (rank, value)

    by_period: dict[date, dict[str, float]] = {}
    by_concept: dict[str, list[tuple[date, float]]] = {}
    for (concept, period_end), (_rank, value) in selected.items():
        by_period.setdefault(period_end, {})[concept] = value
        by_concept.setdefault(concept, []).append((period_end, value))
    ordered = [by_period[period] for period in sorted(by_period)]
    if not ordered or not by_concept:
        return FundamentalInputs(current={})
    current: dict[str, float] = {}
    previous: dict[str, float] = {}
    for concept, observations in by_concept.items():
        sorted_observations = sorted(observations, key=lambda item: item[0])
        current[concept] = sorted_observations[-1][1]
        if len(sorted_observations) >= 2:
            previous[concept] = sorted_observations[-2][1]
    return FundamentalInputs(current=current, previous=previous, history=tuple(ordered))


def _canonical_concept(concept: str) -> str:
    normalized = "".join(character.lower() for character in concept if character.isalnum())
    return CANONICAL_CONCEPTS.get(normalized, concept)


def _availability_timestamp(fact: FactLike) -> float:
    available_at = getattr(fact, "available_at", None)
    if not isinstance(available_at, datetime):
        return float("-inf")
    if available_at.tzinfo is None:
        available_at = available_at.replace(tzinfo=UTC)
    return available_at.timestamp()


def calculate_fundamentals(
    inputs: FundamentalInputs,
    *,
    price: float | None = None,
) -> ResearchValues:
    current = inputs.current
    previous = inputs.previous
    values: dict[str, float] = {}
    missing: dict[str, str] = {}

    for output, concept in GROWTH_PAIRS.items():
        growth = _growth(current.get(concept), previous.get(concept))
        if growth is None:
            missing[output] = f"Need current and previous positive {concept}"
        else:
            values[output] = growth

    _ratio(values, missing, "gross_margin", current.get("gross_profit"), current.get("revenue"))
    _ratio(
        values, missing, "operating_margin", current.get("operating_income"), current.get("revenue")
    )
    _ratio(values, missing, "net_margin", current.get("net_income"), current.get("revenue"))
    _ratio(
        values,
        missing,
        "free_cash_flow_margin",
        current.get("free_cash_flow"),
        current.get("revenue"),
    )

    debt = _first_present(current, "debt", "total_debt", "short_term_debt_and_long_term_debt")
    cash = _first_present(
        current, "cash", "cash_and_equivalents", "cash_and_short_term_investments"
    )
    equity = _first_present(current, "equity", "shareholders_equity", "book_value")
    assets = _first_present(current, "assets", "total_assets")
    operating_income = current.get("operating_income")
    interest_expense = _absolute(
        _first_present(current, "interest_expense", "interest_and_debt_expense")
    )
    current_assets = _first_present(current, "current_assets", "total_current_assets")
    current_liabilities = _first_present(
        current, "current_liabilities", "total_current_liabilities"
    )

    _ratio(values, missing, "cash_to_debt", cash, debt)
    _ratio(values, missing, "debt_to_equity", debt, equity)
    _ratio(values, missing, "debt_to_assets", debt, assets)
    _ratio(values, missing, "interest_coverage", operating_income, interest_expense)
    _ratio(values, missing, "roe", current.get("net_income"), equity)
    _ratio(values, missing, "current_ratio", current_assets, current_liabilities)
    _ratio(values, missing, "roic", _nopat(current), _invested_capital(current, debt, cash, equity))

    fcf_consistency = _fcf_consistency(inputs.history)
    if fcf_consistency is None:
        missing["free_cash_flow_consistency"] = "Need at least three free cash flow observations"
    else:
        values["free_cash_flow_consistency"] = fcf_consistency

    market_cap = _market_cap(current, price)
    enterprise_value = _first_present(current, "enterprise_value", "ev")
    if enterprise_value is None and market_cap is not None:
        enterprise_value = market_cap + (debt or 0.0) - (cash or 0.0)

    _positive_multiple(values, missing, "pe_ratio", market_cap, current.get("net_income"))
    _positive_multiple(values, missing, "ps_ratio", market_cap, current.get("revenue"))
    _positive_multiple(values, missing, "pb_ratio", market_cap, equity)
    _positive_multiple(values, missing, "ev_to_sales", enterprise_value, current.get("revenue"))
    _positive_multiple(values, missing, "ev_to_ebitda", enterprise_value, current.get("ebitda"))
    _positive_multiple(values, missing, "ev_to_ebit", enterprise_value, operating_income)
    _positive_multiple(values, missing, "price_to_fcf", market_cap, current.get("free_cash_flow"))
    _ratio(values, missing, "free_cash_flow_yield", current.get("free_cash_flow"), market_cap)

    return ResearchValues(values=values, missing=missing)


def _to_float(raw: FactValue) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value or value in {float("inf"), float("-inf")}:
        return None
    return value


def _first_present(mapping: Mapping[str, float], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _absolute(value: float | None) -> float | None:
    if value is None:
        return None
    return abs(value)


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous <= 0:
        return None
    return current / previous - 1.0


def _ratio(
    values: dict[str, float],
    missing: dict[str, str],
    name: str,
    numerator: float | None,
    denominator: float | None,
) -> None:
    if numerator is None or denominator is None or denominator <= 0:
        missing[name] = "Need numerator and positive denominator"
        return
    values[name] = numerator / denominator


def _positive_multiple(
    values: dict[str, float],
    missing: dict[str, str],
    name: str,
    numerator: float | None,
    denominator: float | None,
) -> None:
    if numerator is None or denominator is None or numerator <= 0 or denominator <= 0:
        missing[name] = "Need positive market value and positive denominator"
        return
    values[name] = numerator / denominator


def _market_cap(mapping: Mapping[str, float], price: float | None) -> float | None:
    direct = _first_present(mapping, "market_cap", "market_capitalization")
    if direct is not None and direct > 0:
        return direct
    shares = _first_present(mapping, "shares_outstanding", "weighted_average_shares")
    if shares is not None and shares > 0 and price is not None and price > 0:
        return shares * price
    return None


def _nopat(mapping: Mapping[str, float]) -> float | None:
    return _first_present(
        mapping,
        "nopat",
        "net_operating_profit_after_tax",
        "after_tax_operating_income",
    )


def _invested_capital(
    mapping: Mapping[str, float],
    debt: float | None,
    cash: float | None,
    equity: float | None,
) -> float | None:
    direct = _first_present(mapping, "invested_capital", "total_invested_capital")
    if direct is not None and direct > 0:
        return direct
    if debt is None or cash is None or equity is None:
        return None
    invested_capital = debt + equity - cash
    if invested_capital <= 0:
        return None
    return invested_capital


def _fcf_consistency(history: Iterable[Mapping[str, float]]) -> float | None:
    observations = [row["free_cash_flow"] for row in history if "free_cash_flow" in row]
    if len(observations) < 3:
        return None
    return sum(1.0 for value in observations if value > 0) / len(observations)


def serializable_fundamental_inputs(inputs: FundamentalInputs) -> dict[str, Any]:
    return {
        "current": inputs.current,
        "previous": inputs.previous,
        "history_count": len(inputs.history),
    }
