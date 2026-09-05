from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from stanstock.research.fundamentals import (
    calculate_fundamentals,
    inputs_from_fact_maps,
    inputs_from_facts,
)


@dataclass(frozen=True, slots=True)
class FactStub:
    concept: str
    value: Decimal
    period_end: date
    available_at: datetime | None = None


def test_fundamental_calculations_from_normalized_fact_mappings() -> None:
    inputs = inputs_from_fact_maps(
        {
            "revenue": 1200,
            "gross_profit": 600,
            "operating_income": 240,
            "net_income": 180,
            "free_cash_flow": 150,
            "cash_and_equivalents": 300,
            "total_debt": 200,
            "shareholders_equity": 500,
            "total_assets": 1600,
            "interest_expense": -40,
            "current_assets": 700,
            "current_liabilities": 350,
            "nopat": 190,
            "invested_capital": 600,
            "shares_outstanding": 10,
            "ebitda": 260,
        },
        previous={"revenue": 1000, "net_income": 150, "free_cash_flow": 100},
        history=(
            {"free_cash_flow": 80},
            {"free_cash_flow": 100},
            {"free_cash_flow": 150},
        ),
    )

    result = calculate_fundamentals(inputs, price=50)

    assert result.values["revenue_growth"] == pytest.approx(0.20)
    assert result.values["gross_margin"] == pytest.approx(0.50)
    assert result.values["cash_to_debt"] == pytest.approx(1.50)
    assert result.values["debt_to_equity"] == pytest.approx(0.40)
    assert result.values["interest_coverage"] == pytest.approx(6.0)
    assert result.values["roe"] == pytest.approx(180 / 500)
    assert result.values["current_ratio"] == pytest.approx(2.0)
    assert result.values["roic"] == pytest.approx(190 / 600)
    assert result.values["free_cash_flow_consistency"] == pytest.approx(1.0)
    assert result.values["pe_ratio"] == pytest.approx(500 / 180)
    assert result.values["ps_ratio"] == pytest.approx(500 / 1200)
    assert result.values["pb_ratio"] == pytest.approx(1.0)
    assert result.values["ev_to_sales"] == pytest.approx((500 + 200 - 300) / 1200)
    assert result.values["ev_to_ebitda"] == pytest.approx(400 / 260)
    assert result.values["ev_to_ebit"] == pytest.approx(400 / 240)
    assert result.values["price_to_fcf"] == pytest.approx(500 / 150)
    assert result.values["free_cash_flow_yield"] == pytest.approx(150 / 500)


def test_invalid_valuation_inputs_remain_missing() -> None:
    inputs = inputs_from_fact_maps(
        {"revenue": 100, "net_income": -10, "shares_outstanding": 5},
        previous={"revenue": 0},
    )

    result = calculate_fundamentals(inputs, price=10)

    assert "pe_ratio" not in result.values
    assert "pe_ratio" in result.missing
    assert "revenue_growth" not in result.values
    assert "revenue_growth" in result.missing
    assert "ps_ratio" in result.values
    assert "roic" in result.missing
    assert "current_ratio" in result.missing


def test_facts_use_latest_and_previous_per_concept() -> None:
    inputs = inputs_from_facts(
        [
            FactStub("revenue", Decimal("100"), date(2024, 12, 31)),
            FactStub("revenue", Decimal("120"), date(2025, 12, 31)),
            FactStub("net_income", Decimal("12"), date(2024, 9, 30)),
            FactStub("net_income", Decimal("18"), date(2025, 9, 30)),
        ]
    )

    result = calculate_fundamentals(inputs)

    assert result.values["revenue_growth"] == pytest.approx(0.20)
    assert result.values["net_income_growth"] == pytest.approx(0.50)


def test_source_concepts_are_mapped_and_restatements_replace_the_same_period() -> None:
    inputs = inputs_from_facts(
        [
            FactStub(
                "Revenue",
                Decimal("100"),
                date(2024, 12, 31),
                datetime(2025, 2, 1, tzinfo=UTC),
            ),
            FactStub(
                "Revenue",
                Decimal("150"),
                date(2025, 12, 31),
                datetime(2026, 2, 1, tzinfo=UTC),
            ),
            FactStub(
                "Revenue",
                Decimal("160"),
                date(2025, 12, 31),
                datetime(2026, 4, 1, tzinfo=UTC),
            ),
            FactStub(
                "NetIncomeLoss",
                Decimal("16"),
                date(2025, 12, 31),
                datetime(2026, 4, 1, tzinfo=UTC),
            ),
            FactStub(
                "Assets",
                Decimal("500"),
                date(2025, 12, 31),
                datetime(2026, 4, 1, tzinfo=UTC),
            ),
            FactStub(
                "StockholdersEquity",
                Decimal("200"),
                date(2025, 12, 31),
                datetime(2026, 4, 1, tzinfo=UTC),
            ),
        ]
    )

    result = calculate_fundamentals(inputs)

    assert result.values["revenue_growth"] == pytest.approx(0.60)
    assert result.values["net_margin"] == pytest.approx(0.10)
    assert result.values["roe"] == pytest.approx(0.08)
    assert "debt_to_assets" not in result.values
