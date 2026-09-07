from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from stanstock.data.models import Company, DataAsset, FundamentalFact
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.data.sec_fundamentals import build_sec_fundamental_series

pytestmark = pytest.mark.django_db


def _evidence() -> tuple[Company, DataAsset]:
    retrieved_at = datetime(2026, 3, 1, tzinfo=UTC)
    asset = DataAsset.objects.create(
        provider="sec",
        kind="sec_companyfacts",
        subject="0000000001",
        relative_path="tests/sec-companyfacts.json",
        sha256="a" * 64,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
    )
    return Company.objects.create(name="SEC Test", country="US"), asset


def _fact(
    company: Company,
    asset: DataAsset,
    *,
    concept: str,
    source_concept: str,
    value: str,
    start: date | None,
    end: date,
    fiscal_period: str,
    accession: str,
    available_at: datetime,
    unit: str = "USD",
    filing_form: str = "10-Q",
    source_revision: int = 1,
) -> FundamentalFact:
    return FundamentalFact.objects.create(
        company=company,
        provider="sec",
        concept=concept,
        taxonomy="us-gaap",
        source_concept=source_concept,
        value=Decimal(value),
        unit=unit,
        currency="USD" if unit.startswith("USD") else "",
        period_type=(
            FundamentalFact.PeriodType.DURATION
            if start is not None
            else FundamentalFact.PeriodType.INSTANT
        ),
        period_start=start,
        period_end=end,
        fiscal_year=2025,
        fiscal_period=fiscal_period,
        accession=accession,
        filing_form=filing_form,
        filing_date=available_at.date(),
        filed_at=available_at,
        acceptance_at=available_at,
        available_at=available_at,
        availability_basis="acceptance_datetime",
        is_amendment=filing_form.endswith("/A"),
        source_revision=source_revision,
        source_asset=asset,
    )


def test_sec_series_derives_discrete_quarters_ttm_and_fcf() -> None:
    company, asset = _evidence()
    available = datetime(2026, 2, 15, tzinfo=UTC)
    periods = (
        (date(2025, 1, 1), date(2025, 3, 31), "Q1"),
        (date(2025, 1, 1), date(2025, 6, 30), "Q2"),
        (date(2025, 1, 1), date(2025, 9, 30), "Q3"),
        (date(2025, 1, 1), date(2025, 12, 31), "FY"),
    )
    series = {
        "revenue": (
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            ("100", "220", "360", "520"),
            "USD",
        ),
        "operating_cash_flow": (
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
            ("30", "70", "120", "180"),
            "USD",
        ),
        "net_income": (
            "us-gaap:NetIncomeLoss",
            ("10", "22", "36", "52"),
            "USD",
        ),
        "capital_expenditure": (
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
            ("5", "12", "21", "33"),
            "USD",
        ),
        "weighted_average_diluted_shares": (
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
            ("10", "11", "12", "13"),
            "shares",
        ),
    }
    for concept, (source_concept, values, unit) in series.items():
        for index, ((start, end, fiscal_period), value) in enumerate(
            zip(periods, values, strict=True),
            start=1,
        ):
            _fact(
                company,
                asset,
                concept=concept,
                source_concept=source_concept,
                value=value,
                start=start,
                end=end,
                fiscal_period=fiscal_period,
                accession=f"2025-{index}",
                available_at=available + timedelta(days=index),
                unit=unit,
                filing_form="10-K" if fiscal_period == "FY" else "10-Q",
            )

    result = build_sec_fundamental_series(
        FundamentalFact.objects.filter(company=company),
        config=load_sec_fundamentals_config(),
    )

    assert [value.value for value in result.quarters["revenue"]] == [
        Decimal("100"),
        Decimal("120"),
        Decimal("140"),
        Decimal("160"),
    ]
    assert result.ttm["revenue"].value == Decimal("520")
    assert result.ttm["operating_cash_flow"].value == Decimal("180")
    assert result.ttm["capital_expenditure"].value == Decimal("33")
    assert result.ttm["free_cash_flow"].value == Decimal("147")
    assert result.annual["free_cash_flow"][-1].value == Decimal("147")
    assert result.ttm["weighted_average_diluted_shares"].value > Decimal("10")
    assert result.missing == {}


def test_latest_vintage_replaces_period_without_creating_growth_period() -> None:
    company, asset = _evidence()
    original = _fact(
        company,
        asset,
        concept="net_income",
        source_concept="us-gaap:NetIncomeLoss",
        value="100",
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        fiscal_period="FY",
        accession="original",
        available_at=datetime(2026, 2, 1, tzinfo=UTC),
        filing_form="10-K",
    )
    amendment = _fact(
        company,
        asset,
        concept="net_income",
        source_concept="us-gaap:NetIncomeLoss",
        value="95",
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        fiscal_period="FY",
        accession="amendment",
        available_at=datetime(2026, 2, 20, tzinfo=UTC),
        filing_form="10-K/A",
    )

    result = build_sec_fundamental_series(
        [original, amendment],
        config=load_sec_fundamentals_config(),
    )

    assert len(result.selected_facts) == 1
    assert result.selected_facts[0].pk == amendment.pk
    assert len(result.annual["net_income"]) == 1
    assert result.annual["net_income"][0].value == Decimal("95")


def test_latest_source_revision_wins_when_an_observation_reverts() -> None:
    company, asset = _evidence()
    common = {
        "company": company,
        "asset": asset,
        "concept": "net_income",
        "source_concept": "us-gaap:NetIncomeLoss",
        "start": date(2025, 1, 1),
        "end": date(2025, 12, 31),
        "fiscal_period": "FY",
        "accession": "same-accession",
        "filing_form": "10-K",
    }
    _fact(
        **common,
        value="100",
        available_at=datetime(2026, 2, 1, tzinfo=UTC),
        source_revision=1,
    )
    _fact(
        **common,
        value="101",
        available_at=datetime(2026, 2, 2, tzinfo=UTC),
        source_revision=2,
    )
    reverted = _fact(
        **common,
        value="100",
        available_at=datetime(2026, 2, 1, tzinfo=UTC),
        source_revision=3,
    )

    result = build_sec_fundamental_series(
        FundamentalFact.objects.filter(company=company),
        config=load_sec_fundamentals_config(),
    )

    assert result.selected_facts == (reverted,)
    assert result.annual["net_income"][0].value == Decimal("100")


def test_newer_restated_ytd_quarter_overrides_stale_direct_quarter() -> None:
    company, asset = _evidence()
    _fact(
        company,
        asset,
        concept="revenue",
        source_concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        value="130",
        start=date(2025, 4, 1),
        end=date(2025, 6, 30),
        fiscal_period="Q2",
        accession="old-direct-q2",
        available_at=datetime(2025, 7, 15, tzinfo=UTC),
    )
    _fact(
        company,
        asset,
        concept="revenue",
        source_concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        value="110",
        start=date(2025, 1, 1),
        end=date(2025, 3, 31),
        fiscal_period="Q1",
        accession="restated-q1",
        available_at=datetime(2026, 2, 1, tzinfo=UTC),
        filing_form="10-K/A",
    )
    _fact(
        company,
        asset,
        concept="revenue",
        source_concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        value="250",
        start=date(2025, 1, 1),
        end=date(2025, 6, 30),
        fiscal_period="Q2",
        accession="restated-h1",
        available_at=datetime(2026, 2, 1, tzinfo=UTC),
        filing_form="10-K/A",
    )

    result = build_sec_fundamental_series(
        FundamentalFact.objects.filter(company=company),
        config=load_sec_fundamentals_config(),
    )

    q2 = result.quarters["revenue"][-1]
    assert q2.value == Decimal("140")
    assert q2.derivation == "ytd_difference"
    assert q2.accessions == ("restated-q1", "restated-h1")


def test_ttm_rejects_quarters_with_uncovered_calendar_gaps() -> None:
    company, asset = _evidence()
    periods = (
        (date(2025, 1, 1), date(2025, 3, 27)),
        (date(2025, 4, 4), date(2025, 6, 28)),
        (date(2025, 7, 6), date(2025, 9, 29)),
        (date(2025, 10, 7), date(2025, 12, 31)),
    )
    for index, (start, end) in enumerate(periods, start=1):
        _fact(
            company,
            asset,
            concept="revenue",
            source_concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            value=str(index * 100),
            start=start,
            end=end,
            fiscal_period=f"Q{index}",
            accession=f"gap-q{index}",
            available_at=datetime(2026, 2, index, tzinfo=UTC),
        )

    result = build_sec_fundamental_series(
        FundamentalFact.objects.filter(company=company),
        config=load_sec_fundamentals_config(),
    )

    assert "revenue" not in result.ttm


def test_instant_facts_never_enter_duration_arithmetic() -> None:
    company, asset = _evidence()
    instant = _fact(
        company,
        asset,
        concept="assets",
        source_concept="us-gaap:Assets",
        value="500",
        start=None,
        end=date(2025, 12, 31),
        fiscal_period="FY",
        accession="instant",
        available_at=datetime(2026, 2, 1, tzinfo=UTC),
        filing_form="10-K",
    )

    result = build_sec_fundamental_series(
        [instant],
        config=load_sec_fundamentals_config(),
    )

    assert result.latest_instants["assets"].pk == instant.pk
    assert "assets" not in result.annual
    assert "assets" not in result.quarters
    assert "assets" not in result.ttm
