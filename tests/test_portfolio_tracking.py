from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from io import StringIO

import pytest
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.db import DatabaseError, transaction
from django.urls import reverse
from django.utils import timezone

from stanstock.core.models import JobRun
from stanstock.data.models import (
    Company,
    DataAsset,
    LatestMarketData,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.portfolio.jobs import execute_portfolio_snapshot_job
from stanstock.portfolio.models import Portfolio, PortfolioHolding, PortfolioSnapshot
from stanstock.portfolio.service import (
    SAMPLE_PORTFOLIO_POLICY,
    PortfolioValuationError,
    build_sample_portfolio,
    calculate_portfolio_valuation,
    portfolio_snapshot_series,
    record_portfolio_snapshot,
    restore_portfolio,
    upsert_holding,
)
from stanstock.research.models import (
    AnalysisRun,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.opportunities import assess_opportunity


@pytest.fixture
def owner():
    return get_user_model().objects.create_user(
        username="owner",
        password="correct-password",
    )


@pytest.fixture
def priced_listing() -> Listing:
    company = Company.objects.create(
        name="Portfolio Corp",
        country="US",
        sector="Technology",
    )
    security = Security.objects.create(company=company, name="Portfolio Common")
    listing = Listing.objects.create(
        security=security,
        ticker="PORT",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    now = timezone.now()
    asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject=listing.ticker,
        relative_path="tests/portfolio-port.parquet",
        sha256="a" * 64,
        retrieved_at=now,
        available_at=now,
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )
    LatestMarketData.objects.create(
        listing=listing,
        observed_at=now,
        session_date=date(2026, 9, 4),
        close=Decimal("100"),
        previous_close=Decimal("98"),
        volume=1_000_000,
        source_asset=asset,
    )
    return listing


def _analysis(listing: Listing) -> StockAnalysis:
    universe = Universe.objects.create(
        slug="portfolio-opportunities",
        name="Portfolio opportunities",
        config_version="test-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=date(2026, 9, 4),
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="b" * 64,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    now = timezone.now()
    run = AnalysisRun.objects.create(
        generated_at=now,
        data_cutoff=now,
        target_date=snapshot.as_of_date,
        universe_snapshot=snapshot,
        config_version="us-price-baseline-v1",
        config_hash="c" * 64,
        code_revision="test",
    )
    return StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("100"),
        daily_change=Decimal("0.02"),
        overall_score=Decimal("85"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("20"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("70"),
        short_scenario={"bear": -0.03, "base": 0.05, "bull": 0.11},
        data_quality={
            "analysis_mode": "price_only_baseline",
            "fundamentals_used": False,
        },
    )


def _provider_analysis_run(*, count: int = 3) -> tuple[AnalysisRun, list[Listing]]:
    universe = Universe.objects.create(
        slug="provider-opportunities",
        name="Provider opportunities",
        config_version="provider-test-v1",
    )
    target_date = date(2026, 9, 3)
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=target_date,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="e" * 64,
    )
    now = timezone.now()
    run = AnalysisRun.objects.create(
        generated_at=now,
        data_cutoff=now,
        target_date=target_date,
        universe_snapshot=snapshot,
        config_version="us-price-baseline-v1",
        config_hash="f" * 64,
        code_revision="provider-test",
    )
    listings: list[Listing] = []
    for index in range(count):
        ticker = f"LIVE{index + 1}"
        company = Company.objects.create(
            name=f"Live Company {index + 1}",
            country="US",
            sector="Technology",
        )
        security = Security.objects.create(company=company, name=f"{ticker} Common")
        listing = Listing.objects.create(
            security=security,
            ticker=ticker,
            provider_symbol=ticker,
            exchange_mic="XNAS",
            currency="USD",
            region=Region.US,
        )
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        price = Decimal(100 + index * 25)
        asset = DataAsset.objects.create(
            provider="twelve_data",
            kind="price_history",
            subject=ticker,
            relative_path=f"tests/{ticker.lower()}-live.parquet",
            sha256=f"{index + 1}" * 64,
            retrieved_at=now,
            available_at=now,
            metadata={
                "return_definition": "split_adjusted_price_return",
                "dividends_included": False,
            },
        )
        LatestMarketData.objects.create(
            listing=listing,
            observed_at=now,
            session_date=date(2026, 9, 4),
            close=price,
            previous_close=price - Decimal(1),
            volume=1_000_000,
            source_asset=asset,
        )
        StockAnalysis.objects.create(
            run=run,
            listing=listing,
            current_price=price,
            daily_change=Decimal("0.01"),
            overall_score=Decimal(90 - index),
            recommendation=Recommendation.BUY,
            risk_score=Decimal("20"),
            risk_class=RiskClass.LOW,
            confidence=Decimal("80"),
            short_scenario={"bear": -0.03, "base": 0.05, "bull": 0.11},
            data_quality={
                "analysis_mode": "price_only_baseline",
                "fundamentals_used": False,
                "source_assets": [
                    {
                        "id": str(asset.pk),
                        "provider": "twelve_data",
                        "kind": "price_history",
                        "subject": ticker,
                    }
                ],
            },
        )
        listings.append(listing)
    return run, listings


@pytest.mark.django_db
def test_portfolio_valuation_excludes_cash_from_return(owner, priced_listing: Listing) -> None:
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Long term",
        base_currency="USD",
        cash_balance=Decimal("1000"),
    )
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("10"),
        average_cost=Decimal("80"),
    )

    valuation = calculate_portfolio_valuation(portfolio)

    assert valuation.complete is True
    assert valuation.securities_value == Decimal("1000")
    assert valuation.total_value == Decimal("2000")
    assert valuation.cost_basis == Decimal("800")
    assert valuation.unrealized_gain == Decimal("200")
    assert valuation.return_pct == Decimal("0.25")


@pytest.mark.django_db
def test_snapshot_is_idempotent_and_same_day_holding_change_is_preserved(
    owner,
    priced_listing: Listing,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Core", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("2"),
        average_cost=Decimal("80"),
    )

    first, first_created = record_portfolio_snapshot(portfolio)
    duplicate, duplicate_created = record_portfolio_snapshot(portfolio)
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("3"),
        average_cost=Decimal("80"),
    )
    changed, changed_created = record_portfolio_snapshot(portfolio)

    assert first_created is True
    assert duplicate_created is False
    assert duplicate.pk == first.pk
    assert changed_created is True
    assert changed.as_of_date == first.as_of_date
    assert PortfolioSnapshot.objects.filter(portfolio=portfolio).count() == 2


@pytest.mark.django_db
def test_snapshot_deduplication_is_stable_across_code_revisions(
    owner,
    priced_listing: Listing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Deploy", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("2"),
        average_cost=Decimal("80"),
    )
    monkeypatch.setattr("stanstock.portfolio.service.code_revision", lambda: "revision-one")
    first, first_created = record_portfolio_snapshot(portfolio)
    monkeypatch.setattr("stanstock.portfolio.service.code_revision", lambda: "revision-two")
    duplicate, duplicate_created = record_portfolio_snapshot(portfolio)

    assert first_created is True
    assert duplicate_created is False
    assert duplicate.pk == first.pk
    assert duplicate.code_revision == "revision-one"


@pytest.mark.django_db
def test_snapshot_history_uses_recording_order_across_cash_and_market_dates(
    owner,
    priced_listing: Listing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Chronology",
        base_currency="USD",
        cash_balance=Decimal("500"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.service.timezone.localdate",
        lambda: date(2026, 9, 6),
    )
    cash_snapshot, _created = record_portfolio_snapshot(portfolio)
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("5"),
        average_cost=Decimal("90"),
    )
    invested_snapshot, _created = record_portfolio_snapshot(portfolio)

    history = portfolio_snapshot_series(portfolio)

    assert [snapshot.pk for snapshot in history] == [
        cash_snapshot.pk,
        invested_snapshot.pk,
    ]
    assert cash_snapshot.as_of_date == date(2026, 9, 6)
    assert invested_snapshot.as_of_date == date(2026, 9, 4)


@pytest.mark.django_db
def test_snapshot_and_snapshot_positions_are_immutable(
    owner,
    priced_listing: Listing,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Immutable", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("90"),
    )
    snapshot, _created = record_portfolio_snapshot(portfolio)
    position = snapshot.positions.get()

    snapshot.total_value = Decimal("1")
    with pytest.raises(ValidationError):
        snapshot.save()
    with pytest.raises(ValidationError):
        position.delete()
    with pytest.raises(DatabaseError), transaction.atomic():
        PortfolioSnapshot.objects.filter(pk=snapshot.pk).update(total_value=Decimal("1"))


@pytest.mark.django_db
def test_snapshot_refuses_stale_or_missing_prices(owner, priced_listing: Listing) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Stale", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("90"),
    )
    stale_company = Company.objects.create(name="Stale Corp", country="US")
    stale_security = Security.objects.create(company=stale_company)
    stale_listing = Listing.objects.create(
        security=stale_security,
        ticker="STALE",
        exchange_mic="XNYS",
        currency="USD",
        region=Region.US,
    )
    now = timezone.now()
    stale_asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject="STALE",
        relative_path="tests/portfolio-stale.parquet",
        sha256="d" * 64,
        retrieved_at=now,
        available_at=now,
    )
    LatestMarketData.objects.create(
        listing=stale_listing,
        observed_at=timezone.now(),
        session_date=date(2026, 9, 4) - timedelta(days=8),
        close=Decimal("50"),
        source_asset=stale_asset,
    )
    upsert_holding(
        portfolio=portfolio,
        listing=stale_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("45"),
    )

    valuation = calculate_portfolio_valuation(portfolio)

    assert valuation.complete is False
    assert any("STALE price is stale" in issue for issue in valuation.issues)
    with pytest.raises(PortfolioValuationError, match="STALE price is stale"):
        record_portfolio_snapshot(portfolio)


@pytest.mark.django_db
def test_absolute_price_staleness_warns_without_withholding_value(
    owner,
    priced_listing: Listing,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Frozen feed", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("90"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.service.timezone.localdate",
        lambda: date(2026, 9, 20),
    )

    valuation = calculate_portfolio_valuation(portfolio)

    assert valuation.complete is True
    assert valuation.total_value == Decimal("100")
    assert valuation.warnings == (
        "Latest portfolio prices are from 2026-09-04; current values may be outdated.",
    )


@pytest.mark.django_db
def test_snapshot_refuses_listing_deactivated_after_it_was_added(
    owner,
    priced_listing: Listing,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Inactive", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("90"),
    )
    priced_listing.is_active = False
    priced_listing.save(update_fields=["is_active"])

    valuation = calculate_portfolio_valuation(portfolio)

    assert valuation.complete is False
    assert valuation.issues == ("PORT is no longer an active listing.",)
    with pytest.raises(PortfolioValuationError, match="no longer an active"):
        record_portfolio_snapshot(portfolio)


@pytest.mark.django_db
def test_split_sized_price_move_is_flagged_without_rewriting_holding(
    owner,
    priced_listing: Listing,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Split", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("2"),
        average_cost=Decimal("80"),
    )
    first, _created = record_portfolio_snapshot(portfolio)
    market = LatestMarketData.objects.get(listing=priced_listing)
    market.close = Decimal("50")
    market.session_date = date(2026, 9, 5)
    market.save(update_fields=["close", "session_date"])

    valuation = calculate_portfolio_valuation(portfolio)
    second, created = record_portfolio_snapshot(portfolio)

    assert created is True
    assert first.corporate_action_warnings == 0
    assert second.corporate_action_warnings == 1
    assert second.positions.get().corporate_action_suspected is True
    assert valuation.return_pct == Decimal("-0.375")
    assert "model return is withheld" not in " ".join(valuation.warnings)
    assert "quantity and average cost may need review" in " ".join(valuation.warnings)


@pytest.mark.django_db
def test_snapshot_preserves_currency_when_portfolio_currency_changes(owner) -> None:
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Cash",
        base_currency="USD",
        cash_balance=Decimal("100"),
    )
    usd_snapshot, usd_created = record_portfolio_snapshot(portfolio)
    portfolio.base_currency = "EUR"
    portfolio.save(update_fields=["base_currency", "updated_at"])
    eur_snapshot, eur_created = record_portfolio_snapshot(portfolio)

    assert usd_created is True
    assert eur_created is True
    assert usd_snapshot.pk != eur_snapshot.pk
    assert usd_snapshot.base_currency == "USD"
    assert eur_snapshot.base_currency == "EUR"


@pytest.mark.django_db
def test_opportunity_policy_is_analysis_mode_aware(priced_listing: Listing) -> None:
    analysis = _analysis(priced_listing)

    assessment = assess_opportunity(analysis)

    assert assessment.eligible is True
    assert assessment.label == "Strong short-term setup"
    assert assessment.horizon == "short"
    assert assessment.policy_version == "great-opportunity-v2"
    assert assessment.price_band is not None
    assert assessment.price_band.slug == "50_to_300"
    analysis.short_scenario = {"bear": -0.03, "base": -0.01, "bull": 0.05}
    assert assess_opportunity(analysis).eligible is False


@pytest.mark.django_db
def test_sample_portfolio_is_deterministic_idempotent_and_frozen(owner) -> None:
    run, listings = _provider_analysis_run(count=3)

    portfolio, created = build_sample_portfolio(
        owner=owner,
        source_run=run,
        starting_capital=Decimal("100000"),
        top_n=3,
    )
    duplicate, duplicate_created = build_sample_portfolio(
        owner=owner,
        source_run=run,
        starting_capital=Decimal("100000"),
        top_n=3,
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.pk == portfolio.pk
    assert portfolio.source_analysis_run == run
    assert portfolio.construction_policy == SAMPLE_PORTFOLIO_POLICY
    assert portfolio.starting_capital == Decimal("100000")
    assert portfolio.construction_metadata["signal_horizon"] == "short"
    holdings = list(portfolio.holdings.order_by("notes"))
    assert [holding.listing_id for holding in holdings] == [listing.id for listing in listings]
    assert [holding.notes.split(";", 1)[0] for holding in holdings] == [
        "Rank 1",
        "Rank 2",
        "Rank 3",
    ]
    baseline = portfolio.snapshots.get()
    assert baseline.as_of_date == run.target_date
    assert baseline.total_value == Decimal("100000")
    assert baseline.return_pct == Decimal(0)
    assert set(baseline.positions.values_list("source_asset__provider", flat=True)) == {
        "twelve_data"
    }
    invested = sum(
        (holding.quantity * holding.average_cost for holding in holdings),
        Decimal(0),
    )
    assert (invested + portfolio.cash_balance).quantize(Decimal("0.000001")) == Decimal(
        "100000.000000"
    )

    with pytest.raises(PortfolioValuationError, match="frozen"):
        upsert_holding(
            portfolio=portfolio,
            listing=listings[0],
            quantity=Decimal("2"),
            average_cost=Decimal("100"),
        )
    with pytest.raises(ValidationError, match="frozen"):
        holdings[0].delete()
    portfolio.cash_balance += Decimal(1)
    with pytest.raises(ValidationError, match="frozen"):
        portfolio.save()
    portfolio.refresh_from_db()
    other_owner = get_user_model().objects.create_user(
        username="replacement-owner",
        password="replacement-password",
    )
    portfolio.owner = other_owner
    with pytest.raises(ValidationError, match="frozen"):
        portfolio.save()
    portfolio.refresh_from_db()
    portfolio.archived_at = timezone.now()
    portfolio.save(update_fields=["archived_at", "updated_at"])
    replacement, replacement_created = build_sample_portfolio(
        owner=owner,
        source_run=run,
        starting_capital=Decimal("100000"),
        top_n=3,
    )
    assert replacement_created is True
    assert replacement.pk != portfolio.pk
    with pytest.raises(PortfolioValuationError, match="Another active sample portfolio"):
        restore_portfolio(portfolio)


@pytest.mark.django_db
def test_sample_portfolio_rejects_non_provider_analysis(owner, priced_listing: Listing) -> None:
    analysis = _analysis(priced_listing)

    with pytest.raises(PortfolioValuationError, match="provider-backed"):
        build_sample_portfolio(owner=owner, source_run=analysis.run, top_n=1)


@pytest.mark.django_db
def test_under_10_is_excluded_from_new_sample_but_existing_holdings_remain_trackable(
    owner,
) -> None:
    run, listings = _provider_analysis_run(count=3)
    decision_analysis = StockAnalysis.objects.get(run=run, listing=listings[0])
    decision_analysis.current_price = Decimal("9.99")
    decision_analysis.save(update_fields=["current_price"])
    market_data = LatestMarketData.objects.get(listing=listings[0])
    market_data.close = Decimal("125")
    market_data.save(update_fields=["close"])

    sample, created = build_sample_portfolio(
        owner=owner,
        source_run=run,
        starting_capital=Decimal("100000"),
        top_n=3,
    )

    assert created is True
    assert list(
        sample.holdings.order_by("listing__ticker").values_list("listing_id", flat=True)
    ) == [
        listings[1].id,
        listings[2].id,
    ]
    assert sample.construction_metadata["price_band_policy_version"] == "us-price-bands-v1"
    assert sample.construction_metadata["excluded_new_allocation_price_bands"] == ["under_10"]
    assert {item["band"] for item in sample.construction_metadata["selection_price_bands"]} == {
        "50_to_300"
    }
    assert {
        item["price_date"] for item in sample.construction_metadata["selection_price_bands"]
    } == {run.target_date.isoformat()}
    assert {
        item["date_basis"] for item in sample.construction_metadata["selection_price_bands"]
    } == {"decision_target"}

    sample.archived_at = timezone.now()
    sample.save(update_fields=["archived_at", "updated_at"])
    market_data.close = Decimal("400")
    market_data.save(update_fields=["close"])
    rebuilt, rebuilt_created = build_sample_portfolio(
        owner=owner,
        source_run=run,
        starting_capital=Decimal("100000"),
        top_n=3,
    )
    assert rebuilt_created is True
    assert list(
        rebuilt.holdings.order_by("listing__ticker").values_list("listing_id", flat=True)
    ) == [
        listings[1].id,
        listings[2].id,
    ]

    tracked = Portfolio.objects.create(
        owner=owner,
        name="Existing speculative holding",
        base_currency="USD",
    )
    market_data.close = Decimal("9.99")
    market_data.save(update_fields=["close"])
    holding = upsert_holding(
        portfolio=tracked,
        listing=listings[0],
        quantity=Decimal("10"),
        average_cost=Decimal("12"),
    )
    valuation = calculate_portfolio_valuation(tracked)

    assert holding.listing_id == listings[0].id
    assert valuation.positions[0].market_data is not None
    assert valuation.positions[0].market_data.close == Decimal("9.99")


@pytest.mark.django_db
def test_invalid_ineligible_analysis_does_not_abort_sample_construction(owner) -> None:
    run, listings = _provider_analysis_run(count=3)
    ineligible = StockAnalysis.objects.get(run=run, listing=listings[0])
    ineligible.recommendation = Recommendation.HOLD
    ineligible.current_price = Decimal("0")
    ineligible.save(update_fields=["recommendation", "current_price"])

    portfolio, created = build_sample_portfolio(
        owner=owner,
        source_run=run,
        starting_capital=Decimal("100000"),
        top_n=2,
    )

    assert created is True
    assert list(
        portfolio.holdings.order_by("listing__ticker").values_list("listing_id", flat=True)
    ) == [
        listings[1].id,
        listings[2].id,
    ]


@pytest.mark.django_db
def test_sample_portfolio_web_flow_and_split_warning(client, owner) -> None:
    _run, listings = _provider_analysis_run(count=2)
    client.force_login(owner)

    response = client.post(
        reverse("portfolios"),
        {
            "action": "sample",
            "starting_capital": "100000.00",
            "top_n": "2",
        },
    )

    portfolio = Portfolio.objects.get(owner=owner, source_analysis_run__isnull=False)
    assert response.status_code == 302
    assert response.url == reverse("portfolio-detail", args=[portfolio.pk])
    detail = client.get(response.url)
    content = detail.content.decode()
    assert "Research-reference portfolio; composition is frozen." in content
    assert "1-10 trading days" in content
    assert "0.0%" in content

    market = LatestMarketData.objects.get(listing=listings[0])
    market.close = Decimal("50")
    market.session_date = date(2026, 9, 5)
    market.save(update_fields=["close", "session_date"])
    warned = client.get(response.url).content.decode()
    assert "Model price return" in warned
    assert "Withheld" in warned
    assert "split-sized price move" in warned
    record_portfolio_snapshot(portfolio)
    persistently_withheld = client.get(response.url).content.decode()
    assert "Model return withheld." in persistently_withheld
    assert "historical split warning" in persistently_withheld

    holding = portfolio.holdings.first()
    assert holding is not None
    removed = client.post(reverse("portfolio-holding-delete", args=[portfolio.pk, holding.pk]))
    assert removed.status_code == 302
    assert portfolio.holdings.count() == 2


@pytest.mark.django_db
def test_build_sample_portfolio_command_is_idempotent(owner) -> None:
    _provider_analysis_run(count=2)
    output = StringIO()

    call_command(
        "build_sample_portfolio",
        username=owner.username,
        starting_capital="100000",
        top_n=2,
        stdout=output,
    )
    call_command(
        "build_sample_portfolio",
        username=owner.username,
        starting_capital="100000",
        top_n=2,
        stdout=output,
    )

    assert (
        Portfolio.objects.filter(
            owner=owner,
            source_analysis_run__isnull=False,
        ).count()
        == 1
    )
    assert "already exists" in output.getvalue()


@pytest.mark.django_db
def test_portfolio_pages_create_track_highlight_and_isolate_owner(
    client,
    owner,
    priced_listing: Listing,
) -> None:
    _analysis(priced_listing)
    client.force_login(owner)

    created = client.post(
        reverse("portfolios"),
        {
            "name": "Growth",
            "description": "Tracked ideas",
            "base_currency": "USD",
            "cash_balance": "500",
        },
    )
    portfolio = Portfolio.objects.get(owner=owner, name="Growth")
    assert created.status_code == 302
    assert created.url == reverse("portfolio-detail", args=[portfolio.id])

    added = client.post(
        reverse("portfolio-detail", args=[portfolio.id]),
        {
            "action": "holding",
            "listing": str(priced_listing.id),
            "quantity": "5",
            "average_cost": "90",
            "acquired_on": "2026-09-01",
            "notes": "Core position",
        },
    )
    assert added.status_code == 302
    assert PortfolioHolding.objects.get(portfolio=portfolio).quantity == Decimal("5")

    detail = client.get(reverse("portfolio-detail", args=[portfolio.id]))
    content = detail.content.decode()
    assert detail.status_code == 200
    assert "PORT" in content
    assert "Strong short-term setup" in content
    assert PortfolioSnapshot.objects.filter(portfolio=portfolio).count() == 2

    other = get_user_model().objects.create_user(
        username="other",
        password="correct-password",
    )
    client.force_login(other)
    assert client.get(reverse("portfolio-detail", args=[portfolio.id])).status_code == 404


@pytest.mark.django_db
def test_snapshot_portfolios_command_is_observable_and_idempotent(
    owner,
    priced_listing: Listing,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Scheduled", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("90"),
    )
    output = StringIO()

    call_command(
        "snapshot_portfolios",
        target_date="2026-09-06",
        stdout=output,
    )
    call_command(
        "snapshot_portfolios",
        target_date="2026-09-06",
        stdout=output,
    )

    assert PortfolioSnapshot.objects.filter(portfolio=portfolio).count() == 1
    assert list(
        JobRun.objects.filter(job_name="snapshot_portfolios")
        .order_by("attempt")
        .values_list("status", flat=True)
    ) == [JobRun.Status.SUCCESS, JobRun.Status.SKIPPED]


@pytest.mark.django_db
def test_scheduled_snapshot_requires_the_resolved_xnys_session(
    owner,
    priced_listing: Listing,
) -> None:
    portfolio = Portfolio.objects.create(owner=owner, name="Session-bound", base_currency="USD")
    upsert_holding(
        portfolio=portfolio,
        listing=priced_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("90"),
    )

    with pytest.raises(PortfolioValuationError, match="required XNYS session 2026-09-05"):
        record_portfolio_snapshot(
            portfolio,
            expected_as_of_date=date(2026, 9, 5),
        )


@pytest.mark.django_db
def test_snapshot_portfolios_records_partial_failures_without_losing_success(
    owner,
    priced_listing: Listing,
) -> None:
    healthy = Portfolio.objects.create(owner=owner, name="Healthy", base_currency="USD")
    upsert_holding(
        portfolio=healthy,
        listing=priced_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("90"),
    )
    broken = Portfolio.objects.create(owner=owner, name="Broken", base_currency="USD")
    inactive_company = Company.objects.create(name="Inactive Corp", country="US")
    inactive_security = Security.objects.create(company=inactive_company)
    inactive_listing = Listing.objects.create(
        security=inactive_security,
        ticker="INACTIVE",
        exchange_mic="XNYS",
        currency="USD",
        region=Region.US,
        is_active=False,
    )
    PortfolioHolding.objects.create(
        portfolio=broken,
        listing=inactive_listing,
        quantity=Decimal("1"),
        average_cost=Decimal("10"),
    )

    call_command(
        "snapshot_portfolios",
        target_date="2026-09-07",
        stdout=StringIO(),
    )

    run = JobRun.objects.get(job_name="snapshot_portfolios")
    assert run.status == JobRun.Status.SUCCESS
    assert run.details["snapshots_created"] == 1
    assert run.details["snapshots_unchanged"] == 0
    assert run.details["failures"] == [
        "Broken: Portfolio snapshot was not recorded: INACTIVE is no longer an active listing."
    ]

    with pytest.raises(ValueError, match="Broken"):
        execute_portfolio_snapshot_job(
            target_date=date(2026, 9, 7),
            require_all=True,
        )

    strict_run = JobRun.objects.get(
        job_name="scheduled_portfolio_snapshots",
        target_date=date(2026, 9, 7),
    )
    assert strict_run.status == JobRun.Status.FAILED
