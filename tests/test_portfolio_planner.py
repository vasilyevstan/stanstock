from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import count
from threading import Barrier, Event
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from django.contrib import admin
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.db import DatabaseError, connection, connections, transaction
from django.test import RequestFactory
from django.urls import reverse

from stanstock.data.etfs import ensure_investable_spy_listing
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
from stanstock.portfolio.admin import PortfolioAdmin, PortfolioHoldingAdmin
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioDeposit,
    PortfolioHolding,
    PortfolioPlanExecution,
    PortfolioPurchase,
)
from stanstock.portfolio.planner import (
    PortfolioPlanningError,
    calculate_contribution_performance,
    confirm_monthly_contribution_plan,
    preview_monthly_contribution_plan,
    record_external_deposit,
)
from stanstock.portfolio.service import (
    delete_holding,
    record_portfolio_snapshot,
    upsert_holding,
)
from stanstock.research.models import (
    AnalysisRun,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.web.forms import PortfolioForm

pytestmark = pytest.mark.django_db

TARGET_DATE = date(2026, 9, 4)
OBSERVED_AT = datetime(2026, 9, 5, 1, tzinfo=UTC)
TEST_NOW = datetime(2026, 9, 7, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = count()
    monkeypatch.setattr(
        "django.utils.timezone.now",
        lambda: TEST_NOW + timedelta(microseconds=next(ticks)),
    )


def _hold_snapshot_transaction(
    *,
    portfolio_id: UUID,
    snapshot_recorded: Event,
    release_snapshot: Event,
) -> None:
    connections.close_all()
    try:
        with transaction.atomic():
            portfolio = Portfolio.objects.get(pk=portfolio_id)
            record_portfolio_snapshot(portfolio)
            snapshot_recorded.set()
            assert release_snapshot.wait(timeout=20)
    finally:
        connections.close_all()


@pytest.fixture
def owner():
    return get_user_model().objects.create_user(
        username="planner-owner",
        password="test-password",
    )


def _market_asset(ticker: str, marker: str) -> DataAsset:
    return DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject=ticker,
        relative_path=f"tests/planner-{ticker.lower()}-{marker}.parquet",
        sha256=marker * 64,
        retrieved_at=OBSERVED_AT,
        available_at=OBSERVED_AT,
        period_start=TARGET_DATE,
        period_end=TARGET_DATE,
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )


def _spy(*, price: Decimal = Decimal("100")) -> Listing:
    listing = ensure_investable_spy_listing(
        currency="USD",
        mic_code="ARCX",
        valid_from=TARGET_DATE,
    )
    LatestMarketData.objects.create(
        listing=listing,
        observed_at=OBSERVED_AT,
        session_date=TARGET_DATE,
        close=price,
        previous_close=price - Decimal(1),
        volume=10_000_000,
        source_asset=_market_asset("SPY", "a"),
    )
    return listing


def _stock(
    *,
    ticker: str,
    price: Decimal,
    score: Decimal,
    marker: str,
    run: AnalysisRun,
) -> Listing:
    company = Company.objects.create(
        name=f"{ticker} Corp",
        country="US",
        sector="Technology",
    )
    security = Security.objects.create(
        company=company,
        name=f"{ticker} Common",
        security_type=Security.SecurityType.COMMON_STOCK,
    )
    listing = Listing.objects.create(
        security=security,
        ticker=ticker,
        provider_symbol=ticker,
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    UniverseMembership.objects.create(
        snapshot=run.universe_snapshot,
        listing=listing,
        eligible=True,
    )
    asset = _market_asset(ticker, marker)
    LatestMarketData.objects.create(
        listing=listing,
        observed_at=OBSERVED_AT,
        session_date=TARGET_DATE,
        close=price,
        previous_close=price - Decimal(1),
        volume=5_000_000,
        source_asset=asset,
    )
    StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=price,
        daily_change=Decimal("0.01"),
        overall_score=score,
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
    return listing


def _provider_run() -> AnalysisRun:
    universe = Universe.objects.create(
        slug=f"planner-{uuid4()}",
        name="Planner universe",
        config_version="planner-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="f" * 64,
    )
    return AnalysisRun.objects.create(
        generated_at=OBSERVED_AT,
        data_cutoff=OBSERVED_AT,
        target_date=TARGET_DATE,
        universe_snapshot=snapshot,
        config_version="us-price-baseline-v2",
        config_hash="e" * 64,
        code_revision="planner-test",
        status="complete",
    )


def _planner_market(
    *,
    spy_price: Decimal = Decimal("100"),
    stock_prices: tuple[Decimal, ...] = (Decimal("100"),),
) -> tuple[Listing, list[Listing]]:
    spy = _spy(price=spy_price)
    run = _provider_run()
    stocks = [
        _stock(
            ticker=f"PLAN{index + 1}",
            price=price,
            score=Decimal(95 - index),
            marker=str(index + 1),
            run=run,
        )
        for index, price in enumerate(stock_prices)
    ]
    return spy, stocks


def test_external_deposit_is_idempotent_immutable_and_flow_adjusted(owner) -> None:
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Contribution ledger",
        base_currency="USD",
    )
    request_id = uuid4()

    first, created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=request_id,
        note="September",
    )
    duplicate, duplicate_created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=request_id,
        note="September",
    )
    second, second_created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("200"),
        idempotency_key=uuid4(),
        note="October",
    )
    portfolio.refresh_from_db()
    performance = calculate_contribution_performance(portfolio)

    assert created is True
    assert duplicate_created is False
    assert duplicate.pk == first.pk
    assert second_created is True
    assert portfolio.cash_balance == Decimal("800")
    assert first.boundary_snapshot is not None
    assert first.boundary_snapshot.total_value == Decimal(0)
    assert second.boundary_snapshot is not None
    assert second.boundary_snapshot.total_value == Decimal("600")
    assert performance.total_contributions == Decimal("800")
    assert performance.investment_profit_loss == Decimal(0)
    assert performance.return_pct == Decimal(0)

    first.note = "changed"
    with pytest.raises(ValidationError, match="immutable"):
        first.save()
    with pytest.raises(DatabaseError), transaction.atomic():
        PortfolioDeposit.objects.filter(pk=first.pk).update(amount=Decimal("1"))


def test_manual_holding_addition_restarts_contribution_performance(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Manual opening holding",
        base_currency="USD",
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    upsert_holding(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    portfolio.refresh_from_db()

    performance = calculate_contribution_performance(portfolio)
    baseline = portfolio.performance_baselines.get()

    assert performance.total_contributions == Decimal("600")
    assert performance.tracked_contributions == Decimal(0)
    assert performance.tracking_start_value == Decimal("700")
    assert performance.investment_profit_loss == Decimal(0)
    assert performance.return_pct == Decimal(0)
    assert performance.withheld_reason == ""
    baseline.note = "changed"
    with pytest.raises(ValidationError, match="immutable"):
        baseline.save()
    with pytest.raises(DatabaseError), transaction.atomic():
        portfolio.performance_baselines.filter(pk=baseline.pk).update(note="changed")


def test_duplicate_holding_removal_is_idempotent_after_portfolio_lock(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Duplicate removal",
        base_currency="USD",
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    holding = upsert_holding(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    stale_holding = PortfolioHolding.objects.get(pk=holding.pk)

    delete_holding(holding)
    delete_holding(stale_holding)
    portfolio.refresh_from_db()

    assert not PortfolioHolding.objects.filter(pk=holding.pk).exists()
    assert portfolio.performance_baselines.count() == 2
    assert calculate_contribution_performance(portfolio).return_pct == Decimal(0)


def test_unpriceable_coholding_does_not_block_manual_recovery(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market(
        stock_prices=(Decimal("100"), Decimal("100")),
    )
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Recover invalid holdings",
        base_currency="USD",
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    first = upsert_holding(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    second = upsert_holding(
        portfolio=portfolio,
        listing=stocks[1],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    Listing.objects.filter(pk__in=[stocks[0].pk, stocks[1].pk]).update(is_active=False)

    delete_holding(first)
    portfolio.refresh_from_db()
    unavailable = portfolio.performance_baselines.get()

    assert not PortfolioHolding.objects.filter(pk=first.pk).exists()
    assert PortfolioHolding.objects.filter(pk=second.pk).exists()
    assert unavailable.snapshot is None
    assert "no longer an active listing" in unavailable.boundary_issue
    assert calculate_contribution_performance(portfolio).return_pct is None

    delete_holding(second)
    portfolio.refresh_from_db()
    recovered = calculate_contribution_performance(portfolio)

    assert not portfolio.holdings.exists()
    assert portfolio.performance_baselines.count() == 2
    assert portfolio.performance_baselines.first().snapshot is not None
    assert recovered.investment_profit_loss == Decimal(0)
    assert recovered.return_pct == Decimal(0)


def test_fresh_600_preview_is_side_effect_free_and_targets_70_30(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Fresh monthly plan",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )

    before = (
        PortfolioHolding.objects.count(),
        PortfolioPlanExecution.objects.count(),
        PortfolioPurchase.objects.count(),
    )
    plan = preview_monthly_contribution_plan(portfolio)

    assert plan.issues == ()
    assert plan.spy_budget == Decimal("420.000000")
    assert plan.satellite_budget == Decimal("180.000000")
    assert [(purchase.listing, purchase.amount) for purchase in plan.purchases] == [
        (spy, Decimal("420.000000")),
        (stocks[0], Decimal("180.000000")),
    ]
    satellite = plan.purchases[1]
    assert satellite.qualification is not None
    assert satellite.qualification["analysis_id"] == StockAnalysis.objects.get(listing=stocks[0]).pk
    assert satellite.qualification["opportunity"]["horizon"] == "short"  # type: ignore[index]
    assert plan.residual_cash == Decimal("0.000000")
    assert (
        PortfolioHolding.objects.count(),
        PortfolioPlanExecution.objects.count(),
        PortfolioPurchase.objects.count(),
    ) == before


def test_replaced_same_session_analysis_invalidates_plan_and_is_preserved(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Research-bound plan",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    original_plan = preview_monthly_contribution_plan(portfolio)
    original_analysis = StockAnalysis.objects.get(listing=stocks[0])
    replacement_run = AnalysisRun.objects.create(
        generated_at=OBSERVED_AT + timedelta(minutes=1),
        data_cutoff=OBSERVED_AT,
        target_date=TARGET_DATE,
        universe_snapshot=original_analysis.run.universe_snapshot,
        config_version="us-price-baseline-v2",
        config_hash="d" * 64,
        code_revision="replacement-research",
        status="complete",
    )
    replacement = StockAnalysis.objects.create(
        run=replacement_run,
        listing=stocks[0],
        current_price=original_analysis.current_price,
        daily_change=original_analysis.daily_change,
        overall_score=original_analysis.overall_score,
        recommendation=original_analysis.recommendation,
        risk_score=original_analysis.risk_score,
        risk_class=original_analysis.risk_class,
        confidence=original_analysis.confidence,
        confidence_status=original_analysis.confidence_status,
        component_scores=original_analysis.component_scores,
        short_scenario=original_analysis.short_scenario,
        medium_scenario=original_analysis.medium_scenario,
        long_scenario=original_analysis.long_scenario,
        reasons=original_analysis.reasons,
        risks=original_analysis.risks,
        data_quality=original_analysis.data_quality,
    )

    replacement_plan = preview_monthly_contribution_plan(portfolio)

    assert replacement_plan.plan_hash != original_plan.plan_hash
    with pytest.raises(PortfolioPlanningError, match="review a new plan"):
        confirm_monthly_contribution_plan(
            portfolio=portfolio,
            expected_plan_hash=original_plan.plan_hash,
            idempotency_key=uuid4(),
        )
    execution, created = confirm_monthly_contribution_plan(
        portfolio=portfolio,
        expected_plan_hash=replacement_plan.plan_hash,
        idempotency_key=uuid4(),
    )
    satellite_metadata = next(
        purchase
        for purchase in execution.metadata["purchases"]
        if purchase["role"] == PortfolioPurchase.Role.SATELLITE
    )
    qualification = satellite_metadata["qualification"]

    assert created is True
    assert qualification["analysis_id"] == replacement.pk
    assert qualification["analysis_run_id"] == str(replacement_run.pk)
    assert qualification["config_hash"] == replacement_run.config_hash
    assert qualification["source_assets"][0]["sha256"]


def test_confirmed_plan_is_idempotent_and_cannot_reuse_stale_cash(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Confirmed monthly plan",
        base_currency="USD",
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    portfolio.refresh_from_db()
    plan = preview_monthly_contribution_plan(portfolio)
    execution_key = uuid4()

    execution, created = confirm_monthly_contribution_plan(
        portfolio=portfolio,
        expected_plan_hash=plan.plan_hash,
        idempotency_key=execution_key,
    )
    duplicate, duplicate_created = confirm_monthly_contribution_plan(
        portfolio=portfolio,
        expected_plan_hash=plan.plan_hash,
        idempotency_key=execution_key,
    )
    portfolio.refresh_from_db()
    performance = calculate_contribution_performance(portfolio)

    assert created is True
    assert duplicate_created is False
    assert duplicate.pk == execution.pk
    assert portfolio.cash_balance == Decimal("0")
    assert execution.purchases.count() == 2
    assert PortfolioHolding.objects.get(portfolio=portfolio, listing=spy).quantity == Decimal(
        "4.20000000"
    )
    assert PortfolioHolding.objects.get(
        portfolio=portfolio,
        listing=stocks[0],
    ).quantity == Decimal("1.80000000")
    assert performance.investment_profit_loss == Decimal(0)
    assert performance.return_pct == Decimal(0)
    execution.ending_cash = Decimal("1")
    with pytest.raises(ValidationError, match="immutable"):
        execution.save()
    purchase = execution.purchases.first()
    assert purchase is not None
    with pytest.raises(DatabaseError), transaction.atomic():
        PortfolioPurchase.objects.filter(pk=purchase.pk).update(amount=Decimal("1"))
    LatestMarketData.objects.filter(listing__in=[spy, stocks[0]]).update(close=Decimal("110"))
    gain = calculate_contribution_performance(portfolio)
    assert gain.investment_profit_loss == Decimal("60.000000")
    assert gain.return_pct == Decimal("0.10000000")
    with pytest.raises(PortfolioPlanningError, match="review a new plan"):
        confirm_monthly_contribution_plan(
            portfolio=portfolio,
            expected_plan_hash=plan.plan_hash,
            idempotency_key=uuid4(),
        )
    upsert_holding(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("2"),
        average_cost=Decimal("100"),
    )
    rebased = calculate_contribution_performance(portfolio)
    assert portfolio.performance_baselines.count() == 1
    assert rebased.tracked_contributions == Decimal(0)
    assert rebased.investment_profit_loss == Decimal(0)
    assert rebased.return_pct == Decimal(0)
    assert rebased.withheld_reason == ""


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL row-lock regression",
)
@pytest.mark.django_db(transaction=True)
def test_concurrent_confirmations_for_different_portfolios_do_not_deadlock(owner) -> None:
    spy, stocks = _planner_market()
    spy_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Concurrent SPY holder",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    stock_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Concurrent stock holder",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    PortfolioHolding.objects.create(
        portfolio=spy_portfolio,
        listing=spy,
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    PortfolioHolding.objects.create(
        portfolio=stock_portfolio,
        listing=stocks[0],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    plans = {
        spy_portfolio.pk: preview_monthly_contribution_plan(spy_portfolio),
        stock_portfolio.pk: preview_monthly_contribution_plan(stock_portfolio),
    }
    barrier = Barrier(2)

    def confirm(portfolio_id) -> tuple[object, bool]:
        connections.close_all()
        try:
            portfolio = Portfolio.objects.get(pk=portfolio_id)
            barrier.wait(timeout=10)
            execution, created = confirm_monthly_contribution_plan(
                portfolio=portfolio,
                expected_plan_hash=plans[portfolio_id].plan_hash,
                idempotency_key=uuid4(),
            )
            return execution.pk, created
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(confirm, plans))

    assert len({execution_id for execution_id, _created in results}) == 2
    assert all(created for _execution_id, created in results)


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL row-lock regression",
)
@pytest.mark.django_db(transaction=True)
def test_concurrent_deposit_and_confirmation_do_not_deadlock(owner) -> None:
    spy, stocks = _planner_market()
    deposit_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Concurrent deposit",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    confirmation_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Concurrent confirmation",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    for portfolio in (deposit_portfolio, confirmation_portfolio):
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            listing=spy,
            quantity=Decimal("1"),
            average_cost=Decimal("100"),
        )
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            listing=stocks[0],
            quantity=Decimal("1"),
            average_cost=Decimal("100"),
        )
    plan = preview_monthly_contribution_plan(confirmation_portfolio)
    barrier = Barrier(2)

    def deposit() -> bool:
        connections.close_all()
        try:
            portfolio = Portfolio.objects.get(pk=deposit_portfolio.pk)
            barrier.wait(timeout=10)
            _deposit, created = record_external_deposit(
                portfolio=portfolio,
                amount=Decimal("100"),
                idempotency_key=uuid4(),
            )
            return created
        finally:
            connections.close_all()

    def confirm() -> bool:
        connections.close_all()
        try:
            portfolio = Portfolio.objects.get(pk=confirmation_portfolio.pk)
            barrier.wait(timeout=10)
            _execution, created = confirm_monthly_contribution_plan(
                portfolio=portfolio,
                expected_plan_hash=plan.plan_hash,
                idempotency_key=uuid4(),
            )
            return created
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        deposit_future = executor.submit(deposit)
        confirmation_future = executor.submit(confirm)

    assert deposit_future.result(timeout=20) is True
    assert confirmation_future.result(timeout=20) is True


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL row-lock regression",
)
@pytest.mark.django_db(transaction=True)
def test_deposit_does_not_block_on_snapshot_foreign_key_locks(owner) -> None:
    spy, stocks = _planner_market()
    snapshot_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Snapshot lock holder for deposit",
        base_currency="USD",
    )
    deposit_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Deposit during snapshot",
        base_currency="USD",
    )
    for portfolio in (snapshot_portfolio, deposit_portfolio):
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            listing=spy,
            quantity=Decimal("1"),
            average_cost=Decimal("100"),
        )
        PortfolioHolding.objects.create(
            portfolio=portfolio,
            listing=stocks[0],
            quantity=Decimal("1"),
            average_cost=Decimal("100"),
        )
    snapshot_recorded = Event()
    release_snapshot = Event()

    def deposit() -> tuple[PortfolioDeposit, bool]:
        connections.close_all()
        try:
            portfolio = Portfolio.objects.get(pk=deposit_portfolio.pk)
            return record_external_deposit(
                portfolio=portfolio,
                amount=Decimal("100"),
                idempotency_key=uuid4(),
            )
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot_future = executor.submit(
            _hold_snapshot_transaction,
            portfolio_id=snapshot_portfolio.pk,
            snapshot_recorded=snapshot_recorded,
            release_snapshot=release_snapshot,
        )
        assert snapshot_recorded.wait(timeout=10)
        deposit_future = executor.submit(deposit)
        try:
            _deposit, created = deposit_future.result(timeout=10)
        finally:
            release_snapshot.set()
        snapshot_future.result(timeout=10)

    assert created is True


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL row-lock regression",
)
@pytest.mark.django_db(transaction=True)
def test_confirmation_does_not_block_on_snapshot_foreign_key_locks(owner) -> None:
    spy, stocks = _planner_market()
    snapshot_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Snapshot lock holder for confirmation",
        base_currency="USD",
    )
    confirmation_portfolio = Portfolio.objects.create(
        owner=owner,
        name="Confirmation during snapshot",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    for listing in (spy, stocks[0]):
        PortfolioHolding.objects.create(
            portfolio=snapshot_portfolio,
            listing=listing,
            quantity=Decimal("1"),
            average_cost=Decimal("100"),
        )
    plan = preview_monthly_contribution_plan(confirmation_portfolio)
    snapshot_recorded = Event()
    release_snapshot = Event()

    def confirm() -> tuple[PortfolioPlanExecution, bool]:
        connections.close_all()
        try:
            portfolio = Portfolio.objects.get(pk=confirmation_portfolio.pk)
            return confirm_monthly_contribution_plan(
                portfolio=portfolio,
                expected_plan_hash=plan.plan_hash,
                idempotency_key=uuid4(),
            )
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot_future = executor.submit(
            _hold_snapshot_transaction,
            portfolio_id=snapshot_portfolio.pk,
            snapshot_recorded=snapshot_recorded,
            release_snapshot=release_snapshot,
        )
        assert snapshot_recorded.wait(timeout=10)
        confirmation_future = executor.submit(confirm)
        try:
            _execution, created = confirmation_future.result(timeout=10)
        finally:
            release_snapshot.set()
        snapshot_future.result(timeout=10)

    assert created is True


def test_whole_share_mode_carries_unspent_cash(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy, stocks = _planner_market(
        spy_price=Decimal("110"),
        stock_prices=(Decimal("130"),),
    )
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Whole shares",
        base_currency="USD",
        cash_balance=Decimal("600"),
        allow_fractional_shares=False,
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )

    plan = preview_monthly_contribution_plan(portfolio)

    assert [
        (purchase.listing, purchase.quantity, purchase.amount) for purchase in plan.purchases
    ] == [
        (spy, Decimal("3"), Decimal("330.000000")),
        (stocks[0], Decimal("1"), Decimal("130.000000")),
    ]
    assert plan.residual_cash == Decimal("140.000000")


def test_missing_satellite_never_forces_a_purchase(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _spy()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Core only",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )

    plan = preview_monthly_contribution_plan(portfolio)

    assert len(plan.purchases) == 1
    assert plan.purchases[0].listing == spy
    assert plan.purchases[0].amount == Decimal("420.000000")
    assert plan.residual_cash == Decimal("180.000000")
    assert "No provider-backed stock analysis" in plan.satellite_reason


def test_medium_horizon_opportunity_cannot_qualify_short_satellite(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    analysis = StockAnalysis.objects.get(listing=stocks[0])
    data_quality = dict(analysis.data_quality)
    data_quality.update(
        {
            "analysis_mode": "full",
            "fundamentals_used": True,
        }
    )
    StockAnalysis.objects.filter(pk=analysis.pk).update(
        data_quality=data_quality,
        short_scenario={"bear": -0.20, "base": -0.10, "bull": 0.01},
        medium_scenario={"bear": 0.01, "base": 0.20, "bull": 0.40},
    )
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Short evidence only",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )

    plan = preview_monthly_contribution_plan(portfolio)

    assert [purchase.role for purchase in plan.purchases] == [PortfolioPurchase.Role.CORE]
    assert plan.residual_cash == Decimal("180.000000")
    assert "No currently qualified short-horizon" in plan.satellite_reason


def test_existing_spy_overweight_routes_cash_only_to_one_satellite(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="SPY overweight",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=spy,
        quantity=Decimal("15"),
        average_cost=Decimal("90"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )

    plan = preview_monthly_contribution_plan(portfolio)

    assert plan.spy_budget == Decimal("0.000000")
    assert plan.satellite_budget == Decimal("600.000000")
    assert len(plan.purchases) == 1
    assert plan.purchases[0].listing == stocks[0]
    assert plan.purchases[0].amount == Decimal("600.000000")


def test_under_10_stock_is_skipped_for_next_qualified_satellite(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market(
        stock_prices=(Decimal("9.99"), Decimal("100")),
    )
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Guarded satellite",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )

    plan = preview_monthly_contribution_plan(portfolio)

    satellites = [
        purchase for purchase in plan.purchases if purchase.role == PortfolioPurchase.Role.SATELLITE
    ]
    assert len(satellites) == 1
    assert satellites[0].listing == stocks[1]
    assert satellites[0].listing != stocks[0]


def test_stale_spy_price_withholds_plan(owner, monkeypatch: pytest.MonkeyPatch) -> None:
    _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Stale plan",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 20),
    )

    plan = preview_monthly_contribution_plan(portfolio)

    assert plan.purchases == ()
    assert any("SPY market price is stale" in issue for issue in plan.issues)
    assert plan.executable is False


def test_stale_pre_deposit_valuation_never_becomes_a_performance_boundary(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Stale boundary",
        base_currency="USD",
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 20),
    )

    deposit, created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    LatestMarketData.objects.filter(listing=stocks[0]).update(
        session_date=date(2026, 9, 19),
        observed_at=datetime(2026, 9, 20, 1, tzinfo=UTC),
        source_asset=_market_asset("PLAN1", "b"),
        close=Decimal("110"),
    )
    portfolio.refresh_from_db()
    performance = calculate_contribution_performance(portfolio)

    assert created is True
    assert deposit.boundary_snapshot is None
    assert "stale" in deposit.boundary_issue
    assert performance.investment_profit_loss is None
    assert performance.return_pct is None
    assert "lacks a complete pre-flow valuation boundary" in performance.withheld_reason


def test_mixed_session_terminal_valuation_withholds_contribution_return(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Mixed terminal valuation",
        base_currency="USD",
    )
    PortfolioHolding.objects.bulk_create(
        [
            PortfolioHolding(
                portfolio=portfolio,
                listing=spy,
                quantity=Decimal("1"),
                average_cost=Decimal("100"),
            ),
            PortfolioHolding(
                portfolio=portfolio,
                listing=stocks[0],
                quantity=Decimal("1"),
                average_cost=Decimal("100"),
            ),
        ]
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    deposit, created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    LatestMarketData.objects.filter(listing=stocks[0]).update(
        session_date=date(2026, 9, 3),
    )
    portfolio.refresh_from_db()

    performance = calculate_contribution_performance(portfolio)

    assert created is True
    assert deposit.boundary_snapshot is not None
    assert performance.investment_profit_loss is None
    assert performance.return_pct is None
    assert "do not share one market session" in performance.withheld_reason


def test_split_warning_prevents_a_performance_boundary(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Split warning boundary",
        base_currency="USD",
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    record_portfolio_snapshot(portfolio)
    LatestMarketData.objects.filter(listing=stocks[0]).update(
        close=Decimal("40"),
        source_asset=_market_asset("PLAN1", "b"),
    )

    deposit, created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )

    assert created is True
    assert deposit.boundary_snapshot is None
    assert "corporate action" in deposit.boundary_issue


def test_split_warning_remains_unresolved_across_repeated_snapshots(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Persistent split warning",
        base_currency="USD",
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("1"),
        average_cost=Decimal("100"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    deposit, created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    LatestMarketData.objects.filter(listing=stocks[0]).update(
        close=Decimal("40"),
        source_asset=_market_asset("PLAN1", "b"),
    )
    first_warning, first_created = record_portfolio_snapshot(portfolio)
    LatestMarketData.objects.filter(listing=stocks[0]).update(
        source_asset=_market_asset("PLAN1", "c"),
    )
    second_warning, second_created = record_portfolio_snapshot(portfolio)
    portfolio.refresh_from_db()

    performance = calculate_contribution_performance(portfolio)
    plan = preview_monthly_contribution_plan(portfolio)

    assert created is True
    assert deposit.boundary_snapshot is not None
    assert first_created is True
    assert second_created is True
    assert first_warning.corporate_action_warnings == 1
    assert second_warning.corporate_action_warnings == 1
    assert performance.return_pct is None
    assert "corporate action" in performance.withheld_reason
    assert plan.purchases == ()
    assert any("split events" in issue for issue in plan.issues)

    upsert_holding(
        portfolio=portfolio,
        listing=stocks[0],
        quantity=Decimal("2.5"),
        average_cost=Decimal("40"),
    )
    portfolio.refresh_from_db()
    resumed = calculate_contribution_performance(portfolio)

    assert portfolio.performance_baselines.count() == 1
    assert resumed.investment_profit_loss == Decimal(0)
    assert resumed.return_pct == Decimal(0)
    assert resumed.withheld_reason == ""


def test_missing_deposit_boundary_withholds_percentage_performance(owner) -> None:
    company = Company.objects.create(name="Unpriced Corp", country="US")
    security = Security.objects.create(company=company, name="Unpriced Common")
    listing = Listing.objects.create(
        security=security,
        ticker="NOPRICE",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Missing boundary",
        base_currency="USD",
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=listing,
        quantity=Decimal("1"),
        average_cost=Decimal("10"),
    )

    deposit, created = record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    performance = calculate_contribution_performance(portfolio)

    assert created is True
    assert deposit.boundary_snapshot is None
    assert deposit.boundary_issue
    assert performance.investment_profit_loss is None
    assert performance.return_pct is None
    assert "lacks a complete pre-flow valuation boundary" in performance.withheld_reason


def test_portfolio_web_flow_records_deposit_and_confirmed_plan(
    client,
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _planner_market()
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    client.force_login(owner)

    created = client.post(
        reverse("portfolios"),
        {
            "action": "manual",
            "name": "Monthly investing",
            "description": "",
            "base_currency": "USD",
            "cash_balance": "600",
            "monthly_contribution": "600",
            "allow_fractional_shares": "on",
        },
    )
    portfolio = Portfolio.objects.get(owner=owner, name="Monthly investing")
    assert created.status_code == 302
    assert portfolio.deposits.count() == 1

    detail = client.get(reverse("portfolio-detail", args=[portfolio.pk]))
    assert detail.status_code == 200
    assert b"Monthly allocation plan" in detail.content
    assert b"420.00" in detail.content
    plan = preview_monthly_contribution_plan(portfolio)

    executed = client.post(
        reverse("portfolio-detail", args=[portfolio.pk]),
        {
            "action": "execute_plan",
            "plan_hash": plan.plan_hash,
            "idempotency_key": str(uuid4()),
        },
    )

    assert executed.status_code == 302
    assert PortfolioPlanExecution.objects.filter(portfolio=portfolio).count() == 1
    assert PortfolioPurchase.objects.filter(execution__portfolio=portfolio).count() == 2
    completed = client.get(reverse("portfolio-detail", args=[portfolio.pk]))
    assert b"Planner purchases" in completed.content
    assert b"Contribution-adjusted return" in completed.content


def test_rejected_plan_confirmation_renders_a_fresh_retry_hash(
    client,
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, _stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Fresh retry",
        base_currency="USD",
        cash_balance=Decimal("600"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    client.force_login(owner)
    stale_plan = preview_monthly_contribution_plan(portfolio)
    actual_confirm = confirm_monthly_contribution_plan
    deposit_injected = False

    def confirm_after_concurrent_deposit(**kwargs):
        nonlocal deposit_injected
        if not deposit_injected:
            deposit_injected = True
            record_external_deposit(
                portfolio=Portfolio.objects.get(pk=portfolio.pk),
                amount=Decimal("600"),
                idempotency_key=uuid4(),
            )
        return actual_confirm(**kwargs)

    monkeypatch.setattr(
        "stanstock.web.views.confirm_monthly_contribution_plan",
        confirm_after_concurrent_deposit,
    )

    rejected = client.post(
        reverse("portfolio-detail", args=[portfolio.pk]),
        {
            "action": "execute_plan",
            "plan_hash": stale_plan.plan_hash,
            "idempotency_key": str(uuid4()),
        },
    )
    portfolio.refresh_from_db()
    current_plan = preview_monthly_contribution_plan(portfolio)
    rendered_form = rejected.context["plan_confirmation_form"]

    assert rejected.status_code == 400
    assert b"review a new plan" in rejected.content
    assert rendered_form["plan_hash"].value() == current_plan.plan_hash
    assert rendered_form["plan_hash"].value() != stale_plan.plan_hash

    retried = client.post(
        reverse("portfolio-detail", args=[portfolio.pk]),
        {
            "action": "execute_plan",
            "plan_hash": rendered_form["plan_hash"].value(),
            "idempotency_key": rendered_form["idempotency_key"].value(),
        },
    )

    assert retried.status_code == 302
    assert PortfolioPlanExecution.objects.filter(portfolio=portfolio).count() == 1


def test_settings_update_never_overwrites_cash_recorded_during_validation(
    client,
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Concurrent settings",
        base_currency="USD",
    )
    client.force_login(owner)
    original_is_valid = PortfolioForm.is_valid
    deposit_recorded = False

    def is_valid_with_deposit(form: PortfolioForm) -> bool:
        nonlocal deposit_recorded
        valid = original_is_valid(form)
        if valid and not deposit_recorded:
            deposit_recorded = True
            record_external_deposit(
                portfolio=form.instance,
                amount=Decimal("600"),
                idempotency_key=uuid4(),
            )
        return valid

    monkeypatch.setattr(PortfolioForm, "is_valid", is_valid_with_deposit)

    response = client.post(
        reverse("portfolio-detail", args=[portfolio.pk]),
        {
            "action": "update",
            "name": "Concurrent settings updated",
            "description": "Keep funded cash intact",
            "base_currency": "USD",
            "monthly_contribution": "700",
            "allow_fractional_shares": "on",
        },
    )
    portfolio.refresh_from_db()

    assert response.status_code == 302
    assert portfolio.name == "Concurrent settings updated"
    assert portfolio.monthly_contribution == Decimal("700")
    assert portfolio.cash_balance == Decimal("600")
    assert portfolio.deposits.count() == 1
    assert calculate_contribution_performance(portfolio).return_pct == Decimal(0)


def test_admin_updates_cannot_restore_deposited_or_spent_cash(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Admin cash safety",
        base_currency="USD",
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    portfolio_admin = PortfolioAdmin(Portfolio, admin.site)
    request = RequestFactory().post("/admin/portfolio/portfolio/")
    request.user = owner

    stale_before_deposit = Portfolio.objects.get(pk=portfolio.pk)
    record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    stale_before_deposit.description = "Saved after deposit"
    portfolio_admin.save_model(
        request,
        stale_before_deposit,
        SimpleNamespace(changed_data=["description"]),
        change=True,
    )
    portfolio.refresh_from_db()
    assert portfolio.cash_balance == Decimal("600")

    plan = preview_monthly_contribution_plan(portfolio)
    stale_before_confirmation = Portfolio.objects.get(pk=portfolio.pk)
    confirm_monthly_contribution_plan(
        portfolio=portfolio,
        expected_plan_hash=plan.plan_hash,
        idempotency_key=uuid4(),
    )
    stale_before_confirmation.description = "Saved after confirmation"
    portfolio_admin.save_model(
        request,
        stale_before_confirmation,
        SimpleNamespace(changed_data=["description"]),
        change=True,
    )
    portfolio.refresh_from_db()

    assert portfolio.description == "Saved after confirmation"
    assert portfolio.cash_balance == Decimal(0)
    assert calculate_contribution_performance(portfolio).return_pct == Decimal(0)


def test_holding_admin_notes_cannot_erase_concurrent_planner_purchase(
    owner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_listing, stocks = _planner_market()
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Admin holding safety",
        base_currency="USD",
    )
    monkeypatch.setattr(
        "stanstock.portfolio.planner.timezone.localdate",
        lambda: date(2026, 9, 7),
    )
    record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    portfolio.refresh_from_db()
    first_plan = preview_monthly_contribution_plan(portfolio)
    confirm_monthly_contribution_plan(
        portfolio=portfolio,
        expected_plan_hash=first_plan.plan_hash,
        idempotency_key=uuid4(),
    )
    record_external_deposit(
        portfolio=portfolio,
        amount=Decimal("600"),
        idempotency_key=uuid4(),
    )
    portfolio.refresh_from_db()
    second_plan = preview_monthly_contribution_plan(portfolio)
    stale_holding = PortfolioHolding.objects.get(
        portfolio=portfolio,
        listing=stocks[0],
    )
    confirm_monthly_contribution_plan(
        portfolio=portfolio,
        expected_plan_hash=second_plan.plan_hash,
        idempotency_key=uuid4(),
    )
    stale_holding.notes = "Reviewed in admin"
    holding_admin = PortfolioHoldingAdmin(PortfolioHolding, admin.site)
    request = RequestFactory().post("/admin/portfolio/portfolioholding/")
    request.user = owner
    holding_admin.save_model(
        request,
        stale_holding,
        SimpleNamespace(changed_data=["notes"]),
        change=True,
    )
    portfolio.refresh_from_db()
    holding = PortfolioHolding.objects.get(
        portfolio=portfolio,
        listing=stocks[0],
    )

    assert holding.notes == "Reviewed in admin"
    assert holding.quantity == Decimal("3.60000000")
    assert holding.average_cost == Decimal("100")
    assert portfolio.cash_balance == Decimal(0)
    assert PortfolioPurchase.objects.filter(execution__portfolio=portfolio).count() == 4
    assert calculate_contribution_performance(portfolio).return_pct == Decimal(0)
