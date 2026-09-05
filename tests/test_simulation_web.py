from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from django.contrib.auth import get_user_model
from django.test import Client
from django.urls import reverse
from django.utils import timezone

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import (
    Company,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.research.models import (
    AnalysisRun,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.simulation.builders import build_price_panel, build_signals_for_backtest
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from stanstock.simulation.types import SimulationWorkflowError


@pytest.fixture
def auth_client() -> Client:
    client = Client()
    user_model = get_user_model()
    user = user_model.objects.create_user(username="testuser", password="testpassword123")
    client.force_login(user)
    return client


@pytest.fixture
def web_setup_environment(
    tmp_path: Path,
) -> tuple[UniverseSnapshot, Listing, AssetStore]:
    store = AssetStore(root=tmp_path / "web_data")
    company = Company.objects.create(name="Microsoft Corp", country="US")
    sec = Security.objects.create(company=company, name="MSFT Common")
    listing = Listing.objects.create(
        security=sec,
        ticker="MSFT",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(slug="us-tech", name="US Tech", config_version="1.0")
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=date(2026, 1, 1),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="snapshot-hash-123",
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing, eligible=True)

    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)
    price_frame = pl.DataFrame(
        {
            "date": [d0, d1],
            "close": [300.0, 310.0],
            "open": [299.0, 309.0],
        }
    )
    stored = store.write_frame("prices/msft.parquet", price_frame)
    now = timezone.now()
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject="MSFT",
        stored=stored,
        retrieved_at=now,
        available_at=now,
    )

    return snapshot, listing, store


@pytest.mark.django_db
def test_simulations_page_get(auth_client: Client) -> None:
    """GET /simulations renders the page and simulation form."""
    resp = auth_client.get(reverse("simulations"))
    assert resp.status_code == 200
    assert "form" in resp.context
    assert b"Run simulation" in resp.content


@pytest.mark.django_db
def test_simulations_page_post_unauthenticated() -> None:
    """Unauthenticated POST /simulations redirects to login."""
    client = Client()
    resp = client.post(reverse("simulations"), {})
    assert resp.status_code == 302
    location = resp.headers.get("Location", "")
    assert "/login" in location or "accounts/login" in location or "next=" in location


@pytest.mark.django_db
def test_simulations_page_post_portfolio_valid(
    auth_client: Client,
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Valid portfolio POST /simulations creates run, persists UUIDs, redirects to detail."""
    snapshot, listing, store = web_setup_environment

    post_data = {
        "name": "Web Portfolio Test",
        "mode": "portfolio",
        "snapshot": str(snapshot.id),
        "start_date": "2026-01-01",
        "end_date": "2026-01-02",
        "starting_capital": "75000.00",
        "transaction_cost_bps": "10.00",
        "slippage_bps": "5.00",
        "selected_listings": str(listing.id),
    }

    with (
        patch("stanstock.simulation.builders.AssetStore", return_value=store),
        patch("stanstock.simulation.service.AssetStore", return_value=store),
    ):
        resp = auth_client.post(reverse("simulations"), post_data)

    assert resp.status_code == 302
    run = SimulationRun.objects.filter(definition__name="Web Portfolio Test").first()
    assert run is not None
    assert run.status == SimulationRun.Status.COMPLETE
    assert resp.headers.get("Location") == reverse("simulation-detail", kwargs={"run_id": run.id})

    # Verify UUID identity
    for h in run.holdings.all():
        assert h.listing_id == listing.id
    for t in run.trades.all():
        assert t.listing_id == listing.id


@pytest.mark.django_db
def test_simulations_page_post_backtest_no_signals_rejected(
    auth_client: Client,
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Backtest POST /simulations without signals returns 400 and shows form error."""
    snapshot, _, store = web_setup_environment

    initial_defs = SimulationDefinition.objects.count()
    initial_runs = SimulationRun.objects.count()

    post_data = {
        "name": "Web Backtest Fail",
        "mode": "backtest",
        "snapshot": str(snapshot.id),
        "start_date": "2026-01-01",
        "end_date": "2026-01-02",
        "starting_capital": "50000.00",
        "transaction_cost_bps": "10.00",
        "slippage_bps": "5.00",
        "top_n": "5",
    }

    with (
        patch("stanstock.simulation.builders.AssetStore", return_value=store),
        patch("stanstock.simulation.service.AssetStore", return_value=store),
    ):
        resp = auth_client.post(reverse("simulations"), post_data)

    assert resp.status_code == 400
    assert b"No persisted signals found" in resp.content

    # No definition or run created
    assert SimulationDefinition.objects.count() == initial_defs
    assert SimulationRun.objects.count() == initial_runs


@pytest.mark.django_db
def test_simulations_page_post_ineligible_listing_rejected(
    auth_client: Client,
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """ModelMultipleChoiceField validates listing membership against selected snapshot."""
    snapshot, _, store = web_setup_environment

    # Create another listing that is NOT in snapshot membership
    other_comp = Company.objects.create(name="Outside Inc", country="US")
    other_sec = Security.objects.create(company=other_comp, name="Outside Common")
    other_listing = Listing.objects.create(
        security=other_sec,
        ticker="OUTSIDE",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )

    post_data = {
        "name": "Invalid Member Run",
        "mode": "portfolio",
        "snapshot": str(snapshot.id),
        "start_date": "2026-01-01",
        "end_date": "2026-01-02",
        "starting_capital": "50000.00",
        "transaction_cost_bps": "10.00",
        "slippage_bps": "5.00",
        "selected_listings": [str(other_listing.id)],
    }

    with (
        patch("stanstock.simulation.builders.AssetStore", return_value=store),
        patch("stanstock.simulation.service.AssetStore", return_value=store),
    ):
        resp = auth_client.post(reverse("simulations"), post_data)

    assert resp.status_code == 400
    assert b"Selected listings must be eligible members" in resp.content


@pytest.mark.django_db
def test_build_signals_requires_completed_runs_and_rejects_duplicates(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Signals require run.status='complete' and reject duplicate (target_date, listing)."""
    snapshot, listing, _ = web_setup_environment
    d0 = date(2026, 1, 1)
    signal_time = datetime(2026, 1, 1, 21, tzinfo=timezone.get_current_timezone())

    # 1. Incomplete run is ignored
    incomplete_run = AnalysisRun.objects.create(
        generated_at=signal_time,
        data_cutoff=signal_time,
        target_date=d0,
        universe_snapshot=snapshot,
        config_version="v1",
        config_hash="h1",
        code_revision="r1",
        status="running",
    )
    StockAnalysis.objects.create(
        run=incomplete_run,
        listing=listing,
        current_price=Decimal("300.0"),
        overall_score=Decimal("80.0"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("20.0"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("90.0"),
    )

    with pytest.raises(SimulationWorkflowError, match="No persisted signals found"):
        build_signals_for_backtest(
            snapshot=snapshot,
            start_date=d0,
            end_date=d0,
        )

    # 2. Complete run succeeds
    complete_run = AnalysisRun.objects.create(
        generated_at=signal_time,
        data_cutoff=signal_time,
        target_date=d0,
        universe_snapshot=snapshot,
        config_version="v1",
        config_hash="h2",
        code_revision="r2",
        status="complete",
    )
    StockAnalysis.objects.create(
        run=complete_run,
        listing=listing,
        current_price=Decimal("300.0"),
        overall_score=Decimal("85.0"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("20.0"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("90.0"),
    )

    signals = build_signals_for_backtest(
        snapshot=snapshot,
        start_date=d0,
        end_date=d0,
    )
    assert signals.height == 1

    # 3. Duplicate completed run with same target_date and listing is rejected
    dup_run = AnalysisRun.objects.create(
        generated_at=signal_time,
        data_cutoff=signal_time,
        target_date=d0,
        universe_snapshot=snapshot,
        config_version="v1",
        config_hash="h3",
        code_revision="r3",
        status="complete",
    )
    StockAnalysis.objects.create(
        run=dup_run,
        listing=listing,
        current_price=Decimal("300.0"),
        overall_score=Decimal("75.0"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("20.0"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("90.0"),
    )

    with pytest.raises(SimulationWorkflowError, match="Ambiguous duplicate signals detected"):
        build_signals_for_backtest(
            snapshot=snapshot,
            start_date=d0,
            end_date=d0,
        )


@pytest.mark.django_db
def test_observed_backtest_rejects_late_generated_signals(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    snapshot, listing, _store = web_setup_environment
    target = date(2026, 1, 1)
    cutoff = datetime(2026, 1, 1, 21, tzinfo=timezone.get_current_timezone())
    run = AnalysisRun.objects.create(
        generated_at=datetime(2026, 1, 3, 12, tzinfo=timezone.get_current_timezone()),
        data_cutoff=cutoff,
        target_date=target,
        universe_snapshot=snapshot,
        config_version="v1",
        config_hash="late",
        code_revision="test",
        status="complete",
    )
    StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("300"),
        overall_score=Decimal("80"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("20"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("80"),
    )

    with pytest.raises(
        SimulationWorkflowError,
        match="Observed backtests require signals generated on their target date",
    ):
        build_signals_for_backtest(
            snapshot=snapshot,
            start_date=target,
            end_date=target,
        )


@pytest.mark.django_db
def test_research_backtest_keeps_late_generation_provenance(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    snapshot, listing, _store = web_setup_environment
    snapshot.grade = UniverseSnapshot.Grade.RESEARCH
    snapshot.save(update_fields=["grade"])
    target = date(2026, 1, 1)
    cutoff = datetime(2026, 1, 1, 21, tzinfo=timezone.get_current_timezone())
    generated_at = datetime(2026, 1, 3, 12, tzinfo=timezone.get_current_timezone())
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=cutoff,
        target_date=target,
        universe_snapshot=snapshot,
        config_version="v1",
        config_hash="research",
        code_revision="test",
        status="complete",
    )
    StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("300"),
        overall_score=Decimal("80"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("20"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("80"),
    )

    signals = build_signals_for_backtest(
        snapshot=snapshot,
        start_date=target,
        end_date=target,
    )

    assert signals["date"].to_list() == [target]
    assert signals["generated_at"].to_list()[0].date() == generated_at.date()
    assert signals["data_cutoff"].to_list()[0].date() == target


@pytest.mark.django_db
def test_price_panel_rejects_mixed_currency_and_supports_explicit_filter(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    snapshot, usd_listing, store = web_setup_environment
    company = Company.objects.create(name="Euro Co", country="DE")
    security = Security.objects.create(company=company, name="Euro Co Common")
    eur_listing = Listing.objects.create(
        security=security,
        ticker="EURO",
        exchange_mic="XETR",
        currency="EUR",
        region=Region.EUROPE,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=eur_listing, eligible=True)
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 2)],
            "open": [99.0, 100.0],
            "close": [100.0, 101.0],
        }
    )
    stored = store.write_frame("prices/euro.parquet", frame)
    now = timezone.now()
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject="EURO",
        stored=stored,
        retrieved_at=now,
        available_at=now,
    )

    with pytest.raises(SimulationWorkflowError, match="must use one native currency"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
            provider="synthetic_demo",
            asset_store=store,
        )

    panel, _benchmark = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        provider="synthetic_demo",
        base_currency="USD",
        asset_store=store,
    )
    assert set(panel["listing_id"].to_list()) == {str(usd_listing.id)}
    assert set(panel["currency"].to_list()) == {"USD"}


@pytest.mark.django_db
def test_build_price_panel_fails_when_listing_lacks_usable_rows_or_benchmark_empty(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Fail explicitly when any requested listing has no rows in range or benchmark is empty."""
    snapshot, listing, store = web_setup_environment

    # 1. Range outside of available prices (available is 2026-01-01 to 2026-01-02)
    with pytest.raises(SimulationWorkflowError, match="has no usable price rows"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 1, 5),
            end_date=date(2026, 1, 6),
            listing_ids=[listing.id],
            provider="synthetic_demo",
            asset_store=store,
        )

    # 2. Benchmark subject requested with empty rows in range
    now = timezone.now()
    empty_bench_frame = pl.DataFrame(
        {
            "date": [date(2025, 1, 1)],
            "close": [100.0],
        }
    )
    stored_bench = store.write_frame("prices/empty_bench.parquet", empty_bench_frame)
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject="BENCH_EMPTY",
        stored=stored_bench,
        retrieved_at=now,
        available_at=now,
    )

    with pytest.raises(
        SimulationWorkflowError,
        match="Benchmark subject 'BENCH_EMPTY' has no usable price rows",
    ):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
            listing_ids=[listing.id],
            provider="synthetic_demo",
            benchmark_subject="BENCH_EMPTY",
            asset_store=store,
        )


@pytest.mark.django_db
def test_build_price_panel_requires_common_inception_for_selected_listings(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    snapshot, listing, store = web_setup_environment
    company = Company.objects.create(name="Late Price Inc", country="US")
    security = Security.objects.create(company=company, name="Late Price Common")
    late_listing = Listing.objects.create(
        security=security,
        ticker="LATE",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=late_listing, eligible=True)
    late_frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 2)],
            "open": [50.0],
            "close": [51.0],
        }
    )
    stored = store.write_frame("prices/late.parquet", late_frame)
    now = timezone.now()
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject="LATE",
        stored=stored,
        retrieved_at=now,
        available_at=now,
    )

    with pytest.raises(SimulationWorkflowError, match="lack a usable inception close"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
            listing_ids=[listing.id, late_listing.id],
            provider="synthetic_demo",
            asset_store=store,
        )
