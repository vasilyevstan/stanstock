from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
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
from stanstock.simulation.models import SimulationDefinition, SimulationRun


@pytest.fixture
def test_setup_environment(
    tmp_path: Path,
) -> tuple[UniverseSnapshot, Listing, AssetStore]:
    store = AssetStore(root=tmp_path / "data")
    company = Company.objects.create(name="Apple Inc", country="US")
    sec = Security.objects.create(company=company, name="AAPL Common")
    listing = Listing.objects.create(
        security=sec,
        ticker="AAPL",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(slug="us-large", name="US Large Cap", config_version="1.0")
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=date(2026, 1, 1),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="test-hash",
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing, eligible=True)

    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)
    price_frame = pl.DataFrame(
        {
            "date": [d0, d1],
            "close": [150.0, 155.0],
            "open": [149.0, 154.0],
        }
    )
    stored = store.write_frame("prices/aapl.parquet", price_frame)
    now = timezone.now()
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject="AAPL",
        stored=stored,
        retrieved_at=now,
        available_at=now,
    )

    return snapshot, listing, store


@pytest.mark.django_db
def test_simulate_command_portfolio_run(
    test_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Valid portfolio run via simulate command executes and persists run with UUIDs."""
    snapshot, listing, store = test_setup_environment
    out = StringIO()

    with (
        patch("stanstock.simulation.builders.AssetStore", return_value=store),
        patch("stanstock.simulation.service.AssetStore", return_value=store),
    ):
        call_command(
            "simulate",
            "--name=Portfolio Run",
            "--mode=portfolio",
            f"--snapshot={snapshot.id}",
            "--start-date=2026-01-01",
            "--end-date=2026-01-02",
            f"--listings={listing.id}",
            "--capital=50000",
            stdout=out,
        )

    output = out.getvalue()
    assert "Successfully completed simulation run" in output

    runs = SimulationRun.objects.filter(definition__name="Portfolio Run")
    assert runs.count() == 1
    run = runs.first()
    assert run is not None
    assert run.status == SimulationRun.Status.COMPLETE
    assert run.holdings.count() == 2
    assert run.trades.count() == 1

    # Verify UUID identity on holdings and trades
    for h in run.holdings.all():
        assert h.listing_id == listing.id
    for t in run.trades.all():
        assert t.listing_id == listing.id


@pytest.mark.django_db
def test_simulate_command_backtest_no_signals_rejected(
    test_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Backtest mode explicitly rejects when no persisted signals exist for snapshot."""
    snapshot, _, store = test_setup_environment

    initial_defs = SimulationDefinition.objects.count()
    initial_runs = SimulationRun.objects.count()

    with (
        patch("stanstock.simulation.builders.AssetStore", return_value=store),
        patch("stanstock.simulation.service.AssetStore", return_value=store),
    ):
        with pytest.raises(CommandError, match="No persisted signals found"):
            call_command(
                "simulate",
                "--name=Backtest Without Signals",
                "--mode=backtest",
                f"--snapshot={snapshot.id}",
                "--start-date=2026-01-01",
                "--end-date=2026-01-02",
                "--top-n=5",
            )

    # Verify no orphan SimulationDefinition or SimulationRun was created
    assert SimulationDefinition.objects.count() == initial_defs
    assert SimulationRun.objects.count() == initial_runs


@pytest.mark.django_db
def test_simulate_command_rejects_duplicate_listings(
    test_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Duplicate listing UUIDs in --listings must be rejected."""
    snapshot, listing, store = test_setup_environment
    with (
        patch("stanstock.simulation.builders.AssetStore", return_value=store),
        patch("stanstock.simulation.service.AssetStore", return_value=store),
    ):
        with pytest.raises(CommandError, match="Duplicate listing UUIDs detected"):
            call_command(
                "simulate",
                "--name=Dup Listings",
                "--mode=portfolio",
                f"--snapshot={snapshot.id}",
                "--start-date=2026-01-01",
                "--end-date=2026-01-02",
                f"--listings={listing.id},{listing.id}",
            )


@pytest.mark.django_db
def test_simulate_command_rejects_future_end_date(
    test_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """end_date after decision_time.date() must be rejected."""
    snapshot, listing, store = test_setup_environment
    with (
        patch("stanstock.simulation.builders.AssetStore", return_value=store),
        patch("stanstock.simulation.service.AssetStore", return_value=store),
    ):
        with pytest.raises(CommandError, match="cannot be after decision date"):
            call_command(
                "simulate",
                "--name=Future Run",
                "--mode=portfolio",
                f"--snapshot={snapshot.id}",
                "--start-date=2026-01-01",
                "--end-date=2099-01-01",
                f"--listings={listing.id}",
            )
