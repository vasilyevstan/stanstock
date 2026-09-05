from __future__ import annotations

import shutil
from collections.abc import Generator
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from django.conf import settings

from stanstock.data.assets import AssetStore
from stanstock.data.models import (
    Company,
    DataAsset,
    Listing,
    Region,
    Security,
    Universe,
    UniverseSnapshot,
)
from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.models import (
    SimulationDefinition,
    SimulationHolding,
    SimulationRun,
    SimulationTrade,
)
from stanstock.simulation.service import (
    execute_and_persist_simulation,
    load_simulation_run,
    persist_simulation_run,
)
from stanstock.simulation.types import (
    MissingPriceError,
    RebalanceFrequency,
    SimulationConfig,
    SimulationGrade,
    SimulationMode,
)


@pytest.fixture
def sim_asset_store() -> Generator[AssetStore]:
    scratch_dir = settings.DATA_DIR / "sim_scratch_tests"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    yield AssetStore(root=scratch_dir)
    shutil.rmtree(scratch_dir, ignore_errors=True)


@pytest.fixture
def test_setup_models() -> tuple[SimulationDefinition, UniverseSnapshot, Listing, Listing]:
    company1 = Company.objects.create(name="Apple Inc", country="US")
    sec1 = Security.objects.create(company=company1, name="AAPL Common")
    listing1 = Listing.objects.create(
        security=sec1,
        ticker="AAPL",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )

    company2 = Company.objects.create(name="Microsoft Corp", country="US")
    sec2 = Security.objects.create(company=company2, name="MSFT Common")
    listing2 = Listing.objects.create(
        security=sec2,
        ticker="MSFT",
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

    definition = SimulationDefinition.objects.create(
        name="Test Momentum Backtest",
        mode=SimulationDefinition.Mode.BACKTEST,
        config={
            "top_n": 2,
            "rebalance_frequency": "monthly",
            "starting_capital": 50000.0,
            "transaction_cost_bps": 10.0,
            "slippage_bps": 5.0,
        },
    )
    return definition, snapshot, listing1, listing2


@pytest.mark.django_db
def test_persist_and_load_simulation_run(
    sim_asset_store: AssetStore,
    test_setup_models: tuple[SimulationDefinition, UniverseSnapshot, Listing, Listing],
) -> None:
    definition, snapshot, listing1, listing2 = test_setup_models

    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)
    d2 = date(2026, 1, 3)
    prices = pl.DataFrame(
        {
            "date": [d0, d0, d1, d1, d2, d2],
            "listing_id": [
                str(listing1.id),
                str(listing2.id),
                str(listing1.id),
                str(listing2.id),
                str(listing1.id),
                str(listing2.id),
            ],
            "close": [150.0, 300.0, 155.0, 310.0, 160.0, 320.0],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        }
    )
    # Day 0 signal picks AAPL; Day 1 signal switches to MSFT (causing AAPL sell on Day 2)
    signals = pl.DataFrame(
        {
            "date": [d0, d0, d1, d1],
            "listing_id": [str(listing1.id), str(listing2.id), str(listing1.id), str(listing2.id)],
            "score": [10.0, 1.0, 1.0, 10.0],
        }
    )

    config = SimulationConfig(
        starting_capital=50000.0,
        top_n=1,
        rebalance_frequency=RebalanceFrequency.DAILY,
        mode=SimulationMode.BACKTEST,
        grade=SimulationGrade.OBSERVED,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals)

    run = persist_simulation_run(
        definition=definition,
        universe_snapshot=snapshot,
        result=result,
        code_revision="test-rev-123",
        asset_store=sim_asset_store,
        register_data_asset=True,
    )

    assert run.status == SimulationRun.Status.COMPLETE
    assert run.metrics["total_trades"] == 3
    assert run.code_revision == "test-rev-123"
    assert run.result_asset_key.startswith("simulations/")

    # Holdings persisted in DB
    holdings = list(SimulationHolding.objects.filter(run=run))
    assert len(holdings) > 0
    for h in holdings:
        assert h.quantity > 0
        assert h.market_value > 0
        assert 0 <= h.weight <= 1

    # Trades persisted in DB with round-trip side assertion
    db_trades = list(SimulationTrade.objects.filter(run=run).order_by("trade_date", "side"))
    assert len(db_trades) == 3

    # Assert both BUY and SELL sides are present and match model choices
    trade_sides = {t.side for t in db_trades}
    assert trade_sides == {SimulationTrade.Side.BUY, SimulationTrade.Side.SELL}

    # Verify field-by-field round-trip against result.trades
    for db_t in db_trades:
        assert db_t.quantity > 0
        assert db_t.price > 0
        assert db_t.gross_value > 0
        assert db_t.costs >= 0
        matching = result.trades.filter(
            (pl.col("listing_id") == str(db_t.listing_id))
            & (pl.col("trade_date") == db_t.trade_date)
            & (pl.col("side") == db_t.side)
        )
        assert matching.height == 1
        assert matching["side"][0] == db_t.side

    # Parquet registered in DataAsset
    data_asset = DataAsset.objects.get(relative_path=run.result_asset_key)
    assert data_asset.provider == "simulation"
    assert data_asset.kind == "simulation_result"

    # Loaded back via service
    loaded = load_simulation_run(run.id, asset_store=sim_asset_store)
    assert loaded.run.id == run.id
    assert loaded.daily_curves.height == 3
    assert "portfolio_value" in loaded.daily_curves.columns
    assert loaded.holdings_count == len(holdings)
    assert loaded.trades_count == 3


@pytest.mark.django_db
def test_execute_and_persist_failure_records_failed_status(
    sim_asset_store: AssetStore,
    test_setup_models: tuple[SimulationDefinition, UniverseSnapshot, Listing, Listing],
) -> None:
    definition, snapshot, listing1, _ = test_setup_models
    d0 = date(2026, 1, 1)
    d2 = date(2026, 1, 5)

    # Missing price on intermediate weekday for AAPL
    prices = pl.DataFrame(
        {
            "date": [d0, d2],
            "listing_id": [str(listing1.id), str(listing1.id)],
            "close": [100.0, 110.0],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    # Signal on 2025-12-31 trades on d0 (2026-01-01), so AAPL is held going into d1
    signals = pl.DataFrame(
        {
            "date": [date(2025, 12, 31)],
            "listing_id": [str(listing1.id)],
            "score": [1.0],
        }
    )

    definition.config = {
        "top_n": 1,
        "missing_price_policy": "fail",
        "starting_capital": 10000.0,
    }
    definition.save()

    with pytest.raises(MissingPriceError):
        execute_and_persist_simulation(
            definition=definition,
            universe_snapshot=snapshot,
            prices=prices,
            signals=signals,
            calendar=[d0, date(2026, 1, 2), d2],
            asset_store=sim_asset_store,
        )

    failed_runs = SimulationRun.objects.filter(
        definition=definition, status=SimulationRun.Status.FAILED
    )
    assert failed_runs.count() == 1
    failed_run = failed_runs.first()
    assert failed_run is not None
    assert "Missing price" in failed_run.error


def test_example_yaml_configs_parse_valid() -> None:
    """Example YAML configs in config/simulations/ are valid and load correctly."""
    top10_path = Path("config/simulations/top10_monthly.yaml")
    buy_hold_path = Path("config/simulations/selected_buy_and_hold.yaml")

    assert top10_path.exists()
    assert buy_hold_path.exists()

    top10_config = SimulationConfig.from_yaml(top10_path)
    top10_config.validate()
    assert top10_config.top_n == 10
    assert top10_config.rebalance_frequency == RebalanceFrequency.MONTHLY
    assert top10_config.mode == SimulationMode.BACKTEST

    buy_hold_config = SimulationConfig.from_yaml(buy_hold_path)
    buy_hold_config.validate()
    assert buy_hold_config.selected_symbols == ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    assert buy_hold_config.rebalance_frequency == RebalanceFrequency.NEVER
    assert buy_hold_config.mode == SimulationMode.PORTFOLIO


@pytest.mark.django_db
def test_persistence_fails_for_non_uuid_or_unresolved_listing_id(
    sim_asset_store: AssetStore,
    test_setup_models: tuple[SimulationDefinition, UniverseSnapshot, Listing, Listing],
) -> None:
    """Persistence requires permanent UUID listing IDs and leaves no orphan files/assets."""
    definition, snapshot, _, _ = test_setup_models
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)

    # 1. Non-UUID ticker identifier
    prices_ticker = pl.DataFrame(
        {
            "date": [d0, d1],
            "listing_id": ["TICKER_ONLY", "TICKER_ONLY"],
            "close": [100.0, 105.0],
            "symbol": ["TICKER_ONLY", "TICKER_ONLY"],
        }
    )
    signals_ticker = pl.DataFrame(
        {
            "date": [d0],
            "listing_id": ["TICKER_ONLY"],
            "score": [1.0],
        }
    )
    res_ticker = AccountingEngine(SimulationConfig(starting_capital=10000.0, top_n=1)).run(
        prices=prices_ticker, signals=signals_ticker
    )

    with pytest.raises(ValueError, match="Permanent UUID listing IDs are required"):
        persist_simulation_run(
            definition=definition,
            universe_snapshot=snapshot,
            result=res_ticker,
            asset_store=sim_asset_store,
        )

    # Verify no parquet file was written to asset_store
    target_dir = sim_asset_store.root / "simulations"
    assert not target_dir.exists() or len(list(target_dir.iterdir())) == 0

    # 2. Unresolved random UUID
    random_uuid = "12345678-1234-5678-1234-567812345678"
    prices_unresolved = pl.DataFrame(
        {
            "date": [d0, d1],
            "listing_id": [random_uuid, random_uuid],
            "close": [100.0, 105.0],
            "symbol": ["FAKE", "FAKE"],
        }
    )
    signals_unresolved = pl.DataFrame(
        {
            "date": [d0],
            "listing_id": [random_uuid],
            "score": [1.0],
        }
    )
    res_unresolved = AccountingEngine(SimulationConfig(starting_capital=10000.0, top_n=1)).run(
        prices=prices_unresolved, signals=signals_unresolved
    )

    with pytest.raises(ValueError, match="listing UUIDs not found in database"):
        persist_simulation_run(
            definition=definition,
            universe_snapshot=snapshot,
            result=res_unresolved,
            asset_store=sim_asset_store,
        )


@pytest.mark.django_db
def test_persistence_validates_grade_and_mode_match(
    sim_asset_store: AssetStore,
    test_setup_models: tuple[SimulationDefinition, UniverseSnapshot, Listing, Listing],
) -> None:
    """Validate SimulationConfig grade matches UniverseSnapshot and mode matches definition."""
    definition, snapshot, listing1, _ = test_setup_models
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)

    prices = pl.DataFrame(
        {
            "date": [d0, d1],
            "listing_id": [str(listing1.id), str(listing1.id)],
            "close": [100.0, 105.0],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [d0],
            "listing_id": [str(listing1.id)],
            "score": [1.0],
        }
    )

    # 1. Grade mismatch: config says RESEARCH, snapshot is OBSERVED
    cfg_grade_mismatch = SimulationConfig(
        starting_capital=10000.0,
        top_n=1,
        mode=SimulationMode.BACKTEST,
        grade=SimulationGrade.RESEARCH,
    )
    res_grade_mismatch = AccountingEngine(cfg_grade_mismatch).run(prices=prices, signals=signals)
    with pytest.raises(ValueError, match="SimulationConfig grade 'research' does not match"):
        persist_simulation_run(
            definition=definition,
            universe_snapshot=snapshot,
            result=res_grade_mismatch,
            asset_store=sim_asset_store,
        )

    # 2. Mode mismatch: config says PORTFOLIO, definition is BACKTEST
    cfg_mode_mismatch = SimulationConfig(
        starting_capital=10000.0,
        top_n=1,
        mode=SimulationMode.PORTFOLIO,
        grade=SimulationGrade.OBSERVED,
    )
    res_mode_mismatch = AccountingEngine(cfg_mode_mismatch).run(prices=prices, signals=signals)
    with pytest.raises(ValueError, match="SimulationConfig mode 'portfolio' does not match"):
        persist_simulation_run(
            definition=definition,
            universe_snapshot=snapshot,
            result=res_mode_mismatch,
            asset_store=sim_asset_store,
        )


@pytest.mark.django_db
def test_run_persists_and_identifies_all_input_assets(
    sim_asset_store: AssetStore,
    test_setup_models: tuple[SimulationDefinition, UniverseSnapshot, Listing, Listing],
) -> None:
    """Exact input frames are persisted as immutable DataAssets; run identifies all inputs used."""
    definition, snapshot, listing1, _ = test_setup_models
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)

    prices = pl.DataFrame(
        {
            "date": [d0, d1],
            "listing_id": [str(listing1.id), str(listing1.id)],
            "close": [100.0, 105.0],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [d0],
            "listing_id": [str(listing1.id)],
            "score": [1.0],
        }
    )
    benchmark = pl.DataFrame(
        {
            "date": [d0, d1],
            "close": [50.0, 52.0],
        }
    )

    run, _ = execute_and_persist_simulation(
        definition=definition,
        universe_snapshot=snapshot,
        prices=prices,
        signals=signals,
        benchmark_prices=benchmark,
        asset_store=sim_asset_store,
    )

    # 1. Primary result pointer is preserved
    assert run.result_asset_key == f"simulations/{run.id}/results.parquet"

    # 2. Run metrics identify every input used
    assert "input_assets" in run.metrics
    input_assets = run.metrics["input_assets"]
    assert "prices" in input_assets
    assert "signals" in input_assets
    assert "benchmark" in input_assets

    # 3. Check each input asset's recorded metadata and DataAsset row
    for kind, key_name in [
        ("simulation_input_prices", "prices"),
        ("simulation_input_signals", "signals"),
        ("simulation_input_benchmark", "benchmark"),
    ]:
        info = input_assets[key_name]
        asset_id = info["asset_id"]
        sha256 = info["sha256"]
        rel_path = info["relative_path"]

        data_asset = DataAsset.objects.get(id=asset_id)
        assert data_asset.kind == kind
        assert data_asset.sha256 == sha256
        assert data_asset.relative_path == rel_path

    # 4. Result asset DataAsset metadata also records input assets
    result_asset = DataAsset.objects.get(relative_path=run.result_asset_key)
    assert result_asset.metadata["input_assets"] == input_assets

    # 5. Loaded run retrieves the exact input DataFrames
    loaded = load_simulation_run(run.id, asset_store=sim_asset_store)
    assert "prices" in loaded.input_assets
    assert loaded.input_assets["prices"].equals(prices)
    assert "signals" in loaded.input_assets
    assert loaded.input_assets["signals"].equals(signals)
    assert "benchmark" in loaded.input_assets
    assert loaded.input_assets["benchmark"].equals(benchmark)
