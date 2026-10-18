from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from django.contrib.auth import get_user_model
from django.db import connection
from django.test import Client
from django.urls import reverse
from django.utils import timezone

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import (
    Company,
    DataAsset,
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
from stanstock.simulation.builders import (
    build_price_panel,
    build_signals_for_backtest,
    run_simulation_workflow,
)
from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from stanstock.simulation.service import load_simulation_run, persist_simulation_run
from stanstock.simulation.types import (
    ACCOUNTING_VERSION,
    ACCOUNTING_VERSION_PARAMETER,
    SimulationConfig,
    SimulationResult,
    SimulationWorkflowError,
)


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


@pytest.fixture
def persist_web_simulation(web_setup_environment, settings):
    snapshot, listing, store = web_setup_environment
    settings.DATA_DIR = store.root

    def persist(version: str | None = ACCOUNTING_VERSION) -> tuple[SimulationRun, SimulationResult]:
        config = SimulationConfig(
            selected_symbols=[listing.ticker],
            custom_parameters={} if version is None else {ACCOUNTING_VERSION_PARAMETER: version},
        )
        days = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
        prices = pl.DataFrame(
            {
                "date": days,
                "close": [100.0, 110.0, 121.0],
                "listing_id": [str(listing.id)] * 3,
                "symbol": [listing.ticker] * 3,
            }
        )
        benchmark = prices.select("date", "close")
        result = AccountingEngine(config).run(
            prices=prices, benchmark_prices=benchmark, calendar=days
        )
        definition = SimulationDefinition.objects.create(
            name="Version disclosure", mode=config.mode.value, config=config.to_dict()
        )
        run = persist_simulation_run(
            definition=definition,
            universe_snapshot=snapshot,
            result=result,
            input_prices=prices,
            input_benchmark=benchmark,
            asset_store=store,
        )
        return SimulationRun.objects.get(pk=run.pk), result

    return persist


@pytest.mark.django_db
def test_simulations_page_get(auth_client: Client) -> None:
    """GET and HEAD retain historical records, without a creator."""
    resp = auth_client.get(reverse("simulations"))
    assert resp.status_code == 200
    assert "form" not in resp.context
    assert b"Run simulation" not in resp.content
    assert b"Historical simulations" in resp.content
    assert auth_client.head(reverse("simulations")).status_code == 200


@pytest.mark.django_db
def test_simulations_page_post_unauthenticated() -> None:
    """Unauthenticated POST /simulations redirects to login."""
    client = Client()
    resp = client.post(reverse("simulations"), {})
    assert resp.status_code == 302
    location = resp.headers.get("Location", "")
    assert "/login" in location or "accounts/login" in location or "next=" in location


@pytest.mark.django_db
def test_simulation_workflow_records_remain_readable(
    auth_client: Client,
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
    settings,
) -> None:
    """The retained engine's synthetic records still have a list and detail."""
    snapshot, listing, store = web_setup_environment
    settings.DATA_DIR = store.root

    _, run, _ = run_simulation_workflow(
        name="Retained synthetic portfolio",
        mode="portfolio",
        snapshot=snapshot,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        starting_capital=75000,
        selected_listing_ids=[listing.pk],
        asset_store=store,
    )
    assert run.status == SimulationRun.Status.COMPLETE
    detail = auth_client.get(reverse("simulation-detail", args=[run.pk]))
    assert detail.status_code == 200
    assert detail.context["metrics_verified"] is True
    assert b"Retained synthetic portfolio" in auth_client.get(reverse("simulations")).content

    # Verify UUID identity
    for h in run.holdings.all():
        assert h.listing_id == listing.id
    for t in run.trades.all():
        assert t.listing_id == listing.id


@pytest.mark.django_db
@pytest.mark.parametrize("version", (None, ACCOUNTING_VERSION))
def test_simulation_detail_discloses_only_validated_calculation_version(
    auth_client,
    persist_web_simulation,
    version,
):
    run, result = persist_web_simulation(version)
    definition = run.definition
    response = auth_client.get(reverse("simulation-detail", args=[run.id]))
    assert response.status_code == 200
    assert response.context["metrics_verified"] is True
    if version is None:
        assert b"Legacy calculation" in response.content
        assert b"initial trading costs may be omitted" in response.content
        definition.config["custom_parameters"] = {ACCOUNTING_VERSION_PARAMETER: "unknown"}
        definition.save(update_fields=["config"])
        unknown = auth_client.get(reverse("simulation-detail", args=[run.id]))
        assert unknown.context["metrics_verified"] is False
        assert b"Legacy calculation" not in unknown.content
    else:
        assert ACCOUNTING_VERSION.encode() in response.content
        assert b"2 matched benchmark intervals" in response.content
        assert b"Drawdown includes inception capital" in response.content
        assert response.context["calculation"]["coverage"] == result.metrics.comparison_coverage
        assert not {"calculation_contract", "comparison_coverage"} & {
            name for name, _value in response.context["metrics"]
        }
        run.metrics["calculation_contract"]["effective_config"]["custom_parameters"][
            ACCOUNTING_VERSION_PARAMETER
        ] = "unknown"
        run.save(update_fields=["metrics"])
        unknown = auth_client.get(reverse("simulation-detail", args=[run.id]))
        assert unknown.context["metrics_verified"] is False
        assert b"Legacy calculation" not in unknown.content
        run.metrics["calculation_contract"]["effective_config"]["custom_parameters"][
            ACCOUNTING_VERSION_PARAMETER
        ] = ACCOUNTING_VERSION
        run.metrics["comparison_coverage"]["matched_intervals"] = 99
        run.save(update_fields=["metrics"])
        response = auth_client.get(reverse("simulation-detail", args=[run.id]))
        assert response.context["metrics_verified"] is False
        assert response.context["metrics"] == []
        assert b"Calculation provenance could not be verified" in response.content
        assert b"Legacy calculation" not in response.content
        # Removing a v2 marker in mutable surfaces cannot create trusted legacy.
        run.metrics.pop("calculation_contract")
        run.metrics.pop("comparison_coverage")
        run.save(update_fields=["metrics"])
        definition.config["custom_parameters"] = {}
        definition.save(update_fields=["config"])
        response = auth_client.get(reverse("simulation-detail", args=[run.id]))
        assert response.context["metrics_verified"] is False
        assert b"Legacy calculation" not in response.content


@pytest.mark.django_db
@pytest.mark.parametrize(
    "fault",
    (
        "definition",
        "snapshot",
        "simulation_result",
        "simulation_input_prices",
        "simulation_input_benchmark",
        "missing_simulation_result",
        "missing_simulation_input_prices",
        "missing_simulation_input_benchmark",
    ),
)
def test_simulation_detail_withholds_invalid_v2_evidence_between_requests(
    auth_client, persist_web_simulation, web_setup_environment, monkeypatch, fault
):
    run, _result = persist_web_simulation()
    snapshot, _listing, store = web_setup_environment
    url = reverse("simulation-detail", args=[run.id])
    assert auth_client.get(url).context["metrics_verified"] is True
    if fault == "definition":
        other_definition = SimulationDefinition.objects.create(
            name="Unrelated synthetic definition", mode="portfolio", config={}
        )
        SimulationRun.objects.filter(pk=run.id).update(definition=other_definition)
    elif fault == "snapshot":
        other_snapshot = UniverseSnapshot.objects.create(
            universe=snapshot.universe,
            as_of_date=date(2025, 12, 31),
            grade=UniverseSnapshot.Grade.RESEARCH,
            config_hash="unrelated-synthetic-snapshot",
        )
        SimulationRun.objects.filter(pk=run.id).update(universe_snapshot=other_snapshot)
    else:
        asset = DataAsset.objects.get(kind=fault.removeprefix("missing_"), subject=str(run.id))
        path = store.resolve(asset.relative_path)
        if fault.startswith("missing_"):
            path.unlink()
        else:
            path.write_bytes(b"corrupt synthetic registered evidence")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("GET cannot calculate, fetch, resolve credentials or write assets")

    monkeypatch.setattr(AccountingEngine, "run", forbidden)
    monkeypatch.setattr(AssetStore, "write_bytes", forbidden)
    monkeypatch.setattr(AssetStore, "write_frame", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("httpx.AsyncClient.send", forbidden)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", forbidden)

    def read_only(execute, sql, params, many, context):
        assert sql.lstrip().split()[0].upper() not in {"INSERT", "UPDATE", "DELETE"}
        return execute(sql, params, many, context)

    with connection.execute_wrapper(read_only):
        response = auth_client.get(url)
    assert response.status_code == 200
    assert response.context["metrics_verified"] is False
    assert response.context["metrics"] == []
    assert response.context["holdings"] == response.context["trades"] == []
    assert b"Calculation provenance could not be verified" in response.content
    assert str(store.root).encode() not in response.content
    assert b"No metrics were persisted" not in response.content
    assert b"No trades were persisted" not in response.content
    assert b"No holdings were persisted" not in response.content
    with pytest.raises((ValueError, RefreshVerificationError)):
        load_simulation_run(run.id, asset_store=store)


@pytest.mark.django_db
def test_simulation_detail_preserves_recorded_execution_context(
    auth_client, persist_web_simulation, web_setup_environment
):
    run, _result = persist_web_simulation()
    snapshot, _listing, store = web_setup_environment
    definition = run.definition
    definition.mode = SimulationDefinition.Mode.PORTFOLIO
    definition.config = {
        "top_n": 99,
        "custom_parameters": {ACCOUNTING_VERSION_PARAMETER: "unknown"},
    }
    definition.save(update_fields=["mode", "config"])
    snapshot.grade = UniverseSnapshot.Grade.RESEARCH
    snapshot.save(update_fields=["grade"])
    response = auth_client.get(reverse("simulation-detail", args=[run.id]))
    assert response.status_code == 200
    assert response.context["metrics_verified"] is True
    assert response.context["calculation"]["version"] == ACCOUNTING_VERSION
    expected = (
        f'<p class="eyebrow">{SimulationDefinition.Mode.BACKTEST.label} \u00b7 '
        f"{UniverseSnapshot.Grade.OBSERVED.label}</p>"
    )
    assert expected.encode() in response.content
    assert load_simulation_run(run.id, asset_store=store).calculation_version == ACCOUNTING_VERSION


@pytest.mark.django_db
def test_simulations_page_post_backtest_no_signals_rejected(
    auth_client: Client,
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """Even invalid old creator requests stop before signal lookup."""
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

    assert resp.status_code == 405
    assert resp.headers["Allow"] == "GET, HEAD"

    # No definition or run created
    assert SimulationDefinition.objects.count() == initial_defs
    assert SimulationRun.objects.count() == initial_runs


@pytest.mark.django_db
def test_simulations_page_post_ineligible_listing_rejected(
    auth_client: Client,
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    """The retired creator no longer constructs or validates a listing form."""
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

    assert resp.status_code == 405


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
        issued_on_time=True,
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
        issued_on_time=True,
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
        match="Observed backtests require signals issued before the next live session",
    ):
        build_signals_for_backtest(
            snapshot=snapshot,
            start_date=target,
            end_date=target,
        )


@pytest.mark.django_db
def test_observed_backtest_accepts_an_on_time_overnight_signal(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    snapshot, listing, _store = web_setup_environment
    target = date(2026, 1, 2)
    snapshot.as_of_date = target
    snapshot.save(update_fields=["as_of_date"])
    generated_at = datetime(2026, 1, 3, 1, tzinfo=timezone.get_current_timezone())
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=target,
        issued_on_time=True,
        universe_snapshot=snapshot,
        config_version="v1",
        config_hash="overnight",
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

    assert signals.height == 1
    assert signals["data_cutoff"].to_list() == [generated_at]


@pytest.mark.django_db
def test_observed_backtest_rejects_a_late_run_even_if_flagged_on_time(
    web_setup_environment: tuple[UniverseSnapshot, Listing, AssetStore],
) -> None:
    snapshot, listing, _store = web_setup_environment
    target = date(2026, 1, 2)
    snapshot.as_of_date = target
    snapshot.save(update_fields=["as_of_date"])
    generated_at = datetime(2026, 1, 5, 15, tzinfo=timezone.get_current_timezone())
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=datetime(2026, 1, 2, 23, 59, tzinfo=timezone.get_current_timezone()),
        target_date=target,
        issued_on_time=True,
        universe_snapshot=snapshot,
        config_version="v1",
        config_hash="invalid-late",
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
        match="Observed backtests require signals issued before the next live session",
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
def test_price_panel_requires_explicit_base_for_mixed_currency_and_supports_native_filter(
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

    with pytest.raises(SimulationWorkflowError, match="needs an explicit base currency"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 2),
            provider="synthetic_demo",
            asset_store=store,
        )

    panel = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        provider="synthetic_demo",
        restrict_native_currency="USD",
        asset_store=store,
    )
    assert set(panel.prices["listing_id"].to_list()) == {str(usd_listing.id)}
    assert set(panel.prices["currency"].to_list()) == {"USD"}
    assert panel.base_currency == "USD"
    assert panel.fx_rates is None
    assert panel.conversion_applied is False


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
