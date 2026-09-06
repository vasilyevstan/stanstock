from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from uuid import uuid4

import polars as pl
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction
from django.test import override_settings
from django.utils import timezone

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.management.config_loader import default_us_scoring_config_path
from stanstock.data.models import (
    Company,
    FundamentalFact,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.service import analyze_listing, analyze_snapshot


@pytest.mark.django_db
def test_service_persists_analysis_and_appends_immutable_predictions(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))

    first = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
    )
    first_prediction_ids = set(first.predictions[index].pk for index in range(3))
    second = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now + timedelta(seconds=1),
        provider="synthetic",
        store=store,
    )

    assert AnalysisRun.objects.count() == 2
    assert StockAnalysis.objects.count() == 2
    assert Prediction.objects.count() == 6
    assert {prediction.horizon for prediction in first.predictions} == {"short", "medium", "long"}
    assert first_prediction_ids.isdisjoint({prediction.pk for prediction in second.predictions})
    assert all(prediction.probability_positive is None for prediction in first.predictions)
    assert all(prediction.insufficiency_reason for prediction in first.predictions)
    assert all(prediction.issued_on_time for prediction in first.predictions)
    assert first.analysis.reasons == first.computation.reasons
    assert first.analysis.data_quality["coverage"] > 0
    assert first.analysis.data_quality["source_assets"] == first.computation.source_assets
    assert "recommendation_gates" in first.analysis.data_quality
    assert "buy_liquidity_present" in first.analysis.data_quality["recommendation_gates"]
    source_asset = first.analysis.data_quality["source_assets"][0]
    assert source_asset["provider"] == "synthetic"
    assert source_asset["kind"] == "price_history"
    assert source_asset["subject"] == listing.ticker
    assert source_asset["retrieved_at"]
    assert source_asset["available_at"]
    assert first.analysis.component_scores["components"]


@pytest.mark.django_db
def test_analyze_snapshot_creates_one_run_for_multiple_listings(tmp_path) -> None:
    target = date(2026, 8, 31)
    listing_one, snapshot = _listing_and_snapshot(as_of_date=target)
    listing_two = _add_listing_to_snapshot(snapshot)
    store = AssetStore(tmp_path)
    now = timezone.now()
    available_at = datetime(2026, 8, 31, 20, tzinfo=timezone.get_current_timezone())
    for listing in (listing_one, listing_two):
        price_asset = _register_price_asset(store, listing.ticker, available_at)
        _create_facts(listing, price_asset, available_at)

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=now,
        target_date=target,
        provider="synthetic",
        store=store,
    )

    assert len(results) == 2
    assert AnalysisRun.objects.count() == 1
    assert StockAnalysis.objects.count() == 2
    assert Prediction.objects.count() == 6
    assert {result.run.pk for result in results} == {results[0].run.pk}
    assert results[0].run.stocks.count() == 2
    assert results[0].run.generated_at == now
    assert results[0].run.data_cutoff.date() == target
    assert results[0].run.target_date == target
    assert {prediction.target_date for prediction in Prediction.objects.all()} == {target}


@pytest.mark.django_db
def test_analyze_command_accepts_logical_target_date_separate_from_generation_time(
    tmp_path,
) -> None:
    target = date(2026, 8, 30)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    available_at = datetime(2026, 8, 30, 20, tzinfo=timezone.get_current_timezone())
    price_asset = _register_price_asset(store, listing.ticker, available_at)
    _create_facts(listing, price_asset, available_at)

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "analyze",
            snapshot=str(snapshot.pk),
            listing=str(listing.pk),
            provider="synthetic",
            target_date=target.isoformat(),
            stdout=StringIO(),
        )

    run = AnalysisRun.objects.get()
    assert run.target_date == target
    assert run.data_cutoff.date() == target
    assert run.generated_at.date() != run.target_date
    assert set(Prediction.objects.values_list("target_date", flat=True)) == {date(2026, 8, 30)}


@pytest.mark.django_db
def test_analysis_persists_null_prediction_ranges_when_scenario_inputs_are_insufficient(
    tmp_path,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    _register_price_asset(store, listing.ticker, now - timedelta(minutes=5), rows=20)

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
    )

    assert persisted.analysis.short_scenario["bear"] is None
    assert persisted.analysis.medium_scenario["base"] is None
    assert persisted.analysis.long_scenario["bull"] is None
    assert all(prediction.bear_return is None for prediction in persisted.predictions)
    assert all(prediction.base_return is None for prediction in persisted.predictions)
    assert all(prediction.bull_return is None for prediction in persisted.predictions)
    assert all(prediction.insufficiency_reason for prediction in persisted.predictions)


@pytest.mark.django_db
def test_analysis_filters_eligible_price_asset_through_logical_target_date(tmp_path) -> None:
    store = AssetStore(tmp_path)
    target = date(2026, 1, 31)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    now = datetime(2026, 2, 5, tzinfo=timezone.get_current_timezone())
    dates = [target - timedelta(days=20 - index) for index in range(21)]
    asset_closes = [101.0 + index for index in range(21)]
    benchmark_closes = [201.0 + index for index in range(21)]
    _register_explicit_price_asset(
        store,
        listing.ticker,
        datetime(2026, 1, 31, 20, tzinfo=timezone.get_current_timezone()),
        dates + [target + timedelta(days=1)],
        asset_closes + [999.0],
    )
    _register_explicit_price_asset(
        store,
        "BENCH",
        datetime(2026, 1, 31, 20, tzinfo=timezone.get_current_timezone()),
        dates + [target + timedelta(days=1)],
        benchmark_closes + [9999.0],
    )

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        target_date=target,
        provider="synthetic",
        benchmark_subject="BENCH",
        store=store,
    )

    expected_relative = (asset_closes[-1] / asset_closes[0] - 1.0) - (
        benchmark_closes[-1] / benchmark_closes[0] - 1.0
    )
    assert persisted.analysis.current_price == Decimal(str(asset_closes[-1]))
    assert persisted.analysis.run.target_date == target
    assert persisted.computation.indicators.values["last_close"] == asset_closes[-1]
    assert persisted.computation.indicators.values["relative_return_20d"] == pytest.approx(
        expected_relative
    )
    assert {asset["subject"] for asset in persisted.computation.source_assets} >= {
        listing.ticker,
        "BENCH",
    }


@pytest.mark.django_db
def test_analysis_excludes_fundamentals_unavailable_at_historical_cutoff(tmp_path) -> None:
    target = date(2026, 1, 31)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    cutoff_asset = _register_price_asset(
        store,
        listing.ticker,
        datetime(2026, 1, 31, 20, tzinfo=timezone.get_current_timezone()),
    )
    company = listing.security.company
    for period_end, value, available_at in (
        (
            date(2023, 12, 31),
            "100",
            datetime(2025, 2, 1, tzinfo=timezone.get_current_timezone()),
        ),
        (
            date(2024, 12, 31),
            "120",
            datetime(2026, 1, 30, tzinfo=timezone.get_current_timezone()),
        ),
        (
            date(2025, 12, 31),
            "999",
            datetime(2026, 2, 1, tzinfo=timezone.get_current_timezone()),
        ),
    ):
        FundamentalFact.objects.create(
            company=company,
            provider="synthetic",
            concept="Revenue",
            source_concept="us-gaap:Revenue",
            value=Decimal(value),
            unit="USD",
            currency="USD",
            period_end=period_end,
            fiscal_year=period_end.year,
            fiscal_period="FY",
            accession=f"{period_end.year}-{uuid4().hex[:8]}",
            available_at=available_at,
            source_asset=cutoff_asset,
        )

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=datetime(2026, 2, 5, tzinfo=timezone.get_current_timezone()),
        target_date=target,
        provider="synthetic",
        store=store,
    )

    assert persisted.computation.fundamentals.values["revenue_growth"] == pytest.approx(0.20)
    assert persisted.run.data_cutoff.date() == target
    assert all(prediction.data_cutoff.date() == target for prediction in persisted.predictions)


@pytest.mark.django_db
def test_analyze_listing_rejects_nonmember_and_future_target(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    outsider, _other_snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))

    with pytest.raises(ValueError, match="not an eligible member"):
        analyze_listing(
            listing=outsider,
            universe_snapshot=snapshot,
            decision_time=now,
            provider="synthetic",
            store=store,
        )

    with pytest.raises(ValueError, match="cannot be after generation date"):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=now,
            target_date=now.date() + timedelta(days=1),
            provider="synthetic",
            store=store,
        )


@pytest.mark.django_db
def test_predict_command_preserves_source_asset_provenance(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))
    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
    )

    call_command(
        "predict",
        analysis=str(persisted.analysis.pk),
        model_version="manual-v2",
        stdout=StringIO(),
    )

    appended = Prediction.objects.filter(model_version="manual-v2").order_by("horizon")
    assert appended.count() == 3
    assert all(
        prediction.source_assets == persisted.analysis.data_quality["source_assets"]
        for prediction in appended
    )
    assert all(not prediction.issued_on_time for prediction in appended)


@pytest.mark.django_db
def test_price_only_analysis_ignores_fundamentals_and_persists_only_short_predictions(
    tmp_path,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now() - timedelta(seconds=2)
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))

    before = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    _create_extreme_facts(listing, price_asset, now + timedelta(milliseconds=100))
    after = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now + timedelta(seconds=1),
        provider="synthetic",
        store=store,
        config_path=default_us_scoring_config_path(),
    )

    assert before.computation.fundamentals.values == {}
    assert after.computation.fundamentals.values == {}
    assert before.computation.aggregate.overall == after.computation.aggregate.overall
    assert before.computation.risk_score == after.computation.risk_score
    assert before.computation.recommendation == after.computation.recommendation
    assert before.analysis.data_quality["fundamentals_used"] is False
    assert before.run.config_version == "us-price-baseline-v2"
    assert before.analysis.data_quality["factor_policy"] == {
        "macd_indicator": "macd_histogram_pct",
        "macd_score_low": -0.02,
        "macd_score_high": 0.02,
        "abnormal_volume_indicator": "abnormal_volume_strict",
        "liquidity_indicator": "avg_dollar_volume_20d",
        "liquidity_score_low": 1_000_000.0,
        "liquidity_score_high": 50_000_000.0,
        "strict_finite_inputs": True,
        "buy_min_liquidity_20d": 5_000_000.0,
    }
    assert {prediction.horizon for prediction in after.predictions} == {Prediction.Horizon.SHORT}
    assert set(after.analysis.component_scores["horizons"]) == {Prediction.Horizon.SHORT}
    assert all(asset["kind"] == "price_history" for asset in after.computation.source_assets)

    call_command(
        "predict",
        analysis=str(after.analysis.pk),
        model_version="price-only-reissue",
        stdout=StringIO(),
    )
    reissued = Prediction.objects.get(model_version="price-only-reissue")
    assert reissued.horizon == Prediction.Horizon.SHORT
    assert reissued.issued_on_time is False


@pytest.mark.django_db
def test_analysis_rejects_an_observed_run_after_the_next_market_open(tmp_path) -> None:
    target = date(2026, 9, 4)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)

    with pytest.raises(ValueError, match="after the next market session opened"):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=datetime(2026, 9, 8, 14, tzinfo=UTC),
            target_date=target,
            issued_on_time=True,
            provider="synthetic",
            store=AssetStore(tmp_path),
        )


@pytest.mark.django_db
def test_predict_command_default_model_version_is_collision_safe(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))
    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
    )

    call_command("predict", analysis=str(persisted.analysis.pk), stdout=StringIO())
    call_command("predict", analysis=str(persisted.analysis.pk), stdout=StringIO())

    initial_version = persisted.predictions[0].model_version
    appended_versions = set(
        Prediction.objects.exclude(model_version=initial_version).values_list(
            "model_version", flat=True
        )
    )
    assert len(appended_versions) == 2
    assert all(len(version) <= 40 for version in appended_versions)
    assert Prediction.objects.count() == 9


@pytest.mark.django_db
def test_predict_command_explicit_repeated_model_version_fails_without_overwrite(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))
    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
    )

    call_command(
        "predict",
        analysis=str(persisted.analysis.pk),
        model_version="manual-collision",
        stdout=StringIO(),
    )
    with pytest.raises(IntegrityError), transaction.atomic():
        call_command(
            "predict",
            analysis=str(persisted.analysis.pk),
            model_version="manual-collision",
            stdout=StringIO(),
        )

    assert Prediction.objects.filter(model_version="manual-collision").count() == 3


@pytest.mark.django_db
def test_predict_command_rejects_incomplete_scenario_payload(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))
    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
    )
    persisted.analysis.short_scenario = {"base": 0.01, "bull": 0.02}
    persisted.analysis.save(update_fields=["short_scenario"])

    with pytest.raises(CommandError, match="scenario payload is incomplete"):
        call_command(
            "predict",
            analysis=str(persisted.analysis.pk),
            model_version="manual-v3",
            stdout=StringIO(),
        )

    assert Prediction.objects.filter(model_version="manual-v3").count() == 0


def _listing_and_snapshot(
    *,
    as_of_date: date | None = None,
) -> tuple[Listing, UniverseSnapshot]:
    company = Company.objects.create(name="Research Co", country="US", sector="Technology")
    security = Security.objects.create(company=company, name="Research Co Common")
    listing = Listing.objects.create(
        security=security,
        ticker=f"R{uuid4().hex[:6]}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug=f"research-{uuid4().hex[:8]}",
        name="Research universe",
        config_version="default-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=as_of_date or timezone.localdate(),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    return listing, snapshot


def _add_listing_to_snapshot(snapshot: UniverseSnapshot) -> Listing:
    company = Company.objects.create(name=f"Research Co {uuid4().hex[:6]}", country="US")
    security = Security.objects.create(company=company, name="Research Co Secondary")
    listing = Listing.objects.create(
        security=security,
        ticker=f"R{uuid4().hex[:6]}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    return listing


def _register_price_asset(
    store: AssetStore, subject: str, available_at, *, rows: int = 280
) -> object:
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(rows)]
    closes = [50 + index * 0.08 + (index % 9) * 0.02 for index in range(rows)]
    frame = pl.DataFrame(
        {
            "date": dates,
            "open": [close - 0.1 for close in closes],
            "high": [close + 0.8 for close in closes],
            "low": [close - 0.8 for close in closes],
            "close": closes,
            "volume": [500_000 + index * 1_000 for index in range(rows)],
        }
    )
    stored = store.write_frame(f"research-tests/{uuid4().hex}.parquet", frame)
    return register_asset(
        provider="synthetic",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )


def _register_explicit_price_asset(
    store: AssetStore,
    subject: str,
    available_at,
    dates: list[date],
    closes: list[float],
) -> object:
    frame = pl.DataFrame(
        {
            "date": dates,
            "open": [close - 0.1 for close in closes],
            "high": [close + 0.8 for close in closes],
            "low": [close - 0.8 for close in closes],
            "close": closes,
            "volume": [500_000 + index * 1_000 for index in range(len(closes))],
        }
    )
    stored = store.write_frame(f"research-tests/{uuid4().hex}.parquet", frame)
    return register_asset(
        provider="synthetic",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )


def _create_facts(listing: Listing, source_asset: object, available_at) -> None:
    periods = [date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)]
    rows = [
        {"revenue": 900, "net_income": 100, "free_cash_flow": 90},
        {"revenue": 1000, "net_income": 120, "free_cash_flow": 100},
        {
            "revenue": 1150,
            "gross_profit": 600,
            "operating_income": 220,
            "net_income": 150,
            "free_cash_flow": 130,
            "cash_and_equivalents": 240,
            "total_debt": 180,
            "shareholders_equity": 500,
            "total_assets": 1400,
            "interest_expense": -35,
            "current_assets": 650,
            "current_liabilities": 325,
            "nopat": 170,
            "invested_capital": 620,
            "shares_outstanding": 10,
            "ebitda": 260,
        },
    ]
    for period, mapping in zip(periods, rows, strict=True):
        for concept, value in mapping.items():
            FundamentalFact.objects.create(
                company=listing.security.company,
                provider="synthetic",
                concept=concept,
                source_concept=concept,
                value=Decimal(str(value)),
                unit="USD",
                currency="USD",
                period_end=period,
                fiscal_year=period.year,
                fiscal_period="FY",
                accession=f"{period.year}-{uuid4().hex[:8]}",
                available_at=available_at,
                source_asset=source_asset,
            )


def _create_extreme_facts(listing: Listing, source_asset: object, available_at) -> None:
    for concept, value in (
        ("total_debt", "1000000000000"),
        ("shareholders_equity", "1"),
        ("free_cash_flow", "-1000000000"),
        ("interest_expense", "-1000000000"),
    ):
        FundamentalFact.objects.create(
            company=listing.security.company,
            provider="synthetic",
            concept=concept,
            source_concept=concept,
            value=Decimal(value),
            unit="USD",
            currency="USD",
            period_end=date(2025, 12, 31),
            fiscal_year=2025,
            fiscal_period="FY",
            accession=f"extreme-{concept}-{uuid4().hex[:8]}",
            available_at=available_at,
            source_asset=source_asset,
        )
