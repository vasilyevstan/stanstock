from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import IntegrityError, transaction
from django.test import override_settings
from django.utils import timezone
from hypothesis import given, settings
from hypothesis import strategies as st

from stanstock.data.asof import AsOfData, PriceFrameChecksumMismatchError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.management.config_loader import default_us_scoring_config_path
from stanstock.data.models import (
    Company,
    DataAsset,
    FundamentalFact,
    Listing,
    ProviderRecord,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.research.config import V3_EFFECTIVE_CONFIG_HASH, load_scoring_config
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.reporting import (
    canonical_reportable_prediction_filter,
    reportable_prediction_filter,
)
from stanstock.research.service import (
    analyze_listing,
    analyze_snapshot,
    append_predictions,
    compute_listing_analysis,
)


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
    assert first.analysis.forecast_scenarios == {
        "schema_version": 1,
        "horizons": {
            horizon: scenario.as_dict() for horizon, scenario in first.computation.scenarios.items()
        },
    }
    assert all(
        prediction.evidence_role == Prediction.EvidenceRole.DECISION
        for prediction in first.predictions
    )
    assert all(
        prediction.evidence_grade == UniverseSnapshot.Grade.OBSERVED
        for prediction in first.predictions
    )
    assert all(
        prediction.source_mode == Prediction.SourceMode.SYNTHETIC
        for prediction in first.predictions
    )
    assert all(prediction.price_provider == "synthetic" for prediction in first.predictions)
    assert all(
        prediction.method_version == first.run.config_version for prediction in first.predictions
    )
    assert all(prediction.calculation["schema_version"] == 1 for prediction in first.predictions)
    assert {prediction.calculation["score_group"] for prediction in first.predictions} == {
        "short",
        "medium",
        "long",
    }


@pytest.mark.django_db
def test_etf_is_rejected_by_stock_analysis_and_prediction_services(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    listing.security.security_type = Security.SecurityType.ETF
    listing.security.save(update_fields=["security_type"])

    with pytest.raises(ValueError, match="exchange-traded fund"):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=timezone.now(),
            provider="synthetic",
            store=AssetStore(tmp_path),
        )
    with pytest.raises(ValueError, match="exchange-traded fund"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=timezone.now(),
            provider="synthetic",
            store=AssetStore(tmp_path),
        )
    assert AnalysisRun.objects.count() == 0

    listing.security.security_type = Security.SecurityType.COMMON_STOCK
    listing.security.save(update_fields=["security_type"])
    now = timezone.now()
    price_asset = _register_price_asset(
        AssetStore(tmp_path),
        listing.ticker,
        now - timedelta(minutes=5),
    )
    _create_facts(listing, price_asset, now - timedelta(minutes=4))
    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=AssetStore(tmp_path),
    )
    listing.security.security_type = Security.SecurityType.ETF
    listing.security.save(update_fields=["security_type"])

    with pytest.raises(ValueError, match="prediction issuance"):
        append_predictions(
            analysis=persisted.analysis,
            computation=persisted.computation,
            generated_at=now + timedelta(minutes=1),
            data_cutoff=persisted.run.data_cutoff,
            issued_on_time=False,
            supported_horizons=("short",),
            model_version="blocked-etf",
            config_hash_value=persisted.run.config_hash,
            source_assets=persisted.computation.source_assets,
            code_revision_value="test",
        )


@pytest.mark.django_db
def test_prediction_preserves_explicit_price_subject_override(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_subject = f"{listing.ticker}:US"
    price_asset = _register_price_asset(
        store,
        price_subject,
        now - timedelta(minutes=5),
    )
    _create_facts(listing, price_asset, now - timedelta(minutes=4))

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        subject=price_subject,
        store=store,
    )

    assert persisted.analysis.data_quality["price_source"] == {
        "asset_id": str(price_asset.id),
        "provider": "synthetic",
        "subject": price_subject,
    }
    assert all(
        prediction.price_provider == "synthetic"
        and prediction.price_subject == price_subject
        and prediction.calculation["price_subject"] == price_subject
        for prediction in persisted.predictions
    )


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
def test_v2_retains_convenience_price_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    calls: list[str] = []
    original = AsOfData.price_frame_with_diagnostics

    def convenience(self, *, provider, subject, through_date=None):
        calls.append(subject)
        return original(
            self,
            provider=provider,
            subject=subject,
            through_date=through_date,
        )

    monkeypatch.setattr(AsOfData, "price_frame_with_diagnostics", convenience)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        lambda *_args, **_kwargs: pytest.fail("v2 must retain its convenience reader"),
    )

    analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
        config_path=default_us_scoring_config_path(),
    )

    assert calls == [listing.ticker]


def _v3_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"


def _stored_file_paths(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


@pytest.mark.django_db
def test_v3_selects_and_exact_reads_each_requested_source_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    listing_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    benchmark_asset = _register_price_asset(store, "V3BENCH", now - timedelta(minutes=5))
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self, *, provider, kind, subject):
        selections.append(subject)
        return original_select(self, provider=provider, kind=kind, subject=subject)

    def exact_read(self, *, asset, through_date=None):
        reads.append(asset.subject)
        return original_read(self, asset=asset, through_date=through_date)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        exact_read,
    )
    monkeypatch.setattr(
        AsOfData,
        "price_frame_with_diagnostics",
        lambda *_args, **_kwargs: pytest.fail("v3 must not use the convenience reader"),
    )

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        benchmark_subject="V3BENCH",
        store=store,
        config_path=_v3_config_path(),
    )

    assert selections == [listing.ticker, "V3BENCH"]
    assert reads == [listing.ticker, "V3BENCH"]
    assert persisted.run.config_version == "us-price-baseline-v3"
    assert persisted.run.generated_at == now
    assert persisted.run.data_cutoff == now
    assert len(persisted.predictions) == 1
    assert persisted.predictions[0].horizon == Prediction.Horizon.SHORT
    assert {entry["id"] for entry in persisted.computation.source_assets} == {
        str(listing_asset.id),
        str(benchmark_asset.id),
    }
    assert {entry["sha256"] for entry in persisted.computation.source_assets} == {
        listing_asset.sha256,
        benchmark_asset.sha256,
    }
    assert len(persisted.analysis.component_scores["factors"]) == 16
    assert set(persisted.analysis.data_quality["factor_policy"]) == {
        "schema_version",
        "macd_indicator",
        "abnormal_volume_indicator",
        "liquidity_indicator",
        "strict_finite_inputs",
        "rsi",
        "risk_window",
        "beta_roles",
        "factor_maps",
        "risk_penalty_maps",
        "buy_min_liquidity_20d",
    }


@pytest.mark.django_db
def test_v3_omitted_benchmark_performs_zero_benchmark_source_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self, *, provider, kind, subject):
        selections.append(subject)
        return original_select(self, provider=provider, kind=kind, subject=subject)

    def exact_read(self, *, asset, through_date=None):
        reads.append(asset.subject)
        return original_read(self, asset=asset, through_date=through_date)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        exact_read,
    )

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        benchmark_subject=None,
        store=store,
        config_path=_v3_config_path(),
    )

    assert selections == [listing.ticker]
    assert reads == [listing.ticker]
    assert persisted.computation.risk_score is None
    assert persisted.computation.risk_class == RiskClass.INSUFFICIENT
    assert persisted.computation.scenarios["short"].bear is not None
    assert len(persisted.computation.source_assets) == 1
    assert "benchmark" in persisted.computation.indicators.missing["annualized_volatility"].lower()


@pytest.mark.django_db
def test_v3_requested_missing_benchmark_propagates_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self, *, provider, kind, subject):
        selections.append(subject)
        return original_select(self, provider=provider, kind=kind, subject=subject)

    def exact_read(self, *, asset, through_date=None):
        reads.append(asset.subject)
        return original_read(self, asset=asset, through_date=through_date)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        exact_read,
    )

    with pytest.raises(DataAsset.DoesNotExist):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=now,
            provider="synthetic",
            benchmark_subject="MISSING",
            store=store,
            config_path=_v3_config_path(),
        )

    assert selections == [listing.ticker, "MISSING"]
    assert reads == [listing.ticker]
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(kind="analysis_output_manifest").count() == 0


@pytest.mark.django_db
def test_v3_selected_corrupt_benchmark_attempts_exact_read_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    benchmark_asset = _register_price_asset(
        store,
        "CORRUPT",
        now - timedelta(minutes=5),
    )
    corrupt_frame = pl.DataFrame(
        {
            "date": [date(2025, 1, 1) + timedelta(days=index) for index in range(280)],
            "close": [200.0 + index * 0.1 for index in range(280)],
        }
    )
    corrupt_frame.write_parquet(store.resolve(benchmark_asset.relative_path))
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self, *, provider, kind, subject):
        selections.append(subject)
        return original_select(self, provider=provider, kind=kind, subject=subject)

    def exact_read(self, *, asset, through_date=None):
        reads.append(asset.subject)
        return original_read(self, asset=asset, through_date=through_date)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        exact_read,
    )

    with pytest.raises(PriceFrameChecksumMismatchError):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=now,
            provider="synthetic",
            benchmark_subject="CORRUPT",
            store=store,
            config_path=_v3_config_path(),
        )

    assert selections == [listing.ticker, "CORRUPT"]
    assert reads == [listing.ticker, "CORRUPT"]
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(pk=benchmark_asset.pk).exists()
    assert store.resolve(benchmark_asset.relative_path).exists()


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["listing", "snapshot"])
@pytest.mark.parametrize("source_state", ["missing", "corrupt"])
def test_v3_listing_source_failure_matrix_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    source_state: str,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    source_asset = None
    if source_state == "corrupt":
        source_asset = _register_price_asset(
            store,
            listing.ticker,
            now - timedelta(minutes=5),
        )
        pl.DataFrame(
            {
                "date": [date(2025, 1, 1) + timedelta(days=index) for index in range(280)],
                "close": [75.0 + index * 0.1 for index in range(280)],
            }
        ).write_parquet(store.resolve(source_asset.relative_path))
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self, *, provider, kind, subject):
        selections.append(subject)
        return original_select(self, provider=provider, kind=kind, subject=subject)

    def exact_read(self, *, asset, through_date=None):
        reads.append(asset.subject)
        return original_read(self, asset=asset, through_date=through_date)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        exact_read,
    )
    expected_error = (
        DataAsset.DoesNotExist if source_state == "missing" else PriceFrameChecksumMismatchError
    )

    with pytest.raises(expected_error):
        if entry_point == "listing":
            analyze_listing(
                listing=listing,
                universe_snapshot=snapshot,
                decision_time=now,
                provider="synthetic",
                store=store,
                config_path=_v3_config_path(),
            )
        else:
            analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=now,
                provider="synthetic",
                store=store,
                config_path=_v3_config_path(),
                long_forecast_requested=False,
            )

    assert selections == [listing.ticker]
    assert reads == ([] if source_state == "missing" else [listing.ticker])
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(kind="analysis_output_manifest").count() == 0
    if source_asset is not None:
        assert DataAsset.objects.filter(pk=source_asset.pk).exists()
        assert store.resolve(source_asset.relative_path).exists()


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["listing", "snapshot"])
@pytest.mark.parametrize("source_state", ["missing", "corrupt"])
def test_v3_requested_benchmark_failure_matrix_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    source_state: str,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    benchmark_asset = None
    if source_state == "corrupt":
        benchmark_asset = _register_price_asset(
            store,
            "MATRIXBENCH",
            now - timedelta(minutes=5),
        )
        pl.DataFrame(
            {
                "date": [date(2025, 1, 1) + timedelta(days=index) for index in range(280)],
                "close": [150.0 + index * 0.1 for index in range(280)],
            }
        ).write_parquet(store.resolve(benchmark_asset.relative_path))
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self, *, provider, kind, subject):
        selections.append(subject)
        return original_select(self, provider=provider, kind=kind, subject=subject)

    def exact_read(self, *, asset, through_date=None):
        reads.append(asset.subject)
        return original_read(self, asset=asset, through_date=through_date)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        exact_read,
    )
    expected_error = (
        DataAsset.DoesNotExist if source_state == "missing" else PriceFrameChecksumMismatchError
    )

    with pytest.raises(expected_error):
        kwargs = {
            "universe_snapshot": snapshot,
            "decision_time": now,
            "provider": "synthetic",
            "benchmark_subject": "MATRIXBENCH",
            "store": store,
            "config_path": _v3_config_path(),
        }
        if entry_point == "listing":
            analyze_listing(listing=listing, **kwargs)
        else:
            analyze_snapshot(long_forecast_requested=False, **kwargs)

    assert selections == [listing.ticker, "MATRIXBENCH"]
    assert reads == (
        [listing.ticker] if source_state == "missing" else [listing.ticker, "MATRIXBENCH"]
    )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(kind="analysis_output_manifest").count() == 0
    if benchmark_asset is not None:
        assert DataAsset.objects.filter(pk=benchmark_asset.pk).exists()
        assert store.resolve(benchmark_asset.relative_path).exists()


@pytest.mark.django_db
def test_v3_snapshot_repeats_benchmark_selection_and_exact_read_per_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing_one, snapshot = _listing_and_snapshot()
    listing_two = _add_listing_to_snapshot(snapshot)
    store = AssetStore(tmp_path)
    now = timezone.now()
    for listing in (listing_one, listing_two):
        _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _register_price_asset(store, "SNAPBENCH", now - timedelta(minutes=5))
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self, *, provider, kind, subject):
        selections.append(subject)
        return original_select(self, provider=provider, kind=kind, subject=subject)

    def exact_read(self, *, asset, through_date=None):
        reads.append(asset.subject)
        return original_read(self, asset=asset, through_date=through_date)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        exact_read,
    )

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        benchmark_subject="SNAPBENCH",
        store=store,
        config_path=_v3_config_path(),
        long_forecast_requested=False,
    )

    assert len(results) == 2
    assert selections.count("SNAPBENCH") == 2
    assert reads.count("SNAPBENCH") == 2
    assert len(selections) == len(reads) == 4
    assert Prediction.objects.count() == 2


@pytest.mark.django_db
def test_v3_refuses_non_usd_listing_before_source_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    listing.currency = "EUR"
    listing.save(update_fields=["currency"])
    calls = 0

    def select(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("non-USD v3 must fail before persisted-source selection")

    monkeypatch.setattr(AsOfData, "latest_asset", select)

    with pytest.raises(ValueError, match="USD listings only"):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=timezone.now(),
            provider="synthetic",
            store=AssetStore(tmp_path),
            config_path=_v3_config_path(),
        )

    assert calls == 0
    assert AnalysisRun.objects.count() == 0


@pytest.mark.django_db
def test_v3_direct_validation_normalizes_independent_row_permutations() -> None:
    listing, _snapshot = _listing_and_snapshot()
    config = load_scoring_config(_v3_config_path())
    rows = 300
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(rows)]
    frame = pl.DataFrame(
        {
            "date": dates,
            "close": [80.0 + index * 0.1 + (index % 7) * 0.03 for index in range(rows)],
            "volume": [1_000_000.0 + index * 100 for index in range(rows)],
        }
    )
    benchmark = pl.DataFrame(
        {
            "date": dates,
            "close": [100.0 + index * 0.06 + (index % 11) * 0.02 for index in range(rows)],
        }
    )
    decision_time = datetime(2026, 1, 1, tzinfo=UTC)

    canonical = compute_listing_analysis(
        listing=listing,
        price_frame=frame,
        benchmark_frame=benchmark,
        config=config,
        decision_time=decision_time,
        sample_support={"short": 100},
    )
    permutations = (
        (frame.reverse(), benchmark),
        (frame, benchmark.reverse()),
        (frame.reverse(), benchmark.reverse()),
    )

    for permuted_frame, permuted_benchmark in permutations:
        permuted = compute_listing_analysis(
            listing=listing,
            price_frame=permuted_frame,
            benchmark_frame=permuted_benchmark,
            config=config,
            decision_time=decision_time,
            sample_support={"short": 100},
        )
        assert permuted.indicators == canonical.indicators
        assert {key: scenario.as_dict() for key, scenario in permuted.scenarios.items()} == {
            key: scenario.as_dict() for key, scenario in canonical.scenarios.items()
        }
        assert permuted.aggregate == canonical.aggregate
        assert permuted.risk_score == canonical.risk_score
        assert permuted.recommendation == canonical.recommendation
        assert (
            permuted.data_quality["recommendation_gates"]
            == (canonical.data_quality["recommendation_gates"])
        )


@pytest.mark.django_db
def test_v3_passes_the_same_normalized_asset_frame_to_indicators_and_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    listing, _snapshot = _listing_and_snapshot()
    config = load_scoring_config(_v3_config_path())
    frame, benchmark = _scale_fixture(base_price=30.0, volume=1_000_000.0)
    frame = frame.reverse()
    seen: list[pl.DataFrame] = []
    original_indicators = service_module.calculate_indicators
    original_scenarios = service_module.build_scenarios

    def indicators(asset_frame, **kwargs):
        seen.append(asset_frame)
        return original_indicators(asset_frame, **kwargs)

    def scenarios(asset_frame, *args, **kwargs):
        seen.append(asset_frame)
        return original_scenarios(asset_frame, *args, **kwargs)

    monkeypatch.setattr(service_module, "calculate_indicators", indicators)
    monkeypatch.setattr(service_module, "build_scenarios", scenarios)

    compute_listing_analysis(
        listing=listing,
        price_frame=frame,
        benchmark_frame=benchmark,
        config=config,
        decision_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(seen) == 2
    assert seen[0] is seen[1]
    assert seen[0]["date"].is_sorted()


def _scale_fixture(
    *,
    base_price: float,
    volume: float,
    rows: int = 300,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(rows)]
    closes = [
        base_price * (1.0 + index * 0.0007 + ((index % 9) - 4) * 0.0008) for index in range(rows)
    ]
    asset = pl.DataFrame(
        {
            "date": dates,
            "open": [close * 0.997 for close in closes],
            "high": [close * 1.008 for close in closes],
            "low": [close * 0.992 for close in closes],
            "close": closes,
            "volume": [volume * (1.0 + (index % 7) * 0.01) for index in range(rows)],
        }
    )
    benchmark = pl.DataFrame(
        {
            "date": dates,
            "close": [
                100.0 * (1.0 + index * 0.0004 + ((index % 11) - 5) * 0.0005)
                for index in range(rows)
            ],
        }
    )
    return asset, benchmark


def _assert_v3_normalized_outputs_equal(left, right) -> None:
    for key in (
        "return_20d",
        "return_63d",
        "return_126d",
        "close_vs_sma_50",
        "close_vs_sma_200",
        "rsi_14",
        "macd_histogram_pct",
        "52w_position",
        "abnormal_volume_strict",
        "avg_dollar_volume_20d",
        "annualized_volatility",
        "downside_volatility",
        "max_drawdown",
        "beta",
        "relative_return_20d",
        "relative_return_63d",
        "relative_return_252d",
    ):
        assert right.indicators.values[key] == pytest.approx(left.indicators.values[key])
    assert right.indicators.missing == left.indicators.missing
    assert right.aggregate.component_scores.factor_scores == pytest.approx(
        left.aggregate.component_scores.factor_scores
    )
    assert right.aggregate.component_scores.components == pytest.approx(
        left.aggregate.component_scores.components
    )
    assert right.aggregate.component_scores.coverage == left.aggregate.component_scores.coverage
    assert right.aggregate.horizon_scores == pytest.approx(left.aggregate.horizon_scores)
    assert right.aggregate.overall == pytest.approx(left.aggregate.overall)
    assert right.aggregate.confidence == pytest.approx(left.aggregate.confidence)
    assert right.aggregate.missingness_penalty == pytest.approx(left.aggregate.missingness_penalty)
    assert right.aggregate.freshness_penalty == pytest.approx(left.aggregate.freshness_penalty)
    assert right.risk_score == pytest.approx(left.risk_score)
    assert right.risk_class == left.risk_class
    assert right.recommendation == left.recommendation
    assert right.data_quality["recommendation_gates"] == left.data_quality["recommendation_gates"]
    assert right.reasons == left.reasons
    assert right.risks == left.risks
    for horizon in ("short", "medium", "long"):
        left_scenario = left.scenarios[horizon]
        right_scenario = right.scenarios[horizon]
        for field in ("bear", "base", "bull", "probability_positive", "confidence"):
            left_value = getattr(left_scenario, field)
            right_value = getattr(right_scenario, field)
            if left_value is None:
                assert right_value is None
            else:
                assert right_value == pytest.approx(left_value)
        assert right_scenario.confidence_status == left_scenario.confidence_status
        assert right_scenario.insufficiency_reason == left_scenario.insufficiency_reason
        assert right_scenario.method == left_scenario.method


@pytest.mark.django_db
@settings(max_examples=8, deadline=None)
@given(
    k=st.floats(min_value=0.2, max_value=12.0, allow_nan=False, allow_infinity=False),
    base_price=st.floats(
        min_value=5.0,
        max_value=500.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    volume=st.floats(
        min_value=100_000.0,
        max_value=10_000_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_v3_compatible_price_volume_transform_preserves_normalized_decision(
    k: float,
    base_price: float,
    volume: float,
) -> None:
    listing, _snapshot = _listing_and_snapshot()
    config = load_scoring_config(_v3_config_path())
    frame, benchmark = _scale_fixture(base_price=base_price, volume=volume)
    transformed = frame.with_columns(
        *((pl.col(column) * k).alias(column) for column in ("open", "high", "low", "close")),
        (pl.col("volume") / k).alias("volume"),
    )
    kwargs = {
        "listing": listing,
        "benchmark_frame": benchmark,
        "config": config,
        "decision_time": datetime(2026, 1, 1, tzinfo=UTC),
        "sample_support": {"short": 100},
    }

    baseline = compute_listing_analysis(price_frame=frame, **kwargs)
    scaled = compute_listing_analysis(price_frame=transformed, **kwargs)

    _assert_v3_normalized_outputs_equal(baseline, scaled)
    for key in (
        "last_close",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "ema_12",
        "ema_20",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_14",
    ):
        assert scaled.indicators.values[key] == pytest.approx(baseline.indicators.values[key] * k)
    assert scaled.indicators.values["avg_volume_20d"] == pytest.approx(
        baseline.indicators.values["avg_volume_20d"] / k
    )
    assert scaled.current_price == pytest.approx(baseline.current_price * k)


@pytest.mark.django_db
@settings(max_examples=8, deadline=None)
@given(
    k=st.floats(
        min_value=1.01,
        max_value=24.99,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_v3_fixed_volume_price_scale_changes_dollar_turnover_by_k(k: float) -> None:
    listing, _snapshot = _listing_and_snapshot()
    config = load_scoring_config(_v3_config_path())
    provisional, benchmark = _scale_fixture(base_price=40.0, volume=1.0)
    target_turnover = 5_000_000.0 / (k**0.5)
    volume = target_turnover / float(provisional["close"].tail(20).mean())
    frame = provisional.with_columns(pl.lit(volume).alias("volume"))
    transformed = frame.with_columns(
        *((pl.col(column) * k).alias(column) for column in ("open", "high", "low", "close"))
    )
    kwargs = {
        "listing": listing,
        "benchmark_frame": benchmark,
        "config": config,
        "decision_time": datetime(2026, 1, 1, tzinfo=UTC),
        "sample_support": {"short": 100},
    }

    baseline = compute_listing_analysis(price_frame=frame, **kwargs)
    scaled = compute_listing_analysis(price_frame=transformed, **kwargs)

    assert baseline.indicators.values["avg_dollar_volume_20d"] == pytest.approx(target_turnover)
    assert scaled.indicators.values["avg_dollar_volume_20d"] == pytest.approx(target_turnover * k)
    assert target_turnover < 5_000_000 < target_turnover * k
    assert (
        scaled.aggregate.component_scores.factor_scores["risk.avg_volume"]
        > (baseline.aggregate.component_scores.factor_scores["risk.avg_volume"])
    )
    assert (
        scaled.aggregate.component_scores.components["risk_liquidity"]
        > (baseline.aggregate.component_scores.components["risk_liquidity"])
    )
    assert scaled.aggregate.overall > baseline.aggregate.overall
    assert baseline.data_quality["recommendation_gates"]["buy_liquidity"] is False
    assert scaled.data_quality["recommendation_gates"]["buy_liquidity"] is True
    for key in (
        "return_20d",
        "rsi_14",
        "macd_histogram_pct",
        "annualized_volatility",
        "downside_volatility",
        "max_drawdown",
        "beta",
    ):
        assert scaled.indicators.values[key] == pytest.approx(baseline.indicators.values[key])
    assert scaled.risk_score == pytest.approx(baseline.risk_score)
    assert scaled.aggregate.confidence == baseline.aggregate.confidence
    assert scaled.aggregate.component_scores.coverage == (
        baseline.aggregate.component_scores.coverage
    )
    for horizon in ("short", "medium", "long"):
        assert scaled.scenarios[horizon].as_dict() == pytest.approx(
            baseline.scenarios[horizon].as_dict()
        )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("date", None, "null dates"),
        ("close", None, "finite and positive"),
        ("close", 0.0, "finite and positive"),
        ("close", float("nan"), "finite and positive"),
        ("volume", -1.0, "finite and nonnegative"),
        ("volume", float("inf"), "finite and nonnegative"),
    ],
)
def test_v3_direct_validation_rejects_malformed_rows_before_lossy_preparation(
    column: str,
    value,
    message: str,
) -> None:
    listing, _snapshot = _listing_and_snapshot()
    config = load_scoring_config(_v3_config_path())
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 2)],
            "close": [10.0, 11.0],
            "volume": [100.0, 101.0],
        }
    ).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(ValueError, match=message):
        compute_listing_analysis(
            listing=listing,
            price_frame=frame,
            config=config,
            decision_time=datetime(2026, 1, 2, tzinfo=UTC),
        )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pl.DataFrame(
                {
                    "date": [date(2026, 1, 1), date(2026, 1, 1)],
                    "close": [10.0, 11.0],
                }
            ),
            "duplicate normalized dates",
        ),
        (
            pl.DataFrame({"date": [1, 2], "close": [10.0, 11.0]}),
            "unsupported date dtype",
        ),
        (
            pl.DataFrame(
                {
                    "date": ["2026-1-01", "2026-01-02"],
                    "close": [10.0, 11.0],
                }
            ),
            "strict ISO",
        ),
        (
            pl.DataFrame(
                {
                    "date": [date(2026, 1, 1), date(2026, 1, 2)],
                    "close": [1e308, 1e308],
                    "volume": [1e308, 1e308],
                }
            ),
            "products must be finite",
        ),
    ],
    ids=["duplicate-date", "unsupported-date", "non-iso-date", "overflow-product"],
)
def test_v3_direct_validation_rejects_ambiguous_identity_and_products(
    frame: pl.DataFrame,
    message: str,
) -> None:
    listing, _snapshot = _listing_and_snapshot()

    with pytest.raises(ValueError, match=message):
        compute_listing_analysis(
            listing=listing,
            price_frame=frame,
            config=load_scoring_config(_v3_config_path()),
            decision_time=datetime(2026, 1, 2, tzinfo=UTC),
        )


@pytest.mark.django_db
def test_v3_preserves_reported_zero_volume_as_numeric_zero() -> None:
    listing, _snapshot = _listing_and_snapshot()
    frame, benchmark = _scale_fixture(base_price=20.0, volume=0.0)

    result = compute_listing_analysis(
        listing=listing,
        price_frame=frame,
        benchmark_frame=benchmark,
        config=load_scoring_config(_v3_config_path()),
        decision_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.indicators.values["avg_volume_20d"] == 0.0
    assert result.indicators.values["avg_dollar_volume_20d"] == 0.0
    assert result.indicators.values["abnormal_volume_strict"] == 0.0
    assert result.aggregate.component_scores.factor_scores["risk.avg_volume"] == 0.0
    assert result.aggregate.component_scores.factor_scores["risk.abnormal_volume"] == 0.0
    assert result.data_quality["recommendation_gates"]["buy_liquidity_present"] is True
    assert result.data_quality["recommendation_gates"]["buy_liquidity"] is False


@pytest.mark.django_db
def test_v3_asof_clipping_makes_post_target_rows_irrelevant(tmp_path: Path) -> None:
    target = date(2026, 1, 31)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    eligible_dates = [target - timedelta(days=299 - index) for index in range(300)]
    future_dates = [target + timedelta(days=index) for index in range(1, 6)]
    asset_closes = [50.0 + index * 0.05 + (index % 7) * 0.02 for index in range(300)]
    benchmark_closes = [100.0 + index * 0.03 + (index % 11) * 0.01 for index in range(300)]
    available_at = datetime(2026, 2, 5, tzinfo=UTC)
    _register_explicit_price_asset(
        store,
        listing.ticker,
        available_at,
        eligible_dates + future_dates,
        asset_closes + [10_000.0] * len(future_dates),
    )
    _register_explicit_price_asset(
        store,
        "CLIPBENCH",
        available_at,
        eligible_dates + future_dates,
        benchmark_closes + [1.0] * len(future_dates),
    )

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=datetime(2026, 2, 6, tzinfo=UTC),
        target_date=target,
        issued_on_time=False,
        provider="synthetic",
        benchmark_subject="CLIPBENCH",
        store=store,
        config_path=_v3_config_path(),
    )
    direct = compute_listing_analysis(
        listing=listing,
        price_frame=pl.DataFrame(
            {
                "date": eligible_dates,
                "open": [close - 0.1 for close in asset_closes],
                "high": [close + 0.8 for close in asset_closes],
                "low": [close - 0.8 for close in asset_closes],
                "close": asset_closes,
                "volume": [500_000 + index * 1_000 for index in range(300)],
            }
        ),
        benchmark_frame=pl.DataFrame({"date": eligible_dates, "close": benchmark_closes}),
        config=load_scoring_config(_v3_config_path()),
        decision_time=datetime.combine(target, datetime.max.time(), tzinfo=UTC),
    )

    assert persisted.computation.indicators.last_date == target
    assert persisted.computation.indicators.values == pytest.approx(direct.indicators.values)
    assert persisted.computation.aggregate.overall == pytest.approx(direct.aggregate.overall)
    assert persisted.computation.risk_score == pytest.approx(direct.risk_score)
    assert persisted.computation.recommendation == direct.recommendation


@pytest.mark.django_db
def test_v3_analyze_listing_omitted_flag_is_non_on_time_and_non_reportable(
    tmp_path: Path,
) -> None:
    target = date(2026, 9, 4)
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        generated_at - timedelta(minutes=5),
        provider="twelve_data",
    )

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=generated_at,
        provider="twelve_data",
        store=store,
        config_path=_v3_config_path(),
    )

    prediction = persisted.predictions[0]
    assert snapshot.grade == UniverseSnapshot.Grade.OBSERVED
    assert persisted.run.issued_on_time is False
    assert prediction.issued_on_time is False
    assert prediction.source_mode == Prediction.SourceMode.PROVIDER
    assert not Prediction.objects.filter(reportable_prediction_filter()).exists()
    assert not Prediction.objects.filter(canonical_reportable_prediction_filter()).exists()


@pytest.mark.django_db
def test_v3_analyze_listing_explicit_true_raises_without_output(tmp_path: Path) -> None:
    target = date(2026, 9, 4)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)

    with pytest.raises(ValueError, match="on-time issuance requires analyze_snapshot"):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=datetime(2026, 9, 4, 21, tzinfo=UTC),
            issued_on_time=True,
            provider="twelve_data",
            store=AssetStore(tmp_path),
            config_path=_v3_config_path(),
        )

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(kind="analysis_output_manifest").count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize("issued_on_time", [None, False], ids=("omitted", "explicit-false"))
def test_v3_analyze_snapshot_research_flags_remain_unbound_and_non_on_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    issued_on_time: bool | None,
) -> None:
    target = date(2026, 9, 4)
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    _register_price_asset(store, listing.ticker, generated_at - timedelta(minutes=5))
    monkeypatch.delenv("STANSTOCK_CODE_REVISION", raising=False)
    monkeypatch.setattr(
        "stanstock.research.service.clean_git_revision",
        lambda *_args, **_kwargs: pytest.fail("research v3 must not validate a clean revision"),
    )

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=generated_at,
        issued_on_time=issued_on_time,
        provider="synthetic",
        store=store,
        config_path=_v3_config_path(),
        long_forecast_requested=False,
    )

    assert len(results) == 1
    assert results[0].run.issued_on_time is False
    assert all(not prediction.issued_on_time for prediction in results[0].predictions)


@pytest.mark.django_db
def test_v3_analyze_snapshot_explicit_true_can_be_on_time_when_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = date(2026, 9, 4)
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    for subject in (listing.ticker, "SPY"):
        _register_price_asset(
            store,
            subject,
            generated_at - timedelta(minutes=5),
            provider="twelve_data",
        )
    revision = "d6374eb8a25361eb813e1ca79086a696588e1585"
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", revision)
    monkeypatch.setattr("stanstock.research.service.clean_git_revision", lambda _root: revision)

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=generated_at,
        target_date=target,
        issued_on_time=True,
        provider="twelve_data",
        benchmark_subject="SPY",
        store=store,
        config_path=_v3_config_path(),
        long_forecast_requested=False,
    )

    assert len(results) == 1
    persisted = results[0]
    assert persisted.run.issued_on_time is True
    assert persisted.run.config_version == "us-price-baseline-v3"
    assert persisted.run.config_hash == V3_EFFECTIVE_CONFIG_HASH
    assert persisted.run.code_revision == revision
    assert persisted.run.universe_snapshot.grade == UniverseSnapshot.Grade.OBSERVED
    assert len(persisted.predictions) == 1
    prediction = persisted.predictions[0]
    assert prediction.horizon == Prediction.Horizon.SHORT
    assert prediction.issued_on_time is True
    assert prediction.config_hash == V3_EFFECTIVE_CONFIG_HASH
    assert prediction.code_revision == revision
    assert prediction.price_provider == "twelve_data"
    assert prediction.price_subject == listing.ticker
    assert prediction.evidence_grade == UniverseSnapshot.Grade.OBSERVED
    assert prediction.source_mode == Prediction.SourceMode.PROVIDER
    assert {asset["subject"] for asset in prediction.source_assets} == {
        listing.ticker,
        "SPY",
    }
    assert {(asset["provider"], asset["subject"]) for asset in prediction.source_assets} == {
        ("twelve_data", listing.ticker),
        ("twelve_data", "SPY"),
    }
    manifest = DataAsset.objects.get(
        kind="analysis_output_manifest",
        subject=str(persisted.run.id),
    )
    assert manifest.provider == "stanstock"
    assert manifest.sha256
    assert store.resolve(manifest.relative_path).is_file()
    assert set(
        Prediction.objects.filter(reportable_prediction_filter()).values_list("id", flat=True)
    ) == {prediction.id}
    assert set(
        Prediction.objects.filter(canonical_reportable_prediction_filter()).values_list(
            "id", flat=True
        )
    ) == {prediction.id}


@pytest.mark.django_db
@pytest.mark.parametrize(
    (
        "provider",
        "benchmark_subject",
        "raw_revision",
        "clean_result",
        "expected_message",
    ),
    (
        (
            "synthetic",
            "SPY",
            "1" * 40,
            "1" * 40,
            "Observed v3 issuance requires the Twelve Data provider",
        ),
        (
            "twelve_data",
            None,
            "1" * 40,
            "1" * 40,
            "Observed v3 issuance requires the SPY benchmark subject",
        ),
        (
            "twelve_data",
            "QQQ",
            "1" * 40,
            "1" * 40,
            "Observed v3 issuance requires the SPY benchmark subject",
        ),
        (
            "twelve_data",
            "SPY",
            None,
            "1" * 40,
            "Observed v3 issuance requires STANSTOCK_CODE_REVISION",
        ),
        (
            "twelve_data",
            "SPY",
            "",
            "1" * 40,
            "Observed v3 issuance requires a full lowercase 40-hex STANSTOCK_CODE_REVISION",
        ),
        (
            "twelve_data",
            "SPY",
            "working-tree",
            "1" * 40,
            "Observed v3 issuance requires a full lowercase 40-hex STANSTOCK_CODE_REVISION",
        ),
        (
            "twelve_data",
            "SPY",
            "A" * 40,
            "a" * 40,
            "Observed v3 issuance requires a full lowercase 40-hex STANSTOCK_CODE_REVISION",
        ),
        (
            "twelve_data",
            "SPY",
            "1" * 39,
            "1" * 39,
            "Observed v3 issuance requires a full lowercase 40-hex STANSTOCK_CODE_REVISION",
        ),
        (
            "twelve_data",
            "SPY",
            "1" * 40,
            ValueError("dirty checkout"),
            "Observed v3 issuance requires a verifiably clean committed Git revision",
        ),
        (
            "twelve_data",
            "SPY",
            "1" * 40,
            ValueError("Git revision validation failed"),
            "Observed v3 issuance requires a verifiably clean committed Git revision",
        ),
        (
            "twelve_data",
            "SPY",
            "1" * 40,
            "2" * 40,
            "Observed v3 issuance revision does not match the clean committed Git HEAD",
        ),
    ),
    ids=(
        "wrong-provider",
        "missing-benchmark",
        "wrong-benchmark",
        "revision-unset",
        "revision-blank",
        "revision-working-tree",
        "revision-uppercase",
        "revision-wrong-length",
        "dirty-checkout",
        "clean-revision-error",
        "revision-clean-head-mismatch",
    ),
)
def test_observed_v3_binding_failures_are_pre_source_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    benchmark_subject: str | None,
    raw_revision: str | None,
    clean_result: str | ValueError,
    expected_message: str,
) -> None:
    import stanstock.research.service as service_module

    target = date(2026, 9, 4)
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    source_assets = [
        _register_price_asset(
            store,
            subject,
            generated_at - timedelta(minutes=5),
            provider="twelve_data",
        )
        for subject in (listing.ticker, "SPY")
    ]
    source_ids = {asset.id for asset in source_assets}
    files_before = _stored_file_paths(tmp_path)
    reportable_before = Prediction.objects.filter(reportable_prediction_filter()).count()
    canonical_before = Prediction.objects.filter(canonical_reportable_prediction_filter()).count()
    if raw_revision is None:
        monkeypatch.delenv("STANSTOCK_CODE_REVISION", raising=False)
    else:
        monkeypatch.setenv("STANSTOCK_CODE_REVISION", raw_revision)

    clean_calls = 0

    def clean_revision(_root):
        nonlocal clean_calls
        clean_calls += 1
        if isinstance(clean_result, ValueError):
            raise clean_result
        return clean_result

    source_operations: list[str] = []

    def fail_selection(*_args, **_kwargs):
        source_operations.append("selection")
        pytest.fail("invalid observed-v3 binding must fail before source selection")

    def fail_read(*_args, **_kwargs):
        source_operations.append("read")
        pytest.fail("invalid observed-v3 binding must fail before source reads")

    monkeypatch.setattr(service_module, "clean_git_revision", clean_revision)
    monkeypatch.setattr(AsOfData, "latest_asset", fail_selection)
    monkeypatch.setattr(AsOfData, "price_frame_for_asset_with_diagnostics", fail_read)
    monkeypatch.setattr(
        service_module,
        "_resolve_provider_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid observed-v3 binding must fail before provider-plan resolution"
        ),
    )

    with pytest.raises(ValueError) as error:
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=generated_at,
            target_date=target,
            issued_on_time=True,
            provider=provider,
            benchmark_subject=benchmark_subject,
            store=store,
            config_path=_v3_config_path(),
            long_forecast_requested=False,
        )

    assert str(error.value) == expected_message
    assert str(tmp_path) not in str(error.value)
    assert source_operations == []
    expected_clean_calls = int(
        expected_message
        in {
            "Observed v3 issuance requires a verifiably clean committed Git revision",
            "Observed v3 issuance revision does not match the clean committed Git HEAD",
        }
    )
    assert clean_calls == expected_clean_calls
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert (
        DataAsset.objects.filter(
            kind__in=("analysis_output_manifest", "medium_forecast_panel")
        ).count()
        == 0
    )
    assert set(DataAsset.objects.values_list("id", flat=True)) == source_ids
    assert _stored_file_paths(tmp_path) == files_before
    assert Prediction.objects.filter(reportable_prediction_filter()).count() == reportable_before
    assert (
        Prediction.objects.filter(canonical_reportable_prediction_filter()).count()
        == canonical_before
    )


@pytest.mark.django_db
def test_observed_v3_wrong_provider_rejects_before_default_provider_or_output_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    target = date(2026, 9, 4)
    _listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    provider_operations: list[str] = []
    source_operations: list[str] = []
    output_operations: list[str] = []

    def fail_provider_query(*_args, **_kwargs):
        provider_operations.append("filter")
        pytest.fail("observed-v3 admission must precede mutable provider state")

    def fail_source_selection(*_args, **_kwargs):
        source_operations.append("selection")
        pytest.fail("observed-v3 admission must precede source selection")

    def fail_source_read(*_args, **_kwargs):
        source_operations.append("read")
        pytest.fail("observed-v3 admission must precede source reads")

    def fail_output(*_args, **_kwargs):
        output_operations.append("output")
        pytest.fail("observed-v3 admission must precede output operations")

    monkeypatch.setattr(ProviderRecord.objects, "filter", fail_provider_query)
    monkeypatch.setattr(AsOfData, "latest_asset", fail_source_selection)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        fail_source_read,
    )
    for output_name in (
        "_create_analysis_run",
        "build_medium_forecast_panel",
        "_persist_listing_analysis",
        "_finalize_observed_manifest",
    ):
        monkeypatch.setattr(service_module, output_name, fail_output)

    with pytest.raises(ValueError) as error:
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=datetime(2026, 9, 4, 21, tzinfo=UTC),
            target_date=target,
            issued_on_time=True,
            provider="synthetic",
            benchmark_subject="SPY",
            store=store,
            config_path=_v3_config_path(),
        )

    assert str(error.value) == "Observed v3 issuance requires the Twelve Data provider"
    assert str(tmp_path) not in str(error.value)
    assert provider_operations == []
    assert source_operations == []
    assert output_operations == []
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.count() == 0
    assert _stored_file_paths(tmp_path) == set()
    assert not Prediction.objects.filter(reportable_prediction_filter()).exists()
    assert not Prediction.objects.filter(canonical_reportable_prediction_filter()).exists()


@pytest.mark.django_db
def test_observed_v3_rejects_schema_valid_altered_config_before_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    target = date(2026, 9, 4)
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    source_assets = [
        _register_price_asset(
            store,
            subject,
            generated_at - timedelta(minutes=5),
            provider="twelve_data",
        )
        for subject in (listing.ticker, "SPY")
    ]
    altered_path = tmp_path / "altered-v3.yml"
    canonical_text = _v3_config_path().read_text(encoding="utf-8")
    altered_text = canonical_text.replace("  buy_min_score: 72\n", "  buy_min_score: 71\n", 1)
    assert altered_text != canonical_text
    altered_path.write_text(altered_text, encoding="utf-8")
    files_before = _stored_file_paths(tmp_path)
    source_ids = {asset.id for asset in source_assets}
    revision = "1" * 40
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", revision)
    clean_calls = 0

    def clean_revision(_root):
        nonlocal clean_calls
        clean_calls += 1
        return revision

    monkeypatch.setattr(service_module, "clean_git_revision", clean_revision)
    monkeypatch.setattr(
        AsOfData,
        "latest_asset",
        lambda *_args, **_kwargs: pytest.fail(
            "altered observed-v3 config must fail before source selection"
        ),
    )
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        lambda *_args, **_kwargs: pytest.fail(
            "altered observed-v3 config must fail before source reads"
        ),
    )

    with pytest.raises(
        ValueError,
        match="^Observed v3 issuance requires the exact reviewed v3 scoring config$",
    ):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=generated_at,
            target_date=target,
            issued_on_time=True,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=store,
            config_path=altered_path,
            long_forecast_requested=False,
        )

    assert clean_calls == 0
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert (
        DataAsset.objects.filter(
            kind__in=("analysis_output_manifest", "medium_forecast_panel")
        ).count()
        == 0
    )
    assert set(DataAsset.objects.values_list("id", flat=True)) == source_ids
    assert _stored_file_paths(tmp_path) == files_before
    assert not Prediction.objects.filter(reportable_prediction_filter()).exists()
    assert not Prediction.objects.filter(canonical_reportable_prediction_filter()).exists()


@pytest.mark.django_db
def test_v3_analyze_snapshot_explicit_true_rejects_after_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = date(2026, 9, 4)
    _listing, snapshot = _listing_and_snapshot(as_of_date=target)
    revision = "1" * 40
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", revision)
    monkeypatch.setattr("stanstock.research.service.clean_git_revision", lambda _root: revision)

    with pytest.raises(ValueError, match="after the next market session opened"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=datetime(2026, 9, 8, 14, tzinfo=UTC),
            target_date=target,
            issued_on_time=True,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=AssetStore(tmp_path),
            config_path=_v3_config_path(),
            long_forecast_requested=False,
        )

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
def test_v3_analyze_snapshot_explicit_true_rejects_cutoff_unsafe_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    target = date(2026, 9, 4)
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    for subject in (listing.ticker, "SPY"):
        _register_price_asset(
            store,
            subject,
            generated_at - timedelta(minutes=5),
            provider="twelve_data",
        )
    revision = "1" * 40
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", revision)
    monkeypatch.setattr(service_module, "clean_git_revision", lambda _root: revision)
    original_asset_payload = service_module._asset_payload

    def cutoff_unsafe_payload(asset):
        payload = original_asset_payload(asset)
        payload["available_at"] = (generated_at + timedelta(seconds=1)).isoformat()
        return payload

    monkeypatch.setattr(service_module, "_asset_payload", cutoff_unsafe_payload)

    with pytest.raises(ValueError, match="available_at after data cutoff"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=generated_at,
            target_date=target,
            issued_on_time=True,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=store,
            config_path=_v3_config_path(),
            long_forecast_requested=False,
        )

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(kind="analysis_output_manifest").count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    "config_name",
    ("us-price-baseline-v1.yml", "us-price-baseline-v2.yml"),
    ids=("v1", "v2"),
)
def test_v1_v2_snapshot_explicit_true_behavior_does_not_use_v3_binding_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
) -> None:
    target = date(2026, 9, 4)
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(as_of_date=target)
    store = AssetStore(tmp_path)
    _register_price_asset(store, listing.ticker, generated_at - timedelta(minutes=5))
    monkeypatch.delenv("STANSTOCK_CODE_REVISION", raising=False)
    monkeypatch.setattr(
        "stanstock.research.service.clean_git_revision",
        lambda *_args, **_kwargs: pytest.fail("v1/v2 must not use the observed-v3 binding gate"),
    )

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=generated_at,
        target_date=target,
        issued_on_time=True,
        provider="synthetic",
        store=store,
        config_path=Path(__file__).resolve().parents[1] / "config/scoring" / config_name,
        long_forecast_requested=False,
    )

    assert len(results) == 1
    assert results[0].run.issued_on_time is True
    assert all(prediction.issued_on_time for prediction in results[0].predictions)


@pytest.mark.django_db
def test_analyze_command_v3_snapshot_remains_explicitly_non_observed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    generated_at = timezone.now()
    _register_price_asset(store, listing.ticker, generated_at - timedelta(minutes=5))
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "working-tree")
    monkeypatch.setattr(
        "stanstock.research.service.clean_git_revision",
        lambda *_args, **_kwargs: pytest.fail(
            "manage.py analyze v3 must not request observed issuance"
        ),
    )
    stdout = StringIO()

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "analyze",
            snapshot=str(snapshot.pk),
            provider="synthetic",
            config=_v3_config_path(),
            stdout=stdout,
        )

    run = AnalysisRun.objects.get()
    prediction = Prediction.objects.get()
    assert run.config_version == "us-price-baseline-v3"
    assert run.issued_on_time is False
    assert prediction.issued_on_time is False
    assert "research-grade" in stdout.getvalue()
    assert "issued_on_time=False" in stdout.getvalue()


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
    forecast_scenarios = persisted.analysis.forecast_scenarios
    forecast_scenarios["horizons"]["short"] = {"base": 0.01, "bull": 0.02}
    persisted.analysis.forecast_scenarios = forecast_scenarios
    persisted.analysis.save(update_fields=["forecast_scenarios"])

    with pytest.raises(CommandError, match="scenario payload is incomplete"):
        call_command(
            "predict",
            analysis=str(persisted.analysis.pk),
            model_version="manual-v3",
            stdout=StringIO(),
        )

    assert Prediction.objects.filter(model_version="manual-v3").count() == 0


@pytest.mark.django_db
def test_decision_prediction_service_rejects_advisory_horizons(tmp_path) -> None:
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

    with pytest.raises(ValueError, match="scoring-group horizons only"):
        append_predictions(
            analysis=persisted.analysis,
            computation=persisted.computation,
            generated_at=now + timedelta(minutes=1),
            data_cutoff=persisted.run.data_cutoff,
            issued_on_time=False,
            supported_horizons=(Prediction.Horizon.SIX_MONTH.value,),
            model_version="invalid-advisory-decision",
            config_hash_value=persisted.run.config_hash,
            source_assets=persisted.computation.source_assets,
            code_revision_value="test",
        )


@pytest.mark.django_db
def test_analyze_listing_accepts_valid_explicit_issued_on_time_true(tmp_path) -> None:
    """Direct/live explicit `issued_on_time=True` behavior stays unchanged: a
    same-day, observed-snapshot call still succeeds and marks the run and
    every appended prediction on-time."""
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        issued_on_time=True,
        provider="synthetic",
        store=store,
    )

    assert persisted.run.issued_on_time is True
    assert all(prediction.issued_on_time for prediction in persisted.predictions)


@pytest.mark.django_db
def test_analyze_command_listing_path_forces_research_grade_on_same_day_observed_snapshot(
    tmp_path,
) -> None:
    """`manage.py analyze --listing ...` explicitly requests
    `issued_on_time=False` even when the snapshot is OBSERVED-grade and
    generated the same calendar day as its target -- a combination that would
    otherwise auto-infer an on-time (observed) issuance. Evidence grade still
    follows the snapshot; the issuance flags are the exclusion guard, not a
    relabeled evidence grade."""
    listing, snapshot = _listing_and_snapshot()
    assert snapshot.grade == UniverseSnapshot.Grade.OBSERVED
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))
    stdout = StringIO()

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "analyze",
            snapshot=str(snapshot.pk),
            listing=str(listing.pk),
            provider="synthetic",
            stdout=stdout,
        )

    run = AnalysisRun.objects.get()
    assert run.issued_on_time is False
    predictions = Prediction.objects.filter(analysis__run=run)
    assert predictions.exists()
    assert all(not prediction.issued_on_time for prediction in predictions)
    output = stdout.getvalue()
    assert "research-grade" in output
    assert "issued_on_time=False" in output


@pytest.mark.django_db
def test_analyze_command_snapshot_path_forces_research_grade_on_same_day_observed_snapshot(
    tmp_path,
) -> None:
    """The snapshot-wide `manage.py analyze` path (no `--listing`) forces the
    same research-grade `issued_on_time=False` guarantee."""
    listing, snapshot = _listing_and_snapshot()
    assert snapshot.grade == UniverseSnapshot.Grade.OBSERVED
    store = AssetStore(tmp_path)
    now = timezone.now()
    price_asset = _register_price_asset(store, listing.ticker, now - timedelta(minutes=5))
    _create_facts(listing, price_asset, now - timedelta(minutes=4))
    stdout = StringIO()

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "analyze",
            snapshot=str(snapshot.pk),
            provider="synthetic",
            stdout=stdout,
        )

    run = AnalysisRun.objects.get()
    assert run.issued_on_time is False
    predictions = Prediction.objects.filter(analysis__run=run)
    assert predictions.exists()
    assert all(not prediction.issued_on_time for prediction in predictions)
    output = stdout.getvalue()
    assert "research-grade" in output
    assert "issued_on_time=False" in output


def _reportable_prediction_chain(
    snapshot: UniverseSnapshot,
    listing: Listing,
    *,
    generated_at: datetime,
    target_date: date,
    model_version: str,
    method_version: str = "us-price-baseline-v2",
    config_hash: str = "2" * 64,
    price_provider: str = "twelve_data",
    horizon: str = Prediction.Horizon.SHORT,
    evidence_role: str = Prediction.EvidenceRole.DECISION,
    issued_on_time: bool = True,
    run_issued_on_time: bool = True,
    evidence_grade: str = UniverseSnapshot.Grade.OBSERVED,
    source_mode: str = Prediction.SourceMode.PROVIDER,
) -> Prediction:
    """Focused helper for canonical-reporting tests: one reportable-shaped
    Prediction (with its own run/analysis) per call, parameterized on
    exactly the fields relevant to the canonical observation key and
    reportability -- avoids a Cartesian explosion of near-duplicate fixture
    setup across the canonicality tests below."""
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=target_date,
        issued_on_time=run_issued_on_time,
        universe_snapshot=snapshot,
        config_version=method_version,
        config_hash=config_hash,
        code_revision="test-revision",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("100"),
        overall_score=Decimal("70"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("35"),
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("60"),
    )
    return Prediction.objects.create(
        analysis=analysis,
        listing=listing,
        generated_at=generated_at,
        target_date=target_date,
        issued_on_time=issued_on_time,
        horizon=horizon,
        evidence_role=evidence_role,
        evidence_grade=evidence_grade,
        source_mode=source_mode,
        price_provider=price_provider,
        price_subject=listing.ticker,
        price_at_prediction=Decimal("100"),
        bear_return=Decimal("-0.03"),
        base_return=Decimal("0.02"),
        bull_return=Decimal("0.07"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="heuristic",
        insufficiency_reason="",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version=model_version,
        method_version=method_version,
        config_hash=config_hash,
        data_cutoff=generated_at,
        code_revision="test-revision",
    )


def _canonical_ids() -> set:
    return set(
        Prediction.objects.filter(canonical_reportable_prediction_filter()).values_list(
            "id", flat=True
        )
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "gate_override",
    [
        {"issued_on_time": False},
        {"run_issued_on_time": False},
        {"price_provider": ""},
        {"evidence_grade": UniverseSnapshot.Grade.RESEARCH},
        {"source_mode": Prediction.SourceMode.SYNTHETIC},
    ],
    ids=["off_time", "run_off_time", "no_provider", "non_observed_grade", "non_provider_source"],
)
def test_reportable_prediction_filter_requires_every_reportability_gate(
    gate_override: dict[str, object],
) -> None:
    """The base (non-canonical) predicate requires every gate at once:
    observed grade, provider-backed source, on-time prediction and parent
    run, and a non-empty price_provider. No observation-key/sibling logic."""
    listing, snapshot = _listing_and_snapshot()
    reportable = _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=date(2026, 9, 8),
        model_version="v-reportable",
    )
    gate_failing = _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=date(2026, 9, 9),
        model_version="v-gate-failing",
        **gate_override,
    )

    reportable_ids = set(
        Prediction.objects.filter(reportable_prediction_filter()).values_list("id", flat=True)
    )

    assert reportable_ids == {reportable.id}
    assert gate_failing.id not in reportable_ids


@pytest.mark.django_db
def test_canonical_reportable_filter_selects_earliest_generated_at() -> None:
    listing, snapshot = _listing_and_snapshot()
    target = date(2026, 9, 8)
    earliest = _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=target,
        model_version="v-earliest",
    )
    later = _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 2, tzinfo=UTC),
        target_date=target,
        model_version="v-later-reissue",
    )

    assert _canonical_ids() == {earliest.id}
    assert later.id not in _canonical_ids()


@pytest.mark.django_db
def test_canonical_reportable_filter_breaks_generated_at_tie_by_uuid() -> None:
    listing, snapshot = _listing_and_snapshot()
    target = date(2026, 9, 8)
    same_time = datetime(2026, 9, 8, 1, tzinfo=UTC)
    first = _reportable_prediction_chain(
        snapshot, listing, generated_at=same_time, target_date=target, model_version="v-a"
    )
    second = _reportable_prediction_chain(
        snapshot, listing, generated_at=same_time, target_date=target, model_version="v-b"
    )

    expected_winner = min(first.id, second.id)

    assert _canonical_ids() == {expected_winner}


@pytest.mark.django_db
def test_canonical_reportable_filter_ignores_earlier_non_reportable_sibling() -> None:
    """An earlier research-grade/off-time row must never suppress a later
    genuinely reportable one for the same observation key."""
    listing, snapshot = _listing_and_snapshot()
    target = date(2026, 9, 8)
    _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 7, 1, tzinfo=UTC),
        target_date=target,
        model_version="v-research-grade",
        run_issued_on_time=False,
        issued_on_time=False,
    )
    later_reportable = _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=target,
        model_version="v-observed",
    )

    assert _canonical_ids() == {later_reportable.id}


@pytest.mark.django_db
@pytest.mark.parametrize(
    "earliest_status",
    [None, PredictionOutcome.Status.UNRESOLVED, PredictionOutcome.Status.CORPORATE_EVENT],
)
def test_canonical_reportable_filter_keeps_earliest_state_over_later_matured_reissue(
    earliest_status: str | None,
) -> None:
    """A later same-key reissue that matures never displaces an earlier
    reportable row that has no outcome yet, is unresolved, or is a
    corporate event: the canonical predicate looks only at reportability."""
    listing, snapshot = _listing_and_snapshot()
    target = date(2026, 9, 8)
    earliest = _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=target,
        model_version="v-earliest",
    )
    if earliest_status is not None:
        PredictionOutcome.objects.create(
            prediction=earliest,
            evaluated_at=datetime(2026, 10, 1, tzinfo=UTC),
            evaluation_date=date(2026, 10, 1),
            status=earliest_status,
            resolution="Earliest state",
        )
    reissue = _reportable_prediction_chain(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 2, tzinfo=UTC),
        target_date=target,
        model_version="v-matured-reissue",
    )
    PredictionOutcome.objects.create(
        prediction=reissue,
        evaluated_at=datetime(2026, 10, 1, tzinfo=UTC),
        evaluation_date=date(2026, 10, 1),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.05"),
        success=True,
        resolution="Matured reissue",
    )

    assert _canonical_ids() == {earliest.id}
    canonical_matured_outcomes = PredictionOutcome.objects.filter(
        canonical_reportable_prediction_filter("prediction__"),
        status=PredictionOutcome.Status.MATURED,
    )
    assert canonical_matured_outcomes.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("overrides_a", "overrides_b"),
    [
        pytest.param({}, {"target_date": date(2026, 9, 9)}, id="target_date"),
        pytest.param({}, {"horizon": Prediction.Horizon.MEDIUM}, id="horizon"),
        pytest.param(
            {},
            {
                "evidence_role": Prediction.EvidenceRole.ADVISORY,
                "horizon": Prediction.Horizon.SIX_MONTH,
            },
            id="evidence_role_with_paired_valid_horizon",
        ),
        pytest.param({}, {"method_version": "us-price-baseline-v1"}, id="method_version"),
        pytest.param({}, {"config_hash": "3" * 64}, id="config_hash"),
        pytest.param({}, {"price_provider": "alpha_vantage"}, id="price_provider"),
    ],
)
def test_canonical_reportable_filter_key_separates_each_dimension(
    overrides_a: dict[str, object],
    overrides_b: dict[str, object],
) -> None:
    """The observation key keeps distinct listing/target_date/horizon/
    evidence_role/method_version/config_hash/price_provider observations
    independently canonical; only an exact match on every field is treated
    as a reissue of the same observation. `evidence_role` is only varied
    paired with a model-valid horizon for that role."""
    listing, snapshot = _listing_and_snapshot()
    base = {
        "generated_at": datetime(2026, 9, 8, 1, tzinfo=UTC),
        "target_date": date(2026, 9, 8),
    }
    prediction_a = _reportable_prediction_chain(
        snapshot, listing, model_version="v-a", **{**base, **overrides_a}
    )
    prediction_b = _reportable_prediction_chain(
        snapshot, listing, model_version="v-b", **{**base, **overrides_b}
    )

    assert _canonical_ids() == {prediction_a.id, prediction_b.id}


@pytest.mark.django_db
def test_canonical_reportable_filter_key_separates_listing() -> None:
    listing_a, snapshot = _listing_and_snapshot()
    listing_b = _add_listing_to_snapshot(snapshot)
    base = {
        "generated_at": datetime(2026, 9, 8, 1, tzinfo=UTC),
        "target_date": date(2026, 9, 8),
    }
    prediction_a = _reportable_prediction_chain(snapshot, listing_a, model_version="v-a", **base)
    prediction_b = _reportable_prediction_chain(snapshot, listing_b, model_version="v-b", **base)

    assert _canonical_ids() == {prediction_a.id, prediction_b.id}


@pytest.mark.django_db
def test_canonical_reportable_filter_is_lazy_and_compiles_to_not_exists(
    django_assert_num_queries,
) -> None:
    """Building the shared predicate never executes a query itself (no
    Python-side ID materialization), and the resulting SQL expresses "no
    earlier reportable sibling" as a single correlated `NOT EXISTS`
    subquery rather than a window annotation or `DISTINCT ON`."""
    with django_assert_num_queries(0):
        predicate = canonical_reportable_prediction_filter("prediction__")
        queryset = PredictionOutcome.objects.filter(predicate)
        prediction_predicate = canonical_reportable_prediction_filter()
        prediction_queryset = Prediction.objects.filter(prediction_predicate)

    sql = str(queryset.query).upper()
    assert "NOT EXISTS" in sql
    assert "DISTINCT ON" not in sql
    prediction_sql = str(prediction_queryset.query).upper()
    assert "NOT EXISTS" in prediction_sql


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
    store: AssetStore,
    subject: str,
    available_at,
    *,
    rows: int = 280,
    provider: str = "synthetic",
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
        provider=provider,
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


# ---------------------------------------------------------------------------
# Under-$10 shadow assessment: enrichment stays additive on the existing paths.
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_ordinary_priced_analysis_gains_no_shadow_key(tmp_path) -> None:
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

    assert persisted.analysis.current_price >= Decimal("10")
    assert "under10_assessment" not in persisted.analysis.data_quality
    assert "under10_assessment" not in persisted.computation.data_quality


@pytest.mark.django_db
def test_synthetic_under_ten_analysis_records_an_honest_withheld_assessment(tmp_path) -> None:
    listing, snapshot = _listing_and_snapshot()
    store = AssetStore(tmp_path)
    now = timezone.now()
    dates = [timezone.localdate() - timedelta(days=index) for index in range(30)][::-1]
    _register_explicit_price_asset(
        store,
        listing.ticker,
        now - timedelta(minutes=5),
        dates,
        [4.25] * len(dates),
    )

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=now,
        provider="synthetic",
        store=store,
    )

    payload = persisted.analysis.data_quality["under10_assessment"]
    assert payload["policy_version"] == "us-under10-shadow-v1"
    assert payload["evaluated_for"]["reference_close"] == "4.250000"
    # A synthetic demo asset proves no reviewed daily split-only USD basis, so
    # the liquidity diagnostic is withheld rather than invented.
    assert payload["liquidity"]["status"] == "withheld"
    assert payload["liquidity"]["reason"] == "basis_incompatible"
    assert payload["liquidity"]["basis"]["interval"] is None
    # No SEC evidence exists for a synthetic company.
    assert payload["solvency"]["status"] == "insufficient_evidence"
    assert payload["solvency"]["assessed_fact_ids"] == []
    assert payload["split_verification"]["reason"] == "no_reviewed_corporate_actions_source"
    assert payload["activation_eligible"] is False
    assert payload["new_allocation_percent"] == 0
    assert all(
        "under10_assessment" not in prediction.calculation for prediction in persisted.predictions
    )
