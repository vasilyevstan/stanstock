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
