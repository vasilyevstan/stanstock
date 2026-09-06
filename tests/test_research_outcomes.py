from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from uuid import uuid4

import polars as pl
import pytest
from django.core.management import call_command
from django.db import IntegrityError, transaction
from django.test import override_settings
from django.utils import timezone

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import Company, Listing, Region, Security, Universe, UniverseSnapshot
from stanstock.research.jobs import eligible_pending_predictions
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.outcomes import evaluate_prediction


@pytest.mark.django_db
def test_short_prediction_matures_on_nth_observed_session_without_fabricating_weekends(
    tmp_path,
) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    target = prediction.target_date
    sessions = _business_dates_after(target, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store, listing.ticker, _evaluation_time(), sessions, [101 + i for i in range(10)]
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.evaluation_date == sessions[-1]
    assert sessions[-1] != target + timedelta(days=10)
    assert result.outcome.actual_return == Decimal("0.1")
    assert result.outcome.success is True
    assert result.outcome.metadata["horizon_sessions"] == 10


@pytest.mark.django_db
def test_insufficient_observed_sessions_create_unresolved_without_returns(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 9)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store, listing.ticker, _evaluation_time(), sessions, [101 + i for i in range(9)]
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert "Insufficient observed sessions" in result.outcome.resolution
    assert result.outcome.actual_return is None
    assert result.outcome.benchmark_return is None
    assert result.outcome.success is None
    assert result.outcome.error is None


@pytest.mark.django_db
def test_missing_data_asset_creates_unresolved_outcome(tmp_path) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=prediction.target_date + timedelta(days=30),
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert "Unable to load evaluation price history" in result.outcome.resolution
    assert result.outcome.actual_return is None
    assert result.outcome.success is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("evaluation_date", "expected_resolution"),
    [
        (date(2027, 1, 1), "Evaluation date is after actual evaluation time"),
        (date(2026, 1, 1), "Evaluation date is before prediction target date"),
    ],
)
def test_invalid_evaluation_cutoff_dates_are_unresolved(
    tmp_path, evaluation_date: date, expected_resolution: str
) -> None:
    _, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT)

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=evaluation_date,
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert result.outcome.resolution == expected_resolution
    assert result.outcome.actual_return is None


@pytest.mark.django_db
def test_duplicate_usable_session_dates_are_unresolved_not_counted(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions + [sessions[0]],
        [101 + i for i in range(10)] + [111],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert result.outcome.resolution == "Duplicate usable price session dates in evaluation data"
    assert result.outcome.actual_return is None


@pytest.mark.django_db
def test_benchmark_return_uses_close_at_or_before_target_and_same_evaluation_session(
    tmp_path,
) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = _evaluation_time()
    _register_price_asset(
        store, listing.ticker, evaluation_time, sessions, [101 + i for i in range(10)]
    )
    _register_price_asset(
        store,
        "BENCH",
        evaluation_time,
        [
            prediction.target_date - timedelta(days=1),
            sessions[-2],
            sessions[-1] + timedelta(days=3),
        ],
        [200, 220, 9999],
        baseline_date=None,
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        benchmark_subject="BENCH",
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.benchmark_return == Decimal("0.1")
    assert "Benchmark return uses" in result.outcome.resolution


@pytest.mark.django_db
def test_duplicate_benchmark_session_dates_do_not_block_stock_maturity(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = _evaluation_time()
    _register_price_asset(
        store, listing.ticker, evaluation_time, sessions, [101 + i for i in range(10)]
    )
    _register_price_asset(
        store,
        "BENCH",
        evaluation_time,
        [prediction.target_date, sessions[-1], sessions[-1]],
        [200, 220, 221],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        benchmark_subject="BENCH",
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.actual_return == Decimal("0.1")
    assert result.outcome.success is True
    assert result.outcome.benchmark_return is None
    assert (
        "Benchmark unavailable: Duplicate usable price session dates"
        in (result.outcome.metadata["benchmark_resolution"])
    )


@pytest.mark.django_db
def test_existing_matured_outcome_is_idempotently_skipped(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    existing = PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=_evaluation_time() - timedelta(days=1),
        evaluation_date=prediction.target_date + timedelta(days=20),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.1234"),
        success=True,
        resolution="Already matured",
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(store, listing.ticker, _evaluation_time(), sessions, [1_000_000] * 10)

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    existing.refresh_from_db()
    assert result.action == "skipped"
    assert existing.actual_return == Decimal("0.1234")
    assert existing.resolution == "Already matured"


@pytest.mark.django_db
def test_stale_prediction_instance_cannot_overwrite_terminal_outcome(tmp_path) -> None:
    listing, analysis = _analysis()
    stale_prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
    )
    with pytest.raises(PredictionOutcome.DoesNotExist):
        _ = stale_prediction.outcome

    PredictionOutcome.objects.create(
        prediction=Prediction.objects.get(pk=stale_prediction.pk),
        evaluated_at=_evaluation_time() - timedelta(days=1),
        evaluation_date=stale_prediction.target_date + timedelta(days=20),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.1234"),
        success=True,
        resolution="Matured elsewhere",
    )

    result = evaluate_prediction(
        stale_prediction,
        provider="synthetic",
        evaluation_date=date(2027, 1, 1),
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    outcome = PredictionOutcome.objects.get(prediction=stale_prediction)
    assert result.action == "skipped"
    assert outcome.status == PredictionOutcome.Status.MATURED
    assert outcome.actual_return == Decimal("0.1234")


@pytest.mark.django_db
def test_adjusted_target_price_is_classified_as_corporate_event(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions,
        [50 + index for index in range(10)],
        baseline_close=Decimal("50"),
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.CORPORATE_EVENT
    assert result.outcome.actual_return is None
    assert result.outcome.success is None
    assert result.outcome.metadata["prediction_price"] == 100.0
    assert result.outcome.metadata["evaluation_vintage_target_close"] == 50.0


@pytest.mark.django_db
def test_evaluator_does_not_use_future_rows_in_eligible_asset(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions + [sessions[-1] + timedelta(days=1)],
        [101 + i for i in range(10)] + [9999],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.evaluation_date == sessions[-1]
    assert result.outcome.actual_return == Decimal("0.1")


@pytest.mark.django_db
def test_success_semantics_are_recommendation_specific(tmp_path) -> None:
    listing, analysis = _analysis()
    store = AssetStore(tmp_path)
    evaluation_time = _evaluation_time()
    sessions = _business_dates_after(analysis.run.target_date, 10)
    _register_price_asset(store, listing.ticker, evaluation_time, sessions, [98] * 10)
    buy = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY, version="buy"
    )
    avoid = _prediction(
        analysis,
        horizon=Prediction.Horizon.MEDIUM,
        recommendation=Recommendation.AVOID,
        version="avoid",
        target_date=analysis.run.target_date,
    )
    hold = _prediction(
        analysis,
        horizon=Prediction.Horizon.LONG,
        recommendation=Recommendation.HOLD,
        version="hold",
        target_date=analysis.run.target_date,
        bear=Decimal("-0.05"),
        bull=Decimal("0.05"),
    )
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time + timedelta(seconds=1),
        _business_dates_after(analysis.run.target_date, 756),
        [98] * 756,
    )

    assert (
        evaluate_prediction(
            buy,
            provider="synthetic",
            evaluation_date=sessions[-1],
            evaluation_time=evaluation_time,
            store=store,
        ).outcome.success
        is False
    )
    assert (
        evaluate_prediction(
            avoid,
            provider="synthetic",
            evaluation_date=_business_dates_after(analysis.run.target_date, 252)[-1],
            evaluation_time=evaluation_time + timedelta(seconds=1),
            store=store,
        ).outcome.success
        is True
    )
    assert (
        evaluate_prediction(
            hold,
            provider="synthetic",
            evaluation_date=_business_dates_after(analysis.run.target_date, 756)[-1],
            evaluation_time=datetime(2029, 1, 1, 12, tzinfo=timezone.get_current_timezone()),
            store=store,
        ).outcome.success
        is True
    )


@pytest.mark.django_db
def test_evaluate_command_all_pending_uses_tmp_asset_store(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        timezone.now() - timedelta(minutes=1),
        sessions,
        [101 + i for i in range(10)],
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "evaluate",
            all_pending=True,
            provider="synthetic",
            evaluation_date=sessions[-1].isoformat(),
            stdout=StringIO(),
        )

    assert (
        PredictionOutcome.objects.get(prediction=prediction).status
        == PredictionOutcome.Status.MATURED
    )


@pytest.mark.django_db
def test_evaluate_command_default_provider_is_synthetic_demo(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        timezone.now() - timedelta(minutes=1),
        sessions,
        [101 + i for i in range(10)],
        provider="synthetic_demo",
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "evaluate",
            str(prediction.pk),
            evaluation_date=sessions[-1].isoformat(),
            stdout=StringIO(),
        )

    assert (
        PredictionOutcome.objects.get(prediction=prediction).metadata["provider"]
        == "synthetic_demo"
    )


@pytest.mark.django_db
def test_prediction_outcome_constraints_enforce_matured_and_unresolved_contracts() -> None:
    _, analysis = _analysis()
    matured = _prediction(analysis, horizon=Prediction.Horizon.SHORT, version="bad-matured")
    unresolved = _prediction(analysis, horizon=Prediction.Horizon.MEDIUM, version="bad-unresolved")
    corporate_event = _prediction(
        analysis,
        horizon=Prediction.Horizon.LONG,
        version="bad-corporate-event",
    )

    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=matured,
            evaluated_at=_evaluation_time(),
            evaluation_date=matured.target_date,
            status=PredictionOutcome.Status.MATURED,
            resolution="invalid",
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=unresolved,
            evaluated_at=_evaluation_time(),
            evaluation_date=unresolved.target_date,
            status=PredictionOutcome.Status.UNRESOLVED,
            actual_return=Decimal("0.01"),
            resolution="invalid",
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=corporate_event,
            evaluated_at=_evaluation_time(),
            evaluation_date=corporate_event.target_date,
            status=PredictionOutcome.Status.CORPORATE_EVENT,
            actual_return=Decimal("0.01"),
            resolution="invalid",
        )


@pytest.mark.django_db
def test_prediction_constraints_require_positive_price_and_valid_data_cutoff() -> None:
    _, analysis = _analysis()

    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.SHORT,
            version="bad-price",
            price_at_prediction=Decimal("0"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.MEDIUM,
            version="bad-cutoff",
            data_cutoff=analysis.run.generated_at + timedelta(seconds=1),
        )


@pytest.mark.django_db
def test_prediction_constraints_require_non_null_returns_above_negative_one() -> None:
    _, analysis = _analysis()

    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.SHORT,
            version="bad-bear-floor",
            bear=Decimal("-1.0001"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.MEDIUM,
            version="bad-base-floor",
            base=Decimal("-1.0001"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.LONG,
            version="bad-bull-floor",
            bull=Decimal("-1.0001"),
        )


@pytest.mark.django_db
def test_prediction_scenario_constraint_keeps_all_null_or_ordered_all_present_invariant() -> None:
    _, analysis = _analysis()

    all_null = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="all-null",
        bear=None,
        base=None,
        bull=None,
    )
    assert all_null.bear_return is None
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.MEDIUM,
            version="partial-null",
            bear=Decimal("-0.10"),
            base=None,
            bull=Decimal("0.10"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.LONG,
            version="unordered",
            bear=Decimal("0.20"),
            base=Decimal("0.10"),
            bull=Decimal("0.30"),
        )


@pytest.mark.django_db
def test_pending_job_selection_is_provider_and_session_maturity_aware() -> None:
    _, analysis = _analysis()
    live_short = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="live-short",
        source_assets=[{"provider": "twelve_data"}],
    )
    _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="synthetic-short",
        source_assets=[{"provider": "synthetic_demo"}],
    )
    _prediction(
        analysis,
        horizon=Prediction.Horizon.MEDIUM,
        version="live-medium",
        source_assets=[{"provider": "twelve_data"}],
    )

    selected = eligible_pending_predictions(
        provider="twelve_data",
        evaluation_date=date(2026, 1, 16),
    )

    assert selected == [live_short]


@pytest.mark.django_db
def test_pending_job_selection_excludes_terminal_corporate_events() -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="terminal-short",
        source_assets=[{"provider": "twelve_data"}],
    )
    PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=_evaluation_time(),
        evaluation_date=date(2026, 1, 16),
        status=PredictionOutcome.Status.CORPORATE_EVENT,
        resolution="Split basis requires review",
    )

    assert (
        eligible_pending_predictions(
            provider="twelve_data",
            evaluation_date=date(2026, 1, 16),
        )
        == []
    )


def _analysis() -> tuple[Listing, StockAnalysis]:
    company = Company.objects.create(name=f"Outcome Co {uuid4().hex[:6]}", country="US")
    security = Security.objects.create(company=company)
    listing = Listing.objects.create(
        security=security,
        ticker=f"O{uuid4().hex[:6]}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug=f"outcome-{uuid4().hex[:8]}", name="Outcomes", config_version="1"
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=date(2026, 1, 2),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    generated_at = datetime(2026, 1, 2, 21, tzinfo=timezone.get_current_timezone())
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=date(2026, 1, 2),
        universe_snapshot=snapshot,
        config_version="default-v1",
        config_hash="b" * 64,
        code_revision="test",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("100"),
        overall_score=Decimal("70"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("30"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("60"),
    )
    return listing, analysis


def _prediction(
    analysis: StockAnalysis,
    *,
    horizon: Prediction.Horizon,
    recommendation: Recommendation = Recommendation.HOLD,
    version: str | None = None,
    target_date: date | None = None,
    bear: Decimal | None = Decimal("-0.10"),
    base: Decimal | None = Decimal("0.02"),
    bull: Decimal | None = Decimal("0.10"),
    price_at_prediction: Decimal = Decimal("100"),
    data_cutoff: datetime | None = None,
    source_assets: list[dict[str, str]] | None = None,
) -> Prediction:
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=analysis.run.generated_at,
        target_date=target_date or analysis.run.target_date,
        horizon=horizon,
        price_at_prediction=price_at_prediction,
        bear_return=bear,
        base_return=base,
        bull_return=bull,
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="heuristic",
        insufficiency_reason="",
        recommendation=recommendation,
        overall_score=Decimal("70"),
        component_scores={},
        model_version=version or f"outcome-{horizon.value}",
        config_hash="b" * 64,
        data_cutoff=data_cutoff or analysis.run.generated_at,
        source_assets=source_assets or [],
        code_revision="test",
    )


def _business_dates_after(start: date, count: int) -> list[date]:
    dates: list[date] = []
    current = start + timedelta(days=1)
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _evaluation_time() -> datetime:
    return datetime(2026, 12, 31, 12, tzinfo=timezone.get_current_timezone())


def _register_price_asset(
    store: AssetStore,
    subject: str,
    available_at: datetime,
    dates: list[date],
    closes: list[float],
    *,
    provider: str = "synthetic",
    baseline_date: date | None = date(2026, 1, 2),
    baseline_close: Decimal | float = Decimal("100"),
) -> None:
    if baseline_date is not None and baseline_date not in dates:
        dates = [baseline_date, *dates]
        closes = [float(baseline_close), *closes]
    frame = pl.DataFrame(
        {
            "date": dates,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )
    stored = store.write_frame(f"outcomes/{uuid4().hex}.parquet", frame)
    register_asset(
        provider=provider,
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )
