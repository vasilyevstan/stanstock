from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import polars as pl
import pytest
from django.core.exceptions import ValidationError
from django.db import DatabaseError, IntegrityError, connection, transaction

from stanstock.data.models import (
    Company,
    Listing,
    Region,
    Security,
    Universe,
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
from stanstock.research.outcomes import resolve_outcome
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    MOMENTUM_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
)

MIGRATION_MODULE = (
    "stanstock.research.migrations.0009_remove_prediction_prediction_score_in_range_and_more"
)


def _context(config_version: str) -> tuple[Listing, AnalysisRun]:
    company = Company.objects.create(name=f"Product {uuid4().hex[:8]}", country="US")
    security = Security.objects.create(company=company)
    listing = Listing.objects.create(
        security=security,
        ticker=f"P{uuid4().hex[:6]}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug=f"product-{uuid4().hex[:8]}",
        name="Product test",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=date(2026, 9, 11),
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="a" * 64,
    )
    generated_at = datetime(2026, 9, 11, 21, tzinfo=UTC)
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=date(2026, 9, 11),
        universe_snapshot=snapshot,
        config_version=config_version,
        config_hash=(
            PRODUCT_EFFECTIVE_CONFIG_HASH if config_version == PRODUCT_VERSION else "b" * 64
        ),
        code_revision="test-revision",
    )
    return listing, run


def _prospective_analysis(
    listing: Listing,
    run: AnalysisRun,
    *,
    recommendation: str | None = Recommendation.HOLD,
    momentum_reason: str | None = None,
) -> StockAnalysis:
    data_quality = {}
    if momentum_reason is not None:
        data_quality["momentum_insufficiency_reason"] = momentum_reason
    return StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("25.000000"),
        overall_score=None,
        recommendation=recommendation,
        risk_score=None,
        risk_class=RiskClass.MEDIUM,
        confidence=None,
        confidence_status="not_estimated",
        data_quality=data_quality,
    )


def _prospective_prediction(
    analysis: StockAnalysis,
    *,
    method_version: str,
    horizon: str,
    evidence_role: str,
    recommendation: str | None,
    model_version: str,
    insufficiency_reason: str = "",
    scenarios: tuple[Decimal | None, Decimal | None, Decimal | None] = (
        None,
        None,
        None,
    ),
) -> Prediction:
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=analysis.run.generated_at,
        target_date=analysis.run.target_date,
        horizon=horizon,
        evidence_role=evidence_role,
        evidence_grade=UniverseSnapshot.Grade.RESEARCH,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=analysis.listing.ticker,
        price_at_prediction=analysis.current_price,
        bear_return=scenarios[0],
        base_return=scenarios[1],
        bull_return=scenarios[2],
        probability_positive=None,
        confidence=None,
        confidence_status="not_estimated",
        insufficiency_reason=insufficiency_reason,
        recommendation=recommendation,
        overall_score=None,
        component_scores={},
        model_version=model_version,
        method_version=method_version,
        config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        data_cutoff=analysis.run.data_cutoff,
        source_assets=[],
        calculation={"schema_version": "research-product@1"},
        code_revision="test-revision",
    )


@pytest.mark.django_db
def test_prospective_rows_store_honest_nulls_and_guard_six_month_decision() -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(listing, run)

    decision = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.HOLD,
        model_version="momentum-issuance-1",
    )
    advisory = _prospective_prediction(
        analysis,
        method_version=FHS_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        recommendation=None,
        model_version="fhs-issuance-1",
        scenarios=(Decimal("-0.1000"), Decimal("0.0500"), Decimal("0.2000")),
    )

    assert analysis.overall_score is None
    assert analysis.confidence is None
    assert analysis.risk_score is None
    assert decision.overall_score is None
    assert decision.confidence is None
    assert decision.probability_positive is None
    assert advisory.recommendation is None
    assert decision.method_version != decision.model_version


@pytest.mark.django_db
def test_unavailable_prospective_recommendation_requires_explicit_reason() -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(
        listing,
        run,
        recommendation=None,
        momentum_reason="common_session_history_missing",
    )
    analysis.clean()

    unavailable = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=None,
        model_version="momentum-unavailable-1",
        insufficiency_reason="common_session_history_missing",
    )
    unavailable.clean()

    with pytest.raises(IntegrityError), transaction.atomic():
        _prospective_prediction(
            analysis,
            method_version=MOMENTUM_METHOD_VERSION,
            horizon=Prediction.Horizon.SIX_MONTH,
            evidence_role=Prediction.EvidenceRole.DECISION,
            recommendation=None,
            model_version="momentum-unavailable-no-reason",
        )


@pytest.mark.django_db
def test_cross_parent_guards_preserve_custom_legacy_configs_and_refuse_null_escape() -> None:
    listing, run = _context("custom-valid-research-config")
    legacy = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("20"),
        overall_score=Decimal("61"),
        recommendation=Recommendation.HOLD,
        risk_score=None,
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("52"),
        confidence_status="custom",
    )
    assert legacy.pk is not None

    with pytest.raises(IntegrityError), transaction.atomic():
        StockAnalysis.objects.create(
            run=run,
            listing=listing,
            current_price=Decimal("21"),
            overall_score=None,
            recommendation=None,
            risk_score=None,
            risk_class=RiskClass.MEDIUM,
            confidence=None,
            confidence_status="not_estimated",
        )


@pytest.mark.django_db
def test_prospective_methods_cannot_be_attached_to_legacy_parent() -> None:
    listing, run = _context("another-custom-v3")
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("20"),
        overall_score=Decimal("61"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("30"),
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("52"),
    )

    with pytest.raises((IntegrityError, DatabaseError)), transaction.atomic():
        _prospective_prediction(
            analysis,
            method_version=MOMENTUM_METHOD_VERSION,
            horizon=Prediction.Horizon.SIX_MONTH,
            evidence_role=Prediction.EvidenceRole.DECISION,
            recommendation=Recommendation.HOLD,
            model_version="wrong-parent",
        )


@pytest.mark.django_db
def test_prospective_hold_and_unavailable_mature_without_success_claim() -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(listing, run)
    hold = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.HOLD,
        model_version="hold-outcome",
    )
    outcome = PredictionOutcome.objects.create(
        prediction=hold,
        evaluated_at=run.generated_at,
        evaluation_date=run.target_date,
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.1000"),
        success=None,
        resolution="non-directional",
    )
    outcome.clean()

    buy = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.BUY,
        model_version="buy-outcome",
    )
    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=buy,
            evaluated_at=run.generated_at,
            evaluation_date=run.target_date,
            status=PredictionOutcome.Status.MATURED,
            actual_return=Decimal("0.1000"),
            success=None,
            resolution="invalid directional outcome",
        )


@pytest.mark.django_db
def test_momentum_outcomes_use_relative_direction_and_leave_hold_success_null() -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(listing, run)
    buy = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.BUY,
        model_version="relative-buy-outcome",
    )
    hold = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.HOLD,
        model_version="relative-hold-outcome",
    )
    sessions = tuple(run.target_date + timedelta(days=index) for index in range(127))
    stock = pl.DataFrame(
        {
            "date": sessions,
            "close": [25.0, *[25.0] * 125, 30.0],
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )
    benchmark = pl.DataFrame(
        {
            "date": sessions,
            "close": [100.0, *[100.0] * 125, 110.0],
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )

    def price_loader(subject: str, through_date: date) -> pl.DataFrame:
        assert through_date == sessions[-1]
        return benchmark if subject == "SPY" else stock

    buy_result = resolve_outcome(
        buy,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=datetime.combine(sessions[-1], datetime.min.time(), tzinfo=UTC),
        benchmark_subject="SPY",
        price_loader=price_loader,
    )
    hold_result = resolve_outcome(
        hold,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=datetime.combine(sessions[-1], datetime.min.time(), tzinfo=UTC),
        benchmark_subject="SPY",
        price_loader=price_loader,
    )

    assert buy_result.status == PredictionOutcome.Status.MATURED
    assert buy_result.success is True
    assert buy_result.benchmark_return == Decimal("0.1")
    assert hold_result.status == PredictionOutcome.Status.MATURED
    assert hold_result.success is None
    assert (
        hold_result.metadata["success_semantics"]
        == "HOLD or unavailable momentum decisions do not receive success labels"
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("recommendation", "stock_final", "benchmark_final"),
    [
        (Recommendation.BUY, 30.0, 110.0),
        (Recommendation.AVOID, 20.0, 90.0),
    ],
)
def test_directional_momentum_uses_exact_aligned_spy_sessions(
    recommendation: str,
    stock_final: float,
    benchmark_final: float,
) -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(listing, run)
    prediction = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=recommendation,
        model_version=f"aligned-{recommendation}",
    )
    sessions = tuple(run.target_date + timedelta(days=index) for index in range(127))
    stock = pl.DataFrame(
        {
            "date": sessions,
            "close": [25.0, *[25.0] * 125, stock_final],
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )
    benchmark = pl.DataFrame(
        {
            "date": sessions,
            "close": [100.0, *[100.0] * 125, benchmark_final],
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )

    result = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=datetime.combine(sessions[-1], datetime.min.time(), tzinfo=UTC),
        benchmark_subject="SPY",
        price_loader=lambda subject, _through: benchmark if subject == "SPY" else stock,
    )

    assert result.status == PredictionOutcome.Status.MATURED
    assert result.success is True
    assert result.benchmark_return == Decimal(str(benchmark_final / 100.0 - 1.0)).quantize(
        Decimal("0.0001")
    )
    assert result.metadata["benchmark_subject"] == "SPY"
    assert "uses exact" in result.metadata["benchmark_resolution"]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("benchmark_dates", "expected_resolution"),
    [
        (
            "truncated",
            "Prospective momentum benchmark has no exact maturity-session close",
        ),
        (
            "missing_target",
            "Prospective momentum benchmark has no exact target-date close",
        ),
    ],
)
def test_momentum_benchmark_gaps_are_explicitly_unresolved(
    benchmark_dates: str,
    expected_resolution: str,
) -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(listing, run)
    prediction = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.BUY,
        model_version=f"gap-{benchmark_dates}",
    )
    sessions = tuple(run.target_date + timedelta(days=index) for index in range(127))
    stock = pl.DataFrame(
        {"date": sessions, "close": [25.0, *[25.0] * 125, 26.0]},
        schema={"date": pl.Date, "close": pl.Float64},
    )
    selected_dates = sessions[:61] if benchmark_dates == "truncated" else sessions[1:]
    benchmark_closes = (
        [100.0 + 20.0 * index / (len(selected_dates) - 1) for index in range(len(selected_dates))]
        if benchmark_dates == "truncated"
        else [100.0 + index / 10 for index in range(len(selected_dates))]
    )
    benchmark = pl.DataFrame(
        {
            "date": selected_dates,
            "close": benchmark_closes,
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )

    result = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=datetime.combine(sessions[-1], datetime.min.time(), tzinfo=UTC),
        benchmark_subject="SPY",
        price_loader=lambda subject, _through: benchmark if subject == "SPY" else stock,
    )

    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.success is None
    assert result.actual_return is None
    assert result.resolution == expected_resolution


@pytest.mark.django_db
@pytest.mark.parametrize("benchmark_subject", [None, "QQQ", "OTHER"])
def test_momentum_outcome_refuses_caller_selected_benchmark(
    benchmark_subject: str | None,
) -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(listing, run)
    prediction = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.BUY,
        model_version=f"benchmark-binding-{benchmark_subject}",
    )

    def unexpected_loader(_subject: str, _through: date) -> pl.DataFrame:
        raise AssertionError("invalid benchmark binding must fail before price IO")

    result = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=run.target_date + timedelta(days=180),
        evaluated_at=run.generated_at + timedelta(days=180),
        benchmark_subject=benchmark_subject,
        price_loader=unexpected_loader,
    )

    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.success is None
    assert result.resolution == (
        "Prospective momentum decision requires configured benchmark subject SPY"
    )
    assert result.metadata["expected_benchmark_subject"] == "SPY"


@pytest.mark.django_db
def test_momentum_stock_baseline_cannot_fall_back_to_an_equal_prior_close():
    listing, run = _context(PRODUCT_VERSION)
    prediction = _prospective_prediction(
        _prospective_analysis(listing, run),
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.BUY,
        model_version="missing-exact-stock-target",
    )
    sessions = tuple(run.target_date + timedelta(days=index) for index in range(127))
    stock = pl.DataFrame(
        {
            "date": (run.target_date - timedelta(days=1), *sessions[1:]),
            "close": [25.0, *[25.0] * 125, 30.0],
        },
        schema={"date": pl.Date, "close": pl.Float64},
    )
    benchmark = pl.DataFrame(
        {"date": sessions, "close": [100.0, *[100.0] * 125, 110.0]},
        schema={"date": pl.Date, "close": pl.Float64},
    )
    result = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=run.generated_at + timedelta(days=127),
        benchmark_subject="SPY",
        price_loader=lambda subject, _through: benchmark if subject == "SPY" else stock,
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.success is None
    assert result.actual_return is None
    assert result.resolution == "Momentum evaluation requires an exact target-date stock close"


@pytest.mark.django_db
def test_prediction_immutability_survives_nullable_table_rebuild() -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _prospective_analysis(listing, run)
    prediction = _prospective_prediction(
        analysis,
        method_version=MOMENTUM_METHOD_VERSION,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        recommendation=Recommendation.HOLD,
        model_version="immutable-product",
    )

    with pytest.raises(DatabaseError, match="immutable"), transaction.atomic():
        Prediction.objects.filter(pk=prediction.pk).update(insufficiency_reason="mutated")
    with pytest.raises(DatabaseError, match="immutable"), transaction.atomic():
        Prediction.objects.filter(pk=prediction.pk).delete()


@pytest.mark.django_db
def test_product_and_outcome_guards_are_installed_after_migration() -> None:
    with connection.cursor() as cursor:
        if connection.vendor == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            expected = {
                "research_prediction_prevent_update",
                "research_prediction_prevent_delete",
                "research_stockanalysis_product_insert",
                "research_stockanalysis_product_update",
                "research_prediction_product_insert",
                "research_prediction_product_update",
                "research_predictionoutcome_role_insert",
                "research_predictionoutcome_role_update",
            }
        elif connection.vendor == "postgresql":
            cursor.execute(
                """
                SELECT trigger_name
                FROM information_schema.triggers
                WHERE event_object_schema = current_schema()
                """
            )
            expected = {
                "research_prediction_prevent_update",
                "research_prediction_prevent_delete",
                "research_stockanalysis_product_guard",
                "research_prediction_product_guard",
                "research_predictionoutcome_role_guard",
            }
        else:
            pytest.skip(f"Unsupported database vendor: {connection.vendor}")
        installed = {row[0] for row in cursor.fetchall()}

    assert expected <= installed


def test_postgresql_guard_sql_has_cross_parent_method_config_parity() -> None:
    migration = import_module(MIGRATION_MODULE)
    statements: list[str] = []
    schema_editor = SimpleNamespace(
        connection=SimpleNamespace(vendor="postgresql"),
        execute=statements.append,
    )

    migration.install_product_guards(apps=None, schema_editor=schema_editor)

    sql = "\n".join(statements)
    assert "JOIN research_analysisrun run" in sql
    assert PRODUCT_VERSION in sql
    assert MOMENTUM_METHOD_VERSION in sql
    assert FHS_METHOD_VERSION in sql
    assert "stanstock_validate_prediction_outcome_role" in sql


def test_reverse_guard_is_first_reverse_action_and_refuses_prospective_rows() -> None:
    migration = import_module(MIGRATION_MODULE)

    class ExistingRows:
        def exists(self) -> bool:
            return True

    model = SimpleNamespace(objects=SimpleNamespace(filter=lambda *args, **kwargs: ExistingRows()))
    apps = SimpleNamespace(get_model=lambda app_label, model_name: model)

    assert migration.Migration.operations[-1].reverse_code is migration.reject_unsafe_reverse
    with pytest.raises(RuntimeError, match="Cannot reverse prospective"):
        migration.reject_unsafe_reverse(apps, schema_editor=None)


@pytest.mark.django_db
def test_model_validation_refuses_legacy_nulls_before_persistence() -> None:
    listing, run = _context("custom-legacy-v7")
    analysis = StockAnalysis(
        run=run,
        listing=listing,
        current_price=Decimal("20"),
        overall_score=None,
        recommendation=None,
        risk_score=None,
        risk_class=RiskClass.MEDIUM,
        confidence=None,
        confidence_status="not_estimated",
    )

    with pytest.raises(ValidationError, match="Legacy analyses require"):
        analysis.clean()
