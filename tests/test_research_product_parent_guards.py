from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.core.exceptions import ValidationError
from django.db import DatabaseError, IntegrityError, transaction

from stanstock.data.models import Company, Listing, Region, Security, Universe, UniverseSnapshot
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.price_product_config import (
    MOMENTUM_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
)

MIGRATION_MODULE = (
    "stanstock.research.migrations.0009_remove_prediction_prediction_score_in_range_and_more"
)


def _context(
    config_version: str,
    *,
    config_hash: str | None = None,
) -> tuple[Listing, AnalysisRun]:
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
            PRODUCT_EFFECTIVE_CONFIG_HASH
            if config_hash is None and config_version == PRODUCT_VERSION
            else config_hash
            if config_hash is not None
            else "b" * 64
        ),
        code_revision="test-revision",
    )
    return listing, run


def _product_analysis(listing: Listing, run: AnalysisRun) -> StockAnalysis:
    return StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("25.000000"),
        overall_score=None,
        recommendation=Recommendation.HOLD,
        risk_score=None,
        risk_class=RiskClass.MEDIUM,
        confidence=None,
        confidence_status="not_estimated",
    )


def _legacy_analysis(listing: Listing, run: AnalysisRun) -> StockAnalysis:
    return StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("20.000000"),
        overall_score=Decimal("61"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("25"),
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("52"),
        confidence_status="custom",
    )


def _product_prediction_kwargs(
    analysis: StockAnalysis,
    *,
    config_hash: str,
    model_version: str,
    recommendation: str | None = Recommendation.HOLD,
    horizon: str = Prediction.Horizon.SIX_MONTH,
    evidence_role: str = Prediction.EvidenceRole.DECISION,
    method_version: str = MOMENTUM_METHOD_VERSION,
) -> dict[str, object]:
    return {
        "analysis": analysis,
        "listing": analysis.listing,
        "generated_at": analysis.run.generated_at,
        "target_date": analysis.run.target_date,
        "horizon": horizon,
        "evidence_role": evidence_role,
        "evidence_grade": UniverseSnapshot.Grade.RESEARCH,
        "source_mode": Prediction.SourceMode.PROVIDER,
        "price_provider": "twelve_data",
        "price_subject": analysis.listing.ticker,
        "price_at_prediction": analysis.current_price,
        "bear_return": None,
        "base_return": None,
        "bull_return": None,
        "probability_positive": None,
        "confidence": None,
        "confidence_status": "not_estimated",
        "insufficiency_reason": "",
        "recommendation": recommendation,
        "overall_score": None,
        "component_scores": {},
        "model_version": model_version,
        "method_version": method_version,
        "config_hash": config_hash,
        "data_cutoff": analysis.run.data_cutoff,
        "source_assets": [],
        "calculation": {"schema_version": "research-product@1"},
        "code_revision": "test-revision",
    }


def _legacy_prediction_kwargs(
    analysis: StockAnalysis,
    *,
    config_hash: str,
    model_version: str,
) -> dict[str, object]:
    return {
        "analysis": analysis,
        "listing": analysis.listing,
        "generated_at": analysis.run.generated_at,
        "target_date": analysis.run.target_date,
        "horizon": Prediction.Horizon.SHORT,
        "evidence_role": Prediction.EvidenceRole.DECISION,
        "evidence_grade": UniverseSnapshot.Grade.RESEARCH,
        "source_mode": Prediction.SourceMode.PROVIDER,
        "price_provider": "twelve_data",
        "price_subject": analysis.listing.ticker,
        "price_at_prediction": analysis.current_price,
        "bear_return": Decimal("-0.1000"),
        "base_return": Decimal("0.0500"),
        "bull_return": Decimal("0.2000"),
        "probability_positive": None,
        "confidence": Decimal("52"),
        "confidence_status": "custom",
        "insufficiency_reason": "",
        "recommendation": Recommendation.HOLD,
        "overall_score": Decimal("61"),
        "component_scores": {},
        "model_version": model_version,
        "method_version": "custom-legacy-v7",
        "config_hash": config_hash,
        "data_cutoff": analysis.run.data_cutoff,
        "source_assets": [],
        "calculation": {"schema_version": "legacy"},
        "code_revision": "test-revision",
    }


def _guard_sql(vendor: str) -> tuple[str, list[str]]:
    migration = import_module(MIGRATION_MODULE)
    statements: list[str] = []
    schema_editor = SimpleNamespace(
        connection=SimpleNamespace(vendor=vendor),
        execute=statements.append,
    )
    migration.install_product_guards(apps=SimpleNamespace(), schema_editor=schema_editor)
    return "\n".join(statements), statements


@pytest.mark.django_db
def test_analysis_run_parent_updates_block_product_children_but_empty_runs_stay_mutable() -> None:
    listing, run = _context(PRODUCT_VERSION)
    _product_analysis(listing, run)

    run.status = "complete"
    run.save()
    AnalysisRun.objects.filter(pk=run.pk).update(
        config_version=PRODUCT_VERSION, config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH
    )
    run.config_version = "custom-legacy-v7"
    run.config_hash = "c" * 64
    with pytest.raises(ValidationError, match="cannot change config_version or config_hash"):
        run.clean()
    with pytest.raises((DatabaseError, IntegrityError)), transaction.atomic():
        AnalysisRun.objects.filter(pk=run.pk).update(config_version="custom-legacy-v7")
    with pytest.raises((DatabaseError, IntegrityError)), transaction.atomic():
        AnalysisRun.objects.filter(pk=run.pk).update(config_hash="c" * 64)

    empty_listing, empty_run = _context(PRODUCT_VERSION)
    empty_run.config_version = "custom-legacy-v7"
    empty_run.config_hash = "c" * 64
    empty_run.clean()
    AnalysisRun.objects.filter(pk=empty_run.pk).update(
        config_version="custom-legacy-v7",
        config_hash="c" * 64,
    )
    empty_run.refresh_from_db()
    assert empty_run.config_version == "custom-legacy-v7"
    assert empty_run.config_hash == "c" * 64
    assert empty_listing.pk is not None

    legacy_listing, legacy = _context("custom-legacy-v7", config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH)
    _legacy_analysis(legacy_listing, legacy)
    legacy.config_version = "custom-legacy-v8"
    legacy.config_hash = "d" * 64
    legacy.clean()
    legacy.save()
    legacy.refresh_from_db()
    assert legacy.config_version == "custom-legacy-v8"


@pytest.mark.django_db
def test_stockanalysis_reparenting_blocks_product_transplants_and_preserves_legacy_to_legacy() -> (
    None
):
    product_listing, product_run = _context(PRODUCT_VERSION)
    product_analysis = _product_analysis(product_listing, product_run)
    _other_product_listing, other_product_run = _context(PRODUCT_VERSION)
    legacy_listing, legacy_run = _context("custom-legacy-v7", config_hash="c" * 64)
    legacy_run_2 = _context("custom-legacy-v8", config_hash="d" * 64)[1]

    product_analysis.run = other_product_run
    with pytest.raises(ValidationError, match="reparented"):
        product_analysis.clean()
    with pytest.raises((DatabaseError, IntegrityError)), transaction.atomic():
        StockAnalysis.objects.filter(pk=product_analysis.pk).update(run_id=other_product_run.pk)

    legacy_into_product = _legacy_analysis(legacy_listing, legacy_run)
    legacy_into_product.run = product_run
    legacy_into_product.overall_score = None
    legacy_into_product.confidence = None
    legacy_into_product.recommendation = None
    legacy_into_product.confidence_status = "not_estimated"
    with pytest.raises(ValidationError, match="reparented"):
        legacy_into_product.clean()
    with pytest.raises((DatabaseError, IntegrityError)), transaction.atomic():
        StockAnalysis.objects.filter(pk=legacy_into_product.pk).update(
            run_id=product_run.pk,
            overall_score=None,
            confidence=None,
            recommendation=None,
            confidence_status="not_estimated",
        )

    legacy_legacy = _legacy_analysis(product_listing, legacy_run)
    legacy_legacy.run = legacy_run_2
    legacy_legacy.clean()
    StockAnalysis.objects.filter(pk=legacy_legacy.pk).update(run_id=legacy_run_2.pk)
    legacy_legacy.refresh_from_db()
    assert legacy_legacy.run_id == legacy_run_2.pk


@pytest.mark.django_db
def test_prediction_config_hash_must_match_parent_run_and_legacy_predictions_remain_valid() -> None:
    listing, run = _context(PRODUCT_VERSION)
    analysis = _product_analysis(listing, run)
    valid_kwargs = _product_prediction_kwargs(
        analysis,
        config_hash=run.config_hash,
        model_version="product-momentum-1",
    )
    valid_prediction = Prediction.objects.create(**valid_kwargs)
    assert valid_prediction.config_hash == run.config_hash

    wrong_hash_kwargs = _product_prediction_kwargs(
        analysis,
        config_hash="e" * 64,
        model_version="product-momentum-2",
    )
    with pytest.raises(ValidationError, match="config_hash"):
        Prediction(**wrong_hash_kwargs).clean()
    with pytest.raises((DatabaseError, IntegrityError)), transaction.atomic():
        Prediction.objects.create(**wrong_hash_kwargs)

    legacy_listing, legacy_run = _context("custom-legacy-v7", config_hash="c" * 64)
    legacy_analysis = _legacy_analysis(legacy_listing, legacy_run)
    legacy_prediction = Prediction.objects.create(
        **_legacy_prediction_kwargs(
            legacy_analysis,
            config_hash="d" * 64,
            model_version="legacy-decision-1",
        )
    )
    legacy_prediction.clean()
    assert legacy_prediction.config_hash == "d" * 64


@pytest.mark.parametrize("vendor", ["sqlite", "postgresql"])
def test_install_product_guards_include_analysisrun_and_prediction_hash_parity(
    vendor: str,
) -> None:
    sql, _ = _guard_sql(vendor)
    assert "research_analysisrun" in sql
    assert (
        "research_analysisrun_product_update" in sql or "research_analysisrun_product_guard" in sql
    )
    assert "OLD.config_version <> NEW.config_version" in sql
    assert "OLD.config_hash <> NEW.config_hash" in sql
    assert "OLD.run_id <> NEW.run_id" in sql
    assert "config_hash" in sql
    if vendor == "postgresql":
        assert "stanstock_validate_analysisrun_product" in sql
        assert "parent_config_hash" in sql


@pytest.mark.parametrize("vendor", ["sqlite", "postgresql"])
def test_uninstall_product_guards_drop_analysisrun_guard_everywhere(vendor: str) -> None:
    migration = import_module(MIGRATION_MODULE)
    statements: list[str] = []
    schema_editor = SimpleNamespace(
        connection=SimpleNamespace(vendor=vendor),
        execute=statements.append,
    )

    migration.uninstall_product_guards(apps=SimpleNamespace(), schema_editor=schema_editor)
    sql = "\n".join(statements)
    if vendor == "sqlite":
        assert "DROP TRIGGER IF EXISTS research_analysisrun_product_update" in sql
        assert "DROP TRIGGER IF EXISTS research_prediction_product_update" in sql
    else:
        assert "DROP TRIGGER IF EXISTS research_analysisrun_product_guard" in sql
        assert "DROP FUNCTION IF EXISTS stanstock_validate_analysisrun_product()" in sql
