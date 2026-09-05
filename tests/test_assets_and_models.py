from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import polars as pl
import pytest
from django.core.exceptions import ValidationError
from django.db import DatabaseError, transaction
from django.utils import timezone

from stanstock.data.asof import AsOfData
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
    Prediction,
    Recommendation,
    StockAnalysis,
)


@pytest.mark.django_db
def test_asof_uses_latest_asset_available_at_decision_time(tmp_path) -> None:
    store = AssetStore(tmp_path)
    now = timezone.now()
    old_frame = pl.DataFrame({"date": ["2026-09-01"], "close": [100.0]})
    new_frame = pl.DataFrame({"date": ["2026-09-01"], "close": [125.0]})

    old = store.write_frame("prices/old.parquet", old_frame)
    register_asset(
        provider="synthetic",
        kind="price_history",
        subject="TEST",
        stored=old,
        retrieved_at=now - timedelta(days=2),
        available_at=now - timedelta(days=2),
    )
    new = store.write_frame("prices/new.parquet", new_frame)
    register_asset(
        provider="synthetic",
        kind="price_history",
        subject="TEST",
        stored=new,
        retrieved_at=now + timedelta(days=1),
        available_at=now + timedelta(days=1),
    )

    frame = AsOfData(now, store).price_frame(provider="synthetic", subject="TEST")

    assert frame["close"].to_list() == [100.0]


@pytest.mark.django_db
def test_prediction_cannot_be_updated_or_deleted() -> None:
    company = Company.objects.create(name="Test Company", country="US")
    security = Security.objects.create(company=company)
    listing = Listing.objects.create(
        security=security,
        ticker="TEST",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug="test",
        name="Test",
        config_version="1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=timezone.localdate(),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    generated_at = timezone.now()
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=timezone.localdate(),
        universe_snapshot=snapshot,
        config_version="1",
        config_hash="b" * 64,
        code_revision="working-tree",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("100"),
        overall_score=Decimal("75"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("65"),
        risk_class="medium",
        confidence=Decimal("60"),
    )
    generated_at = timezone.now()
    prediction = Prediction.objects.create(
        analysis=analysis,
        listing=listing,
        generated_at=generated_at,
        target_date=timezone.localdate(),
        horizon=Prediction.Horizon.SHORT,
        price_at_prediction=Decimal("100"),
        bear_return=Decimal("-0.02"),
        base_return=Decimal("0.03"),
        bull_return=Decimal("0.07"),
        probability_positive=Decimal("0.62"),
        confidence=Decimal("60"),
        confidence_status="heuristic",
        recommendation=Recommendation.BUY,
        overall_score=Decimal("75"),
        model_version="baseline-v1",
        config_hash="b" * 64,
        data_cutoff=generated_at,
        code_revision="working-tree",
    )

    prediction.base_return = Decimal("0.5")
    with pytest.raises(ValidationError):
        prediction.save()
    with pytest.raises(ValidationError):
        prediction.delete()

    with pytest.raises(DatabaseError), transaction.atomic():
        Prediction.objects.filter(pk=prediction.pk).update(base_return=Decimal("0.5"))

    with pytest.raises(DatabaseError), transaction.atomic():
        Prediction.objects.filter(pk=prediction.pk).delete()
