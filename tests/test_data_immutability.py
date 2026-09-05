from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from django.core.exceptions import ValidationError
from django.db import DatabaseError, transaction
from django.utils import timezone

from stanstock.data.models import Company, DataAsset, FundamentalFact, FxRate


@pytest.fixture
def evidence_records() -> tuple[DataAsset, FundamentalFact, FxRate]:
    now = timezone.now()
    asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="fundamentals",
        subject="immutable-test",
        relative_path="tests/immutable-evidence.json",
        sha256="a" * 64,
        retrieved_at=now,
        available_at=now,
    )
    company = Company.objects.create(name="Immutable Test", country="US")
    fact = FundamentalFact.objects.create(
        company=company,
        provider="synthetic_demo",
        concept="revenue",
        source_concept="revenue",
        value=Decimal("100"),
        unit="USD",
        currency="USD",
        period_end=date(2025, 12, 31),
        accession="immutable-test-v1",
        available_at=now,
        source_asset=asset,
    )
    rate = FxRate.objects.create(
        base_currency="EUR",
        quote_currency="USD",
        observation_date=date(2025, 12, 31),
        value=Decimal("1.10"),
        published_at=now,
        available_at=now,
        source_asset=asset,
    )
    return asset, fact, rate


@pytest.mark.django_db
def test_evidence_models_reject_instance_updates_and_deletes(
    evidence_records: tuple[DataAsset, FundamentalFact, FxRate],
) -> None:
    for record in evidence_records:
        with pytest.raises(ValidationError, match="immutable"):
            record.save()
        with pytest.raises(ValidationError, match="immutable"):
            record.delete()


@pytest.mark.django_db
def test_database_triggers_reject_bulk_evidence_mutation(
    evidence_records: tuple[DataAsset, FundamentalFact, FxRate],
) -> None:
    asset, fact, rate = evidence_records

    with pytest.raises(DatabaseError, match="immutable"):
        with transaction.atomic():
            DataAsset.objects.filter(pk=asset.pk).update(subject="changed")
    with pytest.raises(DatabaseError, match="immutable"):
        with transaction.atomic():
            FundamentalFact.objects.filter(pk=fact.pk).delete()
    with pytest.raises(DatabaseError, match="immutable"):
        with transaction.atomic():
            FxRate.objects.filter(pk=rate.pk).update(value=Decimal("1.20"))
