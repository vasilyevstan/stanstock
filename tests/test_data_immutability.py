from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from django.core.exceptions import ValidationError
from django.db import DatabaseError, transaction
from django.utils import timezone

from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    FxRate,
    SourceObservationEvent,
)


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


@pytest.mark.django_db
def test_company_classification_is_immutable_at_model_and_database_layers(
    evidence_records: tuple[DataAsset, FundamentalFact, FxRate],
) -> None:
    asset, fact, _rate = evidence_records
    observation = CompanyClassificationObservation.objects.create(
        company=fact.company,
        provider="sec",
        scheme="sec_sic",
        code="3571",
        description="Electronic Computers",
        observed_at=asset.available_at,
        available_at=asset.available_at,
        source_asset=asset,
    )

    with pytest.raises(ValidationError, match="immutable"):
        observation.save()
    with pytest.raises(DatabaseError, match="immutable"):
        with transaction.atomic():
            CompanyClassificationObservation.objects.filter(pk=observation.pk).delete()


@pytest.mark.django_db
def test_fundamental_fact_evidence_is_immutable_at_model_and_database_layers(
    evidence_records: tuple[DataAsset, FundamentalFact, FxRate],
) -> None:
    asset, fact, _rate = evidence_records
    evidence = FundamentalFactEvidence.objects.create(
        fact=fact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=asset,
    )

    with pytest.raises(ValidationError, match="immutable"):
        evidence.save()
    with pytest.raises(DatabaseError, match="immutable"):
        with transaction.atomic():
            FundamentalFactEvidence.objects.filter(pk=evidence.pk).delete()


@pytest.mark.django_db
def test_source_observation_event_is_immutable_at_model_and_database_layers(
    evidence_records: tuple[DataAsset, FundamentalFact, FxRate],
) -> None:
    """Observation events are the clock corrections bind to, so they are fixed.

    A mutable event would let a correction's proven availability be moved
    after the fact, which is exactly the look-ahead the event exists to
    close. The bulk paths are covered too, because `Model.save()` guards do
    not run for queryset `update()`/`delete()`.
    """
    asset, _fact, _rate = evidence_records
    event = SourceObservationEvent.objects.create(
        provider="sec",
        kind="sec_companyfacts",
        subject="0000320193",
        content_sha256="a" * 64,
        source_asset=asset,
        observed_at=datetime(2026, 10, 18, 12, tzinfo=UTC),
    )

    with pytest.raises(ValidationError, match="immutable"):
        event.save()
    with pytest.raises(DatabaseError, match="immutable"):
        with transaction.atomic():
            SourceObservationEvent.objects.filter(pk=event.pk).update(
                observed_at=datetime(2026, 8, 15, 12, tzinfo=UTC)
            )
    with pytest.raises(DatabaseError, match="immutable"):
        with transaction.atomic():
            SourceObservationEvent.objects.filter(pk=event.pk).delete()

    # The same retrieval recorded again is one row, not a duplicate.
    again, created = SourceObservationEvent.objects.get_or_create(
        provider="sec",
        kind="sec_companyfacts",
        subject="0000320193",
        content_sha256="a" * 64,
        observed_at=datetime(2026, 10, 18, 12, tzinfo=UTC),
        defaults={"source_asset": asset},
    )
    assert created is False
    assert again.pk == event.pk
