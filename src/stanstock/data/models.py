from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models.base import ModelBase


class ImmutableEvidenceModel(models.Model):
    class Meta:
        abstract = True

    def save(
        self,
        *,
        force_insert: bool | tuple[ModelBase, ...] = False,
        force_update: bool = False,
        using: str | None = None,
        update_fields: Iterable[str] | None = None,
    ) -> None:
        manager = self._meta.default_manager
        if manager is None:
            raise RuntimeError(f"{type(self).__name__} has no default manager")
        if self.pk and manager.filter(pk=self.pk).exists():
            raise ValidationError(f"{type(self).__name__} records are immutable")
        super().save(
            force_insert=force_insert,
            force_update=force_update,
            using=using,
            update_fields=update_fields,
        )

    def delete(
        self,
        using: Any | None = None,
        keep_parents: bool = False,
    ) -> tuple[int, dict[str, int]]:
        raise ValidationError(f"{type(self).__name__} records are immutable")


class Region(models.TextChoices):
    US = "us", "United States"
    EUROPE = "europe", "Europe"


class Company(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=240)
    country = models.CharField(max_length=2)
    sector = models.CharField(max_length=120, blank=True)
    industry = models.CharField(max_length=160, blank=True)
    lei = models.CharField(max_length=20, blank=True, db_index=True)
    cik = models.CharField(max_length=10, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Security(models.Model):
    class SecurityType(models.TextChoices):
        COMMON_STOCK = "common_stock", "Common stock"
        ADR = "adr", "Depositary receipt"
        ETF = "etf", "Exchange-traded fund"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="securities")
    security_type = models.CharField(
        max_length=24,
        choices=SecurityType,
        default=SecurityType.COMMON_STOCK,
    )
    isin = models.CharField(max_length=12, blank=True, db_index=True)
    name = models.CharField(max_length=240, blank=True)

    class Meta:
        verbose_name_plural = "securities"

    def __str__(self) -> str:
        return self.name or str(self.company)


class Listing(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    security = models.ForeignKey(Security, on_delete=models.CASCADE, related_name="listings")
    ticker = models.CharField(max_length=32)
    exchange_mic = models.CharField(max_length=4)
    provider_symbol = models.CharField(max_length=64, blank=True)
    currency = models.CharField(max_length=3)
    region = models.CharField(max_length=12, choices=Region)
    valid_from = models.DateField(null=True, blank=True)
    valid_to = models.DateField(null=True, blank=True)
    is_primary = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["ticker"]
        constraints = [
            models.UniqueConstraint(
                fields=["ticker", "exchange_mic", "valid_from"],
                name="unique_dated_listing",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(valid_from__isnull=True)
                    | models.Q(valid_to__isnull=True)
                    | models.Q(valid_to__gte=models.F("valid_from"))
                ),
                name="listing_valid_date_range",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.ticker} ({self.exchange_mic})"


class Universe(models.Model):
    slug = models.SlugField(primary_key=True)
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True)
    config_version = models.CharField(max_length=40)

    def __str__(self) -> str:
        return self.name


class UniverseSnapshot(models.Model):
    class Grade(models.TextChoices):
        RESEARCH = "research", "Research-grade reconstruction"
        OBSERVED = "observed", "Observed at run time"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    universe = models.ForeignKey(Universe, on_delete=models.PROTECT, related_name="snapshots")
    as_of_date = models.DateField()
    captured_at = models.DateTimeField(auto_now_add=True)
    grade = models.CharField(max_length=12, choices=Grade)
    config_hash = models.CharField(max_length=64)

    class Meta:
        ordering = ["-as_of_date", "-captured_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["universe", "as_of_date", "grade"],
                name="unique_universe_snapshot",
            )
        ]

    def __str__(self) -> str:
        return f"{self.universe_id}:{self.as_of_date}:{self.grade}"


class UniverseMembership(models.Model):
    snapshot = models.ForeignKey(
        UniverseSnapshot,
        on_delete=models.CASCADE,
        related_name="memberships",
    )
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    eligible = models.BooleanField(default=True)
    exclusion_reason = models.CharField(max_length=160, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["snapshot", "listing"],
                name="unique_universe_membership",
            )
        ]

    def __str__(self) -> str:
        return f"{self.snapshot_id}:{self.listing_id}"


class ProviderRecord(models.Model):
    provider = models.CharField(max_length=40, primary_key=True)
    enabled = models.BooleanField(default=False)
    terms_url = models.URLField(blank=True)
    terms_checked_at = models.DateTimeField(null=True, blank=True)
    usage_scope = models.CharField(max_length=120, blank=True)
    status = models.CharField(max_length=32, default="not_configured")
    last_success_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    def __str__(self) -> str:
        return self.provider


class DataAsset(ImmutableEvidenceModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    provider = models.CharField(max_length=40)
    kind = models.CharField(max_length=40)
    subject = models.CharField(max_length=120)
    relative_path = models.CharField(max_length=500, unique=True)
    sha256 = models.CharField(max_length=64, db_index=True)
    retrieved_at = models.DateTimeField()
    available_at = models.DateTimeField()
    period_start = models.DateField(null=True, blank=True)
    period_end = models.DateField(null=True, blank=True)
    schema_version = models.CharField(max_length=32, default="1")
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-available_at", "-retrieved_at"]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(available_at__lte=models.F("retrieved_at")),
                name="asset_available_before_retrieval",
            )
        ]
        indexes = [
            models.Index(
                fields=["provider", "kind", "subject", "available_at"],
                name="asset_asof_lookup",
            )
        ]

    def __str__(self) -> str:
        return f"{self.provider}:{self.kind}:{self.subject}:{self.available_at.isoformat()}"


#: Name of the uniqueness constraint that makes one observation instant
#: name exactly one content. Referenced by ingestion so a racing insert
#: can be told apart from an unrelated integrity fault.
OBSERVATION_INSTANT_CONSTRAINT = "unique_source_observation_instant"


class SourceObservationEvent(ImmutableEvidenceModel):
    """Append-only record that exact provider bytes were observed at a time.

    A `DataAsset` is content-addressed: when a provider serves bytes that are
    identical to an earlier response, ingestion deliberately reuses the
    existing asset row instead of storing a duplicate. That reuse makes
    ``DataAsset.retrieved_at`` the time the content was *first* seen, which is
    the wrong clock for a later observation of the same bytes.

    The concrete failure this exists to prevent is a content reversion: a
    value that goes 100 -> 101 -> 100 restates back to bytes already on file,
    so the third revision would otherwise inherit the *first* retrieval time
    and appear knowable months before it was actually observed.

    Each retrieval therefore appends its own event. Events are immutable and
    unique on ``(provider, kind, subject, observed_at)`` -- deliberately
    *without* the digest -- so the database itself enforces that one
    observation instant names exactly one content. Two concurrent writers
    claiming the same instant cannot both succeed: the loser sees the
    committed row and either succeeds idempotently (same digest) or fails
    explicitly (different digest), before anything is normalized. A
    sequential read-then-write check could not provide that, and a
    ``select_for_update()`` on a row that does not exist yet locks nothing.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    provider = models.CharField(max_length=40)
    kind = models.CharField(max_length=40)
    subject = models.CharField(max_length=120)
    content_sha256 = models.CharField(max_length=64)
    source_asset = models.ForeignKey(
        DataAsset,
        on_delete=models.PROTECT,
        related_name="observation_events",
    )
    observed_at = models.DateTimeField()
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["provider", "kind", "subject", "observed_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["provider", "kind", "subject", "observed_at"],
                name=OBSERVATION_INSTANT_CONSTRAINT,
            )
        ]
        indexes = [
            models.Index(
                fields=["source_asset", "observed_at"],
                name="observation_asset_lookup",
            )
        ]

    def __str__(self) -> str:
        return f"{self.provider}:{self.kind}:{self.subject}:{self.observed_at.isoformat()}"


class FundamentalFact(ImmutableEvidenceModel):
    class PeriodType(models.TextChoices):
        INSTANT = "instant", "Instant"
        DURATION = "duration", "Duration"
        UNCLASSIFIED = "unclassified", "Unclassified"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="facts")
    provider = models.CharField(max_length=40)
    concept = models.CharField(max_length=100)
    taxonomy = models.CharField(max_length=40, blank=True)
    source_concept = models.CharField(max_length=240)
    value = models.DecimalField(max_digits=32, decimal_places=8)
    unit = models.CharField(max_length=24)
    currency = models.CharField(max_length=3, blank=True)
    period_type = models.CharField(
        max_length=16,
        choices=PeriodType,
        default=PeriodType.UNCLASSIFIED,
    )
    period_identity = models.CharField(max_length=160)
    period_start = models.DateField(null=True, blank=True)
    period_end = models.DateField()
    fiscal_year = models.IntegerField(null=True, blank=True)
    fiscal_period = models.CharField(max_length=8, blank=True)
    frame = models.CharField(max_length=32, blank=True)
    accession = models.CharField(max_length=80)
    filing_form = models.CharField(max_length=16, blank=True)
    filing_date = models.DateField(null=True, blank=True)
    filed_at = models.DateTimeField(null=True, blank=True)
    acceptance_at = models.DateTimeField(null=True, blank=True)
    available_at = models.DateTimeField()
    availability_basis = models.CharField(max_length=40, default="legacy")
    ingested_at = models.DateTimeField(auto_now_add=True)
    is_amendment = models.BooleanField(default=False)
    source_revision = models.PositiveIntegerField(default=1)
    observation_hash = models.CharField(max_length=64)
    quality_flags = models.JSONField(default=list, blank=True)
    source_asset = models.ForeignKey(DataAsset, on_delete=models.PROTECT)

    class Meta:
        ordering = ["company", "concept", "-available_at"]
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "company",
                    "provider",
                    "source_concept",
                    "period_identity",
                    "accession",
                    "unit",
                    "source_revision",
                ],
                name="unique_fundamental_vintage",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(filed_at__isnull=True)
                    | models.Q(available_at__gte=models.F("filed_at"))
                ),
                name="fact_available_after_filing",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(acceptance_at__isnull=True)
                    | models.Q(available_at__gte=models.F("acceptance_at"))
                ),
                name="fact_available_after_acceptance",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(period_type="unclassified")
                    | models.Q(
                        period_type="instant",
                        period_start__isnull=True,
                    )
                    | models.Q(
                        period_type="duration",
                        period_start__isnull=False,
                        period_start__lte=models.F("period_end"),
                    )
                ),
                name="fact_period_type_consistent",
            ),
            models.CheckConstraint(
                condition=~models.Q(period_identity="") & ~models.Q(observation_hash=""),
                name="fact_identity_present",
            ),
        ]
        indexes = [
            models.Index(
                fields=["company", "concept", "available_at"],
                name="fact_asof_lookup",
            )
        ]

    def __str__(self) -> str:
        return f"{self.company_id}:{self.concept}:{self.period_end}:{self.accession}"

    def save(
        self,
        *,
        force_insert: bool | tuple[ModelBase, ...] = False,
        force_update: bool = False,
        using: str | None = None,
        update_fields: Iterable[str] | None = None,
    ) -> None:
        from stanstock.data.fact_identity import (
            build_observation_hash,
            build_period_identity,
        )

        if not self.period_identity:
            if self.period_type == self.PeriodType.UNCLASSIFIED:
                self.period_type = (
                    self.PeriodType.DURATION
                    if self.period_start is not None
                    else self.PeriodType.INSTANT
                )
            self.period_identity = build_period_identity(
                period_type=self.period_type,
                period_start=self.period_start,
                period_end=self.period_end,
                fiscal_period=self.fiscal_period,
                frame=self.frame,
            )
        if not self.observation_hash:
            self.observation_hash = build_observation_hash(
                taxonomy=self.taxonomy,
                source_concept=self.source_concept,
                value=self.value,
                unit=self.unit,
                currency=self.currency,
                period_identity=self.period_identity,
                fiscal_year=self.fiscal_year,
                fiscal_period=self.fiscal_period,
                accession=self.accession,
                filing_form=self.filing_form,
                filing_date=self.filing_date,
                acceptance_at=self.acceptance_at,
                frame=self.frame,
            )
        super().save(
            force_insert=force_insert,
            force_update=force_update,
            using=using,
            update_fields=update_fields,
        )


class FundamentalFactEvidence(ImmutableEvidenceModel):
    class Role(models.TextChoices):
        FILING = "filing", "Filing availability"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    fact = models.ForeignKey(
        FundamentalFact,
        on_delete=models.PROTECT,
        related_name="evidence_links",
    )
    role = models.CharField(max_length=24, choices=Role.choices)
    source_asset = models.ForeignKey(DataAsset, on_delete=models.PROTECT)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["fact", "role", "created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["fact", "role"],
                name="unique_fundamental_fact_evidence_role",
            )
        ]

    def __str__(self) -> str:
        return f"{self.fact_id}:{self.role}:{self.source_asset_id}"


class CompanyClassificationObservation(ImmutableEvidenceModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(
        Company,
        on_delete=models.CASCADE,
        related_name="classification_observations",
    )
    provider = models.CharField(max_length=40)
    scheme = models.CharField(max_length=32)
    code = models.CharField(max_length=32)
    description = models.CharField(max_length=240, blank=True)
    observed_at = models.DateTimeField()
    available_at = models.DateTimeField()
    ingested_at = models.DateTimeField(auto_now_add=True)
    accession = models.CharField(max_length=80, blank=True)
    quality_flags = models.JSONField(default=list, blank=True)
    source_asset = models.ForeignKey(DataAsset, on_delete=models.PROTECT)

    class Meta:
        ordering = ["company", "scheme", "-available_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["company", "provider", "scheme", "code", "source_asset"],
                name="unique_company_classification_observation",
            ),
            models.CheckConstraint(
                condition=models.Q(available_at__gte=models.F("observed_at")),
                name="classification_available_after_observed",
            ),
        ]
        indexes = [
            models.Index(
                fields=["company", "scheme", "available_at"],
                name="classification_asof_lookup",
            )
        ]

    def __str__(self) -> str:
        return f"{self.company_id}:{self.scheme}:{self.code}:{self.available_at.isoformat()}"


class FxRate(ImmutableEvidenceModel):
    base_currency = models.CharField(max_length=3)
    quote_currency = models.CharField(max_length=3)
    observation_date = models.DateField()
    value = models.DecimalField(max_digits=20, decimal_places=8)
    published_at = models.DateTimeField()
    available_at = models.DateTimeField()
    source_asset = models.ForeignKey(DataAsset, on_delete=models.PROTECT)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["base_currency", "quote_currency", "observation_date", "available_at"],
                name="unique_fx_vintage",
            ),
            models.CheckConstraint(
                condition=models.Q(value__gt=0),
                name="fx_rate_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(available_at__gte=models.F("published_at")),
                name="fx_available_after_publication",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.base_currency}/{self.quote_currency}:{self.observation_date}:{self.value}"


class LatestMarketData(models.Model):
    listing = models.OneToOneField(
        Listing,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="latest_market_data",
    )
    observed_at = models.DateTimeField()
    session_date = models.DateField()
    close = models.DecimalField(max_digits=20, decimal_places=6)
    previous_close = models.DecimalField(max_digits=20, decimal_places=6, null=True)
    volume = models.BigIntegerField(null=True)
    source_asset = models.ForeignKey(DataAsset, on_delete=models.PROTECT)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=models.Q(close__gt=0),
                name="latest_market_close_positive",
            ),
            models.CheckConstraint(
                condition=(models.Q(previous_close__isnull=True) | models.Q(previous_close__gt=0)),
                name="latest_market_previous_close_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(volume__isnull=True) | models.Q(volume__gte=0),
                name="latest_market_volume_nonnegative",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.listing_id}:{self.session_date.isoformat()}:{self.close}"
