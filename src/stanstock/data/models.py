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


class FundamentalFact(ImmutableEvidenceModel):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name="facts")
    provider = models.CharField(max_length=40)
    concept = models.CharField(max_length=100)
    source_concept = models.CharField(max_length=240)
    value = models.DecimalField(max_digits=32, decimal_places=8)
    unit = models.CharField(max_length=24)
    currency = models.CharField(max_length=3, blank=True)
    period_start = models.DateField(null=True, blank=True)
    period_end = models.DateField()
    fiscal_year = models.IntegerField(null=True, blank=True)
    fiscal_period = models.CharField(max_length=8, blank=True)
    accession = models.CharField(max_length=80)
    filed_at = models.DateTimeField(null=True, blank=True)
    available_at = models.DateTimeField()
    ingested_at = models.DateTimeField(auto_now_add=True)
    is_amendment = models.BooleanField(default=False)
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
                    "period_end",
                    "accession",
                    "unit",
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
        ]
        indexes = [
            models.Index(
                fields=["company", "concept", "available_at"],
                name="fact_asof_lookup",
            )
        ]

    def __str__(self) -> str:
        return f"{self.company_id}:{self.concept}:{self.period_end}:{self.accession}"


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
        return f"{self.listing_id}:{self.observed_at.isoformat()}:{self.close}"
